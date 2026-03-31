# Joint Training: FRCRN + AD分类器

## 为什么做这个

当前pipeline是分离的：MossFormer离线降噪 → XLSR特征提取 → AD分类器训练。降噪器不知道下游任务是什么，可能会丢失对AD分类有用的语音特征。Joint training让FRCRN降噪器同时被降噪质量和AD分类准确率两个目标优化，产出更适合AD检测的降噪结果。

## 核心架构

```
Raw Pitt Audio (B, T)  T = 60s × 16kHz = 960,000
       ↓
 [FRCRN 降噪器]
   stft, unet1        ← frozen (基础降噪能力已学会)
   unet2               ← 可训练 (~7M params, 残差精修层)
   istft
       ↓
  denoised (B, T) ───→ L1_loss(denoised, MossFormer_pseudo_clean)
       ↓
 [XLSR-53]
   前21层 transformer   ← frozen + gradient checkpointing (不存激活，反向时重算)
   后3层 transformer    ← 可训练 (~37.5M params, 适配降噪后音频分布)
       ↓
  features (B, 3000, 1024)
       ↓
 [AD_XLSR_Model]       ← 可训练 (~35K params)
       ↓
  logits (B, 2)   ───→ CE_loss(logits, label)

L_total = α * L_denoise + β * L_classify
L_total.backward()  → 梯度端到端回传
```

## 冻结与训练策略

### 各组件状态

| 组件 | 参数量 | 状态 | 说明 |
|------|--------|------|------|
| FRCRN stft/istft | ~0.8M | frozen | 固定STFT核 |
| FRCRN unet1 | ~6.2M | frozen | 基础降噪能力已学会，保持不动 |
| FRCRN unet2 | ~6.2M | **训练** | 残差精修层，调整降噪方向适配AD分类 |
| XLSR 前21层 | ~262M | frozen + **gradient checkpointing** | 不存激活值，反向传播时重新forward算出来，省显存 |
| XLSR 后3层 | ~37.5M | **训练** | 让特征表示适配降噪后的音频分布 |
| AD_XLSR_Model | ~35K | **训练** | 分类头 |

### 为什么这样冻结

- **unet2 是残差层**：`cmp_mask2 = tanh(unet2(unet1_out)) + cmp_mask1`，训练它不会破坏unet1的基础降噪能力，只调整精修方向
- **XLSR 后3层 finetune**：后面的层更接近任务相关的语义表示，finetune 能让特征适配降噪后的音频分布
- **XLSR 前21层用 gradient checkpointing**：这21层权重不更新，但梯度要穿过它们回传到FRCRN。正常情况 PyTorch 需要存每层中间激活值用于反向传播链式法则计算。gradient checkpointing 不存这些激活，反向到某层时重新跑一遍该层 forward 临时算出来，用完丢掉。代价是慢约30%，但显存从存21层激活降到峰值只存1层

### 显存估算 (60s, B=2, AMP float16)

| 项目 | 显存 |
|------|------|
| XLSR frozen 权重 (300M × 2B) | ~600MB |
| FRCRN 权重 (14M × 2B) | ~28MB |
| Optimizer states (unet2 + XLSR 3层 + AD) × 8B | ~350MB |
| XLSR 可训练3层激活 | ~1.7GB |
| XLSR frozen 21层 gradient checkpointing 峰值 | ~0.6GB |
| FRCRN forward 激活 | ~2GB |
| 梯度 + 框架开销 | ~2-3GB |
| **合计** | **~7-8GB** |

RTX 5090 32GB 绰绰有余。

## Joint Training 步骤

### Step 1: FRCRN模型 (已完成)

FRCRN 源码已从 ModelScope 复制到项目中，去掉了所有 modelscope 依赖。

```
ad_detection/joint_train/frcrn/
├── __init__.py              ← from .frcrn import FRCRN
├── frcrn.py                 ← FRCRN 主模型 (nn.Module, ~14M params)
├── conv_stft.py             ← ConvSTFT / ConviSTFT
├── unet.py                  ← UNet
├── complex_nn.py            ← ComplexConv2d / ComplexBatchNorm2d / ComplexUniDeepFsmn
├── se_module_complex.py     ← SELayer
└── layers/
    ├── __init__.py
    ├── layer_base.py
    └── uni_deep_fsmn.py
```

### Step 2: 修改 SSLModel 支持部分 finetune + gradient checkpointing

**文件**: `ad_detection/train/model.py` — 需要修改 `SSLModel`

```python
class SSLModel(nn.Module):
    def __init__(self, device, freeze_xlsr=True, finetune_last_n=0, ...):
        # ... 加载模型 ...

        # 冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻最后 finetune_last_n 层
        if finetune_last_n > 0:
            for layer in self.model.encoder.layers[-finetune_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def extract_feat(self, input_data, use_checkpoint=False):
        if use_checkpoint:
            # 对 frozen 层用 gradient checkpointing
            # 对 trainable 层正常 forward
            ...
        else:
            # 原有逻辑不变
            model_output = self.model(input_tmp, mask=False, features_only=True)
```

### Step 3: 构建数据加载器

**文件**: `ad_detection/joint_train/joint_dataset.py`

```python
class JointTrainingDataset(Dataset):
    # 每个sample:
    #   raw_path:   data/raw/Pitt/{Control,Dementia}/{session_id}.wav
    #   clean_path: data/denoised/Pitt-MossFormer/{Control,Dementia}/{session_id}.wav
    #   label:      0 (Control) / 1 (Dementia)
    #
    # 音频处理: librosa.load(sr=16000) → pad/截断到 60秒
    # 复用现有 Pitt CSV 的 session_id 和 label

def joint_collate_fn(batch):
    # padding到batch内最长 + 生成attention mask
    # return: raw_padded(B,T), clean_padded(B,T), labels(B,), lengths(B,)
```

### Step 4: 组装Joint模型

**文件**: `ad_detection/joint_train/joint_model.py`

```python
class JointDenoiseADModel(nn.Module):
    def __init__(self, frcrn, xlsr_model, ad_model):
        self.frcrn = frcrn           # unet2 可训练
        self.xlsr_model = xlsr_model # 后3层可训练 + 前21层gradient checkpointing
        self.ad_model = ad_model     # 可训练

    def forward(self, raw_audio, lengths):
        out_list = self.frcrn(raw_audio)
        denoised = out_list[4]                # wav_l2 (B, T)

        xlsr_feat, _ = self.xlsr_model.extract_feat(denoised, use_checkpoint=True)

        feat_lengths = lengths // 320
        mask = create_length_mask(feat_lengths, xlsr_feat.shape[1])
        logits = self.ad_model(xlsr_feat, mask)

        return denoised, logits
```

### Step 5: 训练配置

**文件**: `ad_detection/joint_train/joint_config.py`

```python
# RTX 5090 32GB, 60秒音频
JOINT_SECOND_LENGTH = 60
JOINT_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16  # 有效batch = 32
USE_AMP = True

XLSR_FINETUNE_LAST_N = 3

# Joint Training (无预热，直接联合训练)
MAX_EPOCHS = 50
FRCRN_LR = 1e-5           # unet2
XLSR_LR = 1e-5            # 后3层
AD_LR = 3e-3              # 分类器
WEIGHT_DECAY = 1e-2

ALPHA = 1.0               # L_denoise 权重
BETA = 1.0                # L_classify 权重

PATIENCE = 10
RANDOM_SEEDS = [21, 42, 84, 168, 336]
```

### Step 6: Joint Training循环

**文件**: `ad_detection/joint_train/joint_train.py`

```python
# 三组参数，三个学习率
optimizer = AdamW([
    {'params': frcrn.unet2.parameters(), 'lr': 1e-5},
    {'params': xlsr_last3_params, 'lr': 1e-5},
    {'params': ad_model.parameters(), 'lr': 3e-3},
], weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

for epoch in range(MAX_EPOCHS):
    for i, (raw, clean, labels, lengths) in enumerate(train_loader):
        with torch.cuda.amp.autocast():
            denoised, logits = joint_model(raw, lengths)
            loss_denoise = F.l1_loss(denoised, clean)
            loss_classify = F.cross_entropy(logits, labels, weight=class_weights)
            loss_total = ALPHA * loss_denoise + BETA * loss_classify

        scaler.scale(loss_total / GRADIENT_ACCUMULATION_STEPS).backward()

        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # Validate, early stopping, save best model
```

### Step 7: Notebook入口

**文件**: `ad_detection/joint_train/train_notebook/joint-training-Pitt.ipynb`

## 关键代码路径

| 现有文件 | 作用 | 是否修改 |
|---------|------|---------|
| `ad_detection/train/model.py` | SSLModel, AD_XLSR_Model | **需修改**: SSLModel 增加 finetune_last_n 和 gradient checkpointing |
| `ad_detection/train/config.py` | 现有超参数 | **不修改** |
| `ad_detection/train/dataset.py` | 现有FeatureDataset | **不修改**, 参考其collate |
| `ad_detection/train/train.py` | 现有训练循环 | **不修改**, 参考其validate/early stopping |
| `ad_detection/data/raw/Pitt/` | 原始Pitt音频 | Joint training输入 |
| `ad_detection/data/denoised/Pitt-MossFormer/` | MossFormer降噪音频 | Pseudo clean target |

| 已有文件 | 状态 |
|--------|------|
| `ad_detection/joint_train/frcrn/` | **已完成** |

| 待创建文件 | 作用 |
|-----------|------|
| `ad_detection/joint_train/joint_config.py` | Joint training超参数 |
| `ad_detection/joint_train/joint_dataset.py` | 三元组数据加载 |
| `ad_detection/joint_train/joint_model.py` | JointDenoiseADModel |
| `ad_detection/joint_train/joint_train.py` | 训练循环 |
| `ad_detection/joint_train/train_notebook/joint-training-Pitt.ipynb` | 实验Notebook |
