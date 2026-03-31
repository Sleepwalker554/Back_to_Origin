import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from joint_config import (
    MAX_EPOCHS, WEIGHT_DECAY, ETA_MIN, ALPHA, BETA,
    FRCRN_LR, XLSR_LR, AD_LR, PHASE1_GRAD_ACCUM, PHASE2_GRAD_ACCUM,
    USE_AMP, XLSR_FINETUNE_LAST_N, USE_GRADIENT_CHECKPOINT,
    XLSR_MAX_TIME_STEPS, PATIENCE, WARMUP_EPOCHS, AD_DROPOUT,
)
from joint_model import JointDenoiseADModel, JointFRCRN, JointSSLModel, AD_XLSR_Model


def build_joint_model(device, frcrn_pretrained_path=None):
    """构建 Joint Training 模型"""
    frcrn_model = JointFRCRN(pretrained_path=frcrn_pretrained_path).to(device)
    xlsr_model = JointSSLModel(
        device,
        finetune_last_n=XLSR_FINETUNE_LAST_N,
        use_checkpoint=USE_GRADIENT_CHECKPOINT,
    )
    ad_model = AD_XLSR_Model(dropout=AD_DROPOUT).to(device)

    joint_model = JointDenoiseADModel(
        frcrn_model, xlsr_model, ad_model,
        xlsr_max_time_steps=XLSR_MAX_TIME_STEPS,
    ).to(device)

    return joint_model


def build_phase1_optimizer(joint_model):
    """Phase 1: 只训练 AD 分类器"""
    param_groups = [{
        'params': joint_model.ad_model.parameters(),
        'lr': AD_LR,
        'name': 'ad_classifier',
    }]

    for pg in param_groups:
        n = sum(p.numel() for p in pg['params'])
        print(f"  Phase 1 - Optimizer group '{pg['name']}': {n:,} params, lr={pg['lr']}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=WARMUP_EPOCHS, eta_min=ETA_MIN)

    return optimizer, scheduler


def build_phase2_optimizer(joint_model):
    """Phase 2: AD 分类器 + FRCRN unet2 (+ 可选 XLSR finetune)"""
    param_groups = []

    frcrn_params = [p for p in joint_model.frcrn_model.parameters() if p.requires_grad]
    if frcrn_params:
        param_groups.append({
            'params': frcrn_params,
            'lr': FRCRN_LR,
            'name': 'frcrn_unet2',
        })

    xlsr_params = [p for p in joint_model.xlsr_model.parameters() if p.requires_grad]
    if xlsr_params:
        param_groups.append({
            'params': xlsr_params,
            'lr': XLSR_LR,
            'name': 'xlsr_finetune',
        })

    param_groups.append({
        'params': joint_model.ad_model.parameters(),
        'lr': AD_LR * 0.1,  # Phase 2 分类器用更小的 lr，避免破坏已学到的东西
        'name': 'ad_classifier',
    })

    for pg in param_groups:
        n = sum(p.numel() for p in pg['params'])
        print(f"  Phase 2 - Optimizer group '{pg['name']}': {n:,} params, lr={pg['lr']}")

    phase2_epochs = MAX_EPOCHS - WARMUP_EPOCHS
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=ETA_MIN)

    return optimizer, scheduler


def train_one_epoch(joint_model, train_loader, optimizer, device,
                    epoch=None, class_weights=None, phase=1, grad_accum=16):
    """
    phase=1: FRCRN 冻结, 只有 classify loss
    phase=2: FRCRN 解冻, denoise + classify loss
    """
    joint_model.ad_model.train()
    if phase == 2:
        joint_model.frcrn_model.train()
    else:
        joint_model.frcrn_model.eval()

    total_loss = 0
    total_denoise_loss = 0
    total_classify_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    desc = f"Epoch {epoch} [Phase {phase}] - Training" if epoch is not None else "Training"
    pbar = tqdm(train_loader, desc=desc, leave=False)

    for i, (raw, clean, labels) in enumerate(pbar):
        raw = raw.to(device)
        clean = clean.to(device)
        labels = labels.to(device)

        if phase == 1:
            # Phase 1: FRCRN+XLSR 不需要梯度, 省显存省时间
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_AMP):
                    denoised, _ = joint_model.frcrn_model(raw)
                    xlsr_feat = joint_model.xlsr_model.extract_feat(denoised)
                    # pad/truncate + mask
                    seq_len = xlsr_feat.shape[1]
                    if seq_len > joint_model.xlsr_max_time_steps:
                        xlsr_feat = xlsr_feat[:, :joint_model.xlsr_max_time_steps, :]
                        mask = torch.ones(xlsr_feat.shape[0], joint_model.xlsr_max_time_steps,
                                          device=xlsr_feat.device)
                    elif seq_len < joint_model.xlsr_max_time_steps:
                        pad_len = joint_model.xlsr_max_time_steps - seq_len
                        xlsr_feat = F.pad(xlsr_feat, (0, 0, 0, pad_len))
                        mask = torch.ones(xlsr_feat.shape[0], joint_model.xlsr_max_time_steps,
                                          device=xlsr_feat.device)
                        mask[:, seq_len:] = 0
                    else:
                        mask = torch.ones(xlsr_feat.shape[0], seq_len, device=xlsr_feat.device)

            # 只有分类 loss, 只有 AD 分类器有梯度
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_AMP):
                logits = joint_model.ad_model(xlsr_feat, mask)
                loss_classify = F.cross_entropy(logits, labels, weight=class_weights)
                loss_total = BETA * loss_classify
                loss_denoise = F.l1_loss(denoised, clean)  # 仅监控

            loss_scaled = loss_total / grad_accum
            loss_scaled.backward()

        else:
            # Phase 2: 端到端, FRCRN + AD 都有梯度
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_AMP):
                denoised, logits = joint_model(raw)
                loss_denoise = F.l1_loss(denoised, clean)
                loss_classify = F.cross_entropy(logits, labels, weight=class_weights)
                loss_total = ALPHA * loss_denoise + BETA * loss_classify

            loss_scaled = loss_total / grad_accum
            loss_scaled.backward()

        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in joint_model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # 统计
        total_loss += loss_total.item()
        total_denoise_loss += loss_denoise.item()
        total_classify_loss += loss_classify.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss / (i + 1):.4f}',
            'denoise': f'{total_denoise_loss / (i + 1):.4f}',
            'cls': f'{total_classify_loss / (i + 1):.4f}',
            'acc': f'{correct / total:.4f}',
        })

    # 处理尾部不足 grad_accum 的梯度
    if (i + 1) % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in joint_model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'denoise_loss': total_denoise_loss / n_batches,
        'classify_loss': total_classify_loss / n_batches,
        'accuracy': correct / total,
    }


def validate(joint_model, val_loader, device, epoch=None, class_weights=None):
    joint_model.eval()

    total_loss = 0
    total_denoise_loss = 0
    total_classify_loss = 0
    correct = 0
    total = 0
    control_correct = 0
    control_total = 0
    dementia_correct = 0
    dementia_total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    desc = f"Epoch {epoch} - Validation" if epoch is not None else "Validation"
    pbar = tqdm(val_loader, desc=desc, leave=False)

    with torch.no_grad():
        for raw, clean, labels in pbar:
            raw = raw.to(device)
            clean = clean.to(device)
            labels = labels.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_AMP):
                denoised, logits = joint_model(raw)
                loss_denoise = F.l1_loss(denoised, clean)
                loss_classify = F.cross_entropy(logits, labels, weight=class_weights)
                loss_total = ALPHA * loss_denoise + BETA * loss_classify

            predictions = torch.argmax(logits, dim=1)

            total_loss += loss_total.item()
            total_denoise_loss += loss_denoise.item()
            total_classify_loss += loss_classify.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for pred, label in zip(predictions, labels):
                if label == 0:
                    control_total += 1
                    if pred == label:
                        control_correct += 1
                else:
                    dementia_total += 1
                    if pred == label:
                        dementia_correct += 1
                if pred == 1 and label == 1:
                    true_positives += 1
                elif pred == 1 and label == 0:
                    false_positives += 1
                elif pred == 0 and label == 1:
                    false_negatives += 1

            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{correct / total:.4f}',
            })

    n_batches = len(val_loader)
    precision = true_positives / (true_positives + false_positives) \
        if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) \
        if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0

    return {
        'loss': total_loss / n_batches,
        'denoise_loss': total_denoise_loss / n_batches,
        'classify_loss': total_classify_loss / n_batches,
        'accuracy': correct / total,
        'control_acc': control_correct / control_total if control_total > 0 else 0,
        'dementia_acc': dementia_correct / dementia_total if dementia_total > 0 else 0,
        'f1_score': f1_score,
    }


def train(seed, phase1_train_loader, phase1_val_loader,
          phase2_train_loader, phase2_val_loader,
          output_dir, device,
          frcrn_pretrained_path=None,
          class_weight_control=1.0, class_weight_dementia=1.0):
    """
    Two-phase joint training pipeline

    Phase 1: 冻结 FRCRN, 只训练 AD 分类器 (WARMUP_EPOCHS, batch=32)
    Phase 2: 解冻 FRCRN unet2, 联合训练 (MAX_EPOCHS - WARMUP_EPOCHS, batch=2)

    Returns:
        seed, best_metrics, training_history
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    class_weights = torch.tensor(
        [class_weight_control, class_weight_dementia],
        dtype=torch.float32, device=device,
    )

    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 构建模型
    joint_model = build_joint_model(device, frcrn_pretrained_path)

    # 训练历史
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    epochs_list = []

    # Early stopping (只在 Phase 2 生效)
    best_val_acc = 0
    best_metrics = {}
    patience_counter = 0
    best_epoch = 0
    stopped_epoch = 0

    # ============ Phase 1: 只训练 AD 分类器 ============
    print(f"\n{'='*60}")
    print(f"Phase 1: Training AD classifier only ({WARMUP_EPOCHS} epochs)")
    print(f"{'='*60}")

    # Phase 1 冻结 FRCRN unet2
    for p in joint_model.frcrn_model.parameters():
        p.requires_grad = False

    optimizer1, scheduler1 = build_phase1_optimizer(joint_model)

    for epoch in range(WARMUP_EPOCHS):
        train_metrics = train_one_epoch(
            joint_model, phase1_train_loader, optimizer1, device,
            epoch=epoch + 1, class_weights=class_weights,
            phase=1, grad_accum=PHASE1_GRAD_ACCUM,
        )
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['accuracy'])

        scheduler1.step()

        val_metrics = validate(
            joint_model, phase1_val_loader, device,
            epoch=epoch + 1, class_weights=class_weights,
        )
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])
        epochs_list.append(epoch)

        print(f"Epoch {epoch + 1} [Phase 1]: "
              f"train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} "
              f"val_f1={val_metrics['f1_score']:.4f} "
              f"denoise={val_metrics['denoise_loss']:.4f}")

        # Phase 1 也记录最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            best_metrics = val_metrics.copy()
            torch.save(joint_model.frcrn_model.state_dict(), save_dir / 'frcrn_best.pth')
            torch.save(joint_model.xlsr_model.state_dict(), save_dir / 'xlsr_best.pth')
            torch.save(joint_model.ad_model.state_dict(), save_dir / 'ad_model_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'phase': 1,
                'best_val_acc': best_val_acc,
                'best_metrics': best_metrics,
            }, save_dir / 'meta.pth')

    print(f"\nPhase 1 done: best val_acc={best_val_acc*100:.2f}% at epoch {best_epoch}")

    # ============ Phase 2: 联合训练 ============
    print(f"\n{'='*60}")
    print(f"Phase 2: Joint training FRCRN + AD ({MAX_EPOCHS - WARMUP_EPOCHS} epochs)")
    print(f"{'='*60}")

    # 解冻 FRCRN unet2
    for p in joint_model.frcrn_model.frcrn.unet2.parameters():
        p.requires_grad = True

    optimizer2, scheduler2 = build_phase2_optimizer(joint_model)

    # Phase 2 early stopping 从 Phase 1 最佳 val_acc 开始
    patience_counter = 0

    for epoch in range(WARMUP_EPOCHS, MAX_EPOCHS):
        train_metrics = train_one_epoch(
            joint_model, phase2_train_loader, optimizer2, device,
            epoch=epoch + 1, class_weights=class_weights,
            phase=2, grad_accum=PHASE2_GRAD_ACCUM,
        )
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['accuracy'])

        scheduler2.step()

        val_metrics = validate(
            joint_model, phase2_val_loader, device,
            epoch=epoch + 1, class_weights=class_weights,
        )
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])
        epochs_list.append(epoch)

        print(f"Epoch {epoch + 1} [Phase 2]: "
              f"train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} "
              f"val_f1={val_metrics['f1_score']:.4f} "
              f"denoise={val_metrics['denoise_loss']:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            best_metrics = val_metrics.copy()
            patience_counter = 0
            torch.save(joint_model.frcrn_model.state_dict(), save_dir / 'frcrn_best.pth')
            torch.save(joint_model.xlsr_model.state_dict(), save_dir / 'xlsr_best.pth')
            torch.save(joint_model.ad_model.state_dict(), save_dir / 'ad_model_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'phase': 2,
                'best_val_acc': best_val_acc,
                'best_metrics': best_metrics,
            }, save_dir / 'meta.pth')
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            stopped_epoch = epoch + 1
            break

    print(f"\nSeed {seed}:\n"
          f"Val Acc={best_metrics['accuracy'] * 100:.2f}%, "
          f"F1={best_metrics['f1_score']:.4f}, "
          f"Val Loss={best_metrics['loss']:.4f}\n"
          f"Control Acc={best_metrics['control_acc'] * 100:.2f}%, "
          f"Dementia Acc={best_metrics['dementia_acc'] * 100:.2f}%\n"
          f"Denoise Loss={best_metrics['denoise_loss']:.4f}\n"
          f"Best epoch: {best_epoch} | "
          f"Early stop at: epoch {stopped_epoch if stopped_epoch > 0 else MAX_EPOCHS}")

    training_history = {
        'epochs': epochs_list,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }

    return seed, best_metrics, training_history
