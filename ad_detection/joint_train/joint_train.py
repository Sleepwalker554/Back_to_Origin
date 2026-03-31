import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from joint_config import (
    MAX_EPOCHS, WEIGHT_DECAY, ETA_MIN, ALPHA, BETA,
    FRCRN_LR, XLSR_LR, AD_LR, GRADIENT_ACCUMULATION_STEPS,
    USE_AMP, XLSR_FINETUNE_LAST_N, USE_GRADIENT_CHECKPOINT,
    XLSR_MAX_TIME_STEPS, PATIENCE,
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
    ad_model = AD_XLSR_Model(dropout=0.4).to(device)

    joint_model = JointDenoiseADModel(
        frcrn_model, xlsr_model, ad_model,
        xlsr_max_time_steps=XLSR_MAX_TIME_STEPS,
    ).to(device)

    return joint_model


def build_optimizer(joint_model):
    """三组参数，三个学习率"""
    param_groups = [
        {
            'params': [p for p in joint_model.frcrn_model.parameters() if p.requires_grad],
            'lr': FRCRN_LR,
            'name': 'frcrn_unet2',
        },
        {
            'params': [p for p in joint_model.xlsr_model.parameters() if p.requires_grad],
            'lr': XLSR_LR,
            'name': 'xlsr_last3',
        },
        {
            'params': joint_model.ad_model.parameters(),
            'lr': AD_LR,
            'name': 'ad_classifier',
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=ETA_MIN)

    return optimizer, scheduler


def train_one_epoch(joint_model, train_loader, optimizer, device,
                    epoch=None, class_weights=None):
    joint_model.frcrn_model.train()
    joint_model.ad_model.train()
    # XLSR 保持 eval (BatchNorm/Dropout 行为)，但后3层仍然接收梯度

    total_loss = 0
    total_denoise_loss = 0
    total_classify_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    desc = f"Epoch {epoch} - Training" if epoch is not None else "Training"
    pbar = tqdm(train_loader, desc=desc, leave=False)

    for i, (raw, clean, labels) in enumerate(pbar):
        raw = raw.to(device)
        clean = clean.to(device)
        labels = labels.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_AMP):
            denoised, logits = joint_model(raw)

            loss_denoise = F.l1_loss(denoised, clean)
            loss_classify = F.cross_entropy(logits, labels, weight=class_weights)
            loss_total = ALPHA * loss_denoise + BETA * loss_classify
            loss_scaled = loss_total / GRADIENT_ACCUMULATION_STEPS

        loss_scaled.backward()

        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
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

    # 处理尾部不足 accumulation steps 的梯度
    if (i + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
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


def train(seed, train_loader, val_loader, output_dir, device,
          frcrn_pretrained_path=None,
          class_weight_control=1.0, class_weight_dementia=1.0):
    """
    Joint training pipeline

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
    optimizer, scheduler = build_optimizer(joint_model)

    # 训练历史
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    epochs_list = []

    # Early stopping
    best_val_acc = 0
    best_metrics = {}
    patience_counter = 0
    best_epoch = 0
    stopped_epoch = 0

    for epoch in range(MAX_EPOCHS):
        train_metrics = train_one_epoch(
            joint_model, train_loader, optimizer, device,
            epoch=epoch + 1, class_weights=class_weights,
        )
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['accuracy'])

        scheduler.step()

        val_metrics = validate(
            joint_model, val_loader, device,
            epoch=epoch + 1, class_weights=class_weights,
        )
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])
        epochs_list.append(epoch)

        print(f"Epoch {epoch + 1}: "
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
            # 分别保存三个模型
            torch.save(joint_model.frcrn_model.state_dict(), save_dir / 'frcrn_best.pth')
            torch.save(joint_model.xlsr_model.state_dict(), save_dir / 'xlsr_best.pth')
            torch.save(joint_model.ad_model.state_dict(), save_dir / 'ad_model_best.pth')
            # 保存训练元信息
            torch.save({
                'epoch': epoch + 1,
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
          f"Early stop at: epoch {stopped_epoch if stopped_epoch > 0 else MAX_EPOCHS} "
          f"(best epoch: {best_epoch})")

    training_history = {
        'epochs': epochs_list,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }

    return seed, best_metrics, training_history
