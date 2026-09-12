"""
MobileNetV2 模型训练脚本
- 使用预训练的MobileNetV2进行迁移学习
- 准确率达到85%停止训练
- 连续3轮验证集无改进则早停
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (CLASSIFIED_DIR, MODEL_DIR, BEST_MODEL_PATH,
                    CLASS_NAMES, CLASS_NAMES_CN, TRAIN_CONFIG)


def get_data_transforms():
    """获取数据增强和预处理变换"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(TRAIN_CONFIG['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((TRAIN_CONFIG['image_size'], TRAIN_CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def build_model(num_classes):
    """构建MobileNetV2模型"""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # 冻结前面的特征提取层
    for param in model.features.parameters():
        param.requires_grad = False
    # 只解冻最后几层
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    # 替换分类头
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='训练中', ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='评估中', ncols=100):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def calibrate_temperature(model, dataloader, device):
    """在验证集上拟合温度缩放参数，校准模型置信度"""
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            labels_list.append(labels)

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    temperature = torch.nn.Parameter(torch.tensor(1.5))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    t = temperature.item()

    # 校准前后对比
    raw_probs = torch.softmax(all_logits, dim=1)
    cal_probs = torch.softmax(all_logits / t, dim=1)
    raw_conf = raw_probs.max(dim=1).values.mean().item()
    cal_conf = cal_probs.max(dim=1).values.mean().item()
    print(f"\n温度校准完成: T = {t:.4f}")
    print(f"  校准前平均置信度: {raw_conf*100:.1f}% -> 校准后: {cal_conf*100:.1f}%")

    return t


def plot_training_history(history, save_dir):
    """绘制训练历史曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.axhline(y=TRAIN_CONFIG['target_accuracy'], color='r',
                linestyle='--', label=f'Target ({TRAIN_CONFIG["target_accuracy"]*100}%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"训练曲线已保存到: {os.path.join(save_dir, 'training_history.png')}")


def plot_confusion_matrix(cm, class_names, save_dir):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"混淆矩阵已保存到: {os.path.join(save_dir, 'confusion_matrix.png')}")


def main():
    print("=" * 60)
    print("甘蔗病虫害识别 - MobileNetV2 模型训练")
    print("=" * 60)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型保存目录
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 数据加载
    train_transform, val_transform = get_data_transforms()

    train_dir = os.path.join(CLASSIFIED_DIR, 'train')
    valid_dir = os.path.join(CLASSIFIED_DIR, 'valid')
    test_dir = os.path.join(CLASSIFIED_DIR, 'test')

    if not os.path.exists(train_dir):
        print("[错误] 分类数据目录不存在，请先运行 prepare_data.py")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_transform)

    print(f"\n训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(valid_dataset)} 张图片")
    print(f"类别: {train_dataset.classes}")
    print(f"类别映射: {train_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset,
                              batch_size=TRAIN_CONFIG['batch_size'],
                              shuffle=True,
                              num_workers=TRAIN_CONFIG['num_workers'],
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=TRAIN_CONFIG['batch_size'],
                              shuffle=False,
                              num_workers=TRAIN_CONFIG['num_workers'],
                              pin_memory=True)

    # 构建模型
    num_classes = len(train_dataset.classes)
    model = build_model(num_classes)
    model = model.to(device)
    print(f"\nMobileNetV2 模型已加载，分类数: {num_classes}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=TRAIN_CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=2)

    # 训练循环
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    print(f"\n开始训练 (目标准确率: {TRAIN_CONFIG['target_accuracy']*100}%, "
          f"早停耐心: {TRAIN_CONFIG['patience']} 轮)")
    print("-" * 60)

    for epoch in range(1, TRAIN_CONFIG['max_epochs'] + 1):
        print(f"\nEpoch {epoch}/{TRAIN_CONFIG['max_epochs']}")

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer, device)
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, criterion, device)

        # 学习率调度
        scheduler.step(val_acc)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存模型
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_to_idx': train_dataset.class_to_idx,
                'classes': train_dataset.classes,
            }
            torch.save(save_dict, BEST_MODEL_PATH)
            print(f"  [*] 新最佳模型已保存! (准确率: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  未改进 ({patience_counter}/{TRAIN_CONFIG['patience']})")

        # 检查是否达到目标准确率
        if val_acc >= TRAIN_CONFIG['target_accuracy']:
            print(f"\n[OK] 达到目标准确率 {val_acc*100:.2f}% >= {TRAIN_CONFIG['target_accuracy']*100}%，停止训练!")
            break

        # 检查早停
        if patience_counter >= TRAIN_CONFIG['patience']:
            print(f"\n[OK] 连续 {TRAIN_CONFIG['patience']} 轮无改进，早停!")
            break

    elapsed = time.time() - start_time
    print(f"\n训练完成! 耗时: {elapsed/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")

    # 绘制训练曲线
    plot_training_history(history, MODEL_DIR)

    # 温度校准
    print("\n" + "=" * 60)
    print("模型温度校准")
    print("=" * 60)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    temperature = calibrate_temperature(model, valid_loader, device)
    checkpoint['temperature'] = temperature
    torch.save(checkpoint, BEST_MODEL_PATH)
    print(f"温度参数已保存到模型文件")

    # 在测试集上评估
    if os.path.exists(test_dir):
        print("\n" + "=" * 60)
        print("在测试集上评估最佳模型")
        print("=" * 60)

        test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
        test_loader = DataLoader(test_dataset,
                                 batch_size=TRAIN_CONFIG['batch_size'],
                                 shuffle=False,
                                 num_workers=TRAIN_CONFIG['num_workers'])

        # 加载最佳模型
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc, preds, labels = evaluate(model, test_loader,
                                                       criterion, device)
        print(f"\n测试集 - Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%")

        # 分类报告
        target_names = [f"{cn}({CLASS_NAMES_CN.get(cn, cn)})" for cn in train_dataset.classes]
        report = classification_report(labels, preds, target_names=target_names)
        print(f"\n分类报告:\n{report}")

        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        plot_confusion_matrix(cm, train_dataset.classes, MODEL_DIR)

        # 保存测试结果
        results = {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'best_val_accuracy': float(best_val_acc),
            'total_epochs': len(history['train_loss']),
            'classes': train_dataset.classes,
            'class_to_idx': train_dataset.class_to_idx,
        }
        results_path = os.path.join(MODEL_DIR, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n训练结果已保存到: {results_path}")

    print("\n模型训练流程全部完成!")


if __name__ == '__main__':
    main()
