'''
# The MS COCO classification challenge

Razmig Kéchichian
...
'''
classes = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",       
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
           "hair drier", "toothbrush")

# ============================
# 3.1 Custom Datasets
# ============================
import os
import csv
from datetime import datetime
from pathlib import Path
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class COCOTrainImageDataset(Dataset):
    def __init__(self, img_dir, annotations_dir, max_images=None, transform=None):
        self.img_labels = sorted(glob("*.cls", root_dir=annotations_dir))
        if max_images:
            self.img_labels = self.img_labels[:max_images]
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

        # 提前加载所有标签到内存（one-hot）
        self.labels_cache = []
        for label_file in self.img_labels:
            path = os.path.join(annotations_dir, label_file)
            with open(path, "r") as f:
                labels = [int(line) for line in f.readlines()]
            one_hot = torch.zeros(80, dtype=torch.float32)
            if len(labels) > 0:
                idx = torch.tensor(labels, dtype=torch.long)
                one_hot[idx] = 1.0  # 等价于 scatter，写法更直观
            self.labels_cache.append(one_hot)

    def __len__(self):
        # ★ 训练集按标签文件计数
        return len(self.img_labels)

    def __getitem__(self, idx):
        # 由标签文件名推回同名图片
        img_path = os.path.join(self.img_dir, Path(self.img_labels[idx]).stem + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = self.labels_cache[idx]
        return image, labels

# —— 划分 CSV —— 
def export_split_csv(dataset: COCOTrainImageDataset, indices, out_csv_path: str):
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "labels"])  # labels 空格分隔
        for idx in indices:
            stem = Path(dataset.img_labels[idx]).stem
            one_hot = dataset.labels_cache[idx]
            ids = torch.nonzero(one_hot, as_tuple=False).squeeze(1).cpu().tolist()
            w.writerow([stem, " ".join(map(str, ids))])

# —— 逐 epoch 指标 CSV（便于画图）——
def append_metrics_csv(csv_path: str, epoch: int, split: str, res: dict, lr: float):
    header = ["epoch","split","loss","accuracy","precision","recall","f1_at_050","mAP","lr"]
    need_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow([
            epoch, split,
            f"{res['loss']:.6f}",
            f"{res['accuracy']:.6f}",
            f"{res['precision']:.6f}",
            f"{res['recall']:.6f}",
            f"{res['f1_at_050']:.6f}",
            f"{res['mAP']:.6f}",
            f"{lr:.8f}",
        ])

# ============================
# 3.2 Training and validation loops
# ============================
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, average_precision_score

def train_loop(train_loader, net, criterion, optimizer, device,
               scaler=None, mbatch_loss_group=-1):
    net.train()
    running_loss = 0.0
    mbatch_losses = []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", unit="batch")

    for i, (inputs, labels) in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = net(inputs)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        if mbatch_loss_group > 0 and (i+1) % mbatch_loss_group == 0:
            mbatch_losses.append(running_loss / mbatch_loss_group); running_loss = 0.0

    if mbatch_loss_group > 0:
        return mbatch_losses

import torch

def validation_loop(val_loader, net, criterion, num_classes, device,
                    multi_task=False, one_hot=False, class_metrics=False):
    """
    仅使用固定的概率阈值 0.5：
    - predictions = (sigmoid(outputs) > 0.5)
    - 返回 f1_at_050（micro-F1@0.50）
    """
    net.eval()
    total_loss = 0.0
    correct = 0
    size = len(val_loader.dataset)

    class_total = torch.zeros(num_classes, device=device)
    class_tp = torch.zeros(num_classes, device=device)
    class_fp = torch.zeros(num_classes, device=device)

    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", unit="batch"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = net(images)
                total_loss += criterion(outputs, labels).item() * images.size(0)

            if not multi_task:
                predictions = torch.zeros_like(outputs)
                predictions[torch.arange(outputs.shape[0]), torch.argmax(outputs, dim=1)] = 1.0
            else:
                predictions = (torch.sigmoid(outputs) > 0.5).float()

            if not one_hot:
                labels_mat = torch.zeros_like(outputs, device=device)
                labels_mat[torch.arange(outputs.shape[0]), labels] = 1.0
                labels = labels_mat

            tps = (predictions * labels).sum(dim=0)
            fps = (predictions * (1 - labels)).sum(dim=0)
            lbls = labels.sum(dim=0)

            class_tp += tps
            class_fp += fps
            class_total += lbls
            correct += tps.sum()

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())

    class_prec = class_tp / (class_tp + class_fp + 1e-8)
    class_recall = class_tp / (class_total + 1e-8)

    freqs = class_total
    class_weights = 1. / (freqs + 1e-8)
    class_weights /= class_weights.sum()

    prec = (class_prec * class_weights).sum()
    recall = (class_recall * class_weights).sum()
    f1_weighted = 2. / (1/prec + 1/recall + 1e-8)
    val_loss = total_loss / size
    accuracy = correct / freqs.sum()  # ≈ micro-recall

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    pred_bin_05 = (all_probs >= 0.5).astype(int)
    f1_at_050 = f1_score(all_labels, pred_bin_05, average="micro", zero_division=0)
    map_score = average_precision_score(all_labels, all_probs, average="macro")

    def to_float(x):
        return x.item() if torch.is_tensor(x) else float(x)

    results = {
        "loss": to_float(val_loss),
        "accuracy": to_float(accuracy),
        "precision": to_float(prec),
        "recall": to_float(recall),
        "f1": to_float(f1_weighted),     # 加权 F1（固定阈值）
        "f1_at_050": to_float(f1_at_050),# micro-F1@0.50（打印/CSV用）
        "mAP": to_float(map_score)
    }

    if class_metrics:
        class_results = []
        for p, r in zip(class_prec, class_recall):
            f1_c = (0 if p == 0 and r == 0 else 2. / (1/(p+1e-8) + 1/(r+1e-8)))
            class_results.append({"f1": to_float(f1_c), "precision": to_float(p), "recall": to_float(r)})
        results = results, class_results

    return results

# ============================
# 3.3 Tensorboard logging (optional)
# ============================
def update_graphs(summary_writer, epoch, train_results, test_results,
                  train_class_results=None, test_class_results=None, 
                  class_names = None, mbatch_group=-1, mbatch_count=0, mbatch_losses=None):
    if mbatch_group > 0:
        for i in range(len(mbatch_losses)):
            summary_writer.add_scalar("Losses/Train mini-batches",
                                  mbatch_losses[i],
                                  epoch * mbatch_count + (i+1)*mbatch_group)

    summary_writer.add_scalars("Losses/Train Loss vs Test Loss",
                               {"Train Loss" : train_results["loss"],
                                "Test Loss" : test_results["loss"]},
                               (epoch + 1) if not mbatch_group > 0
                                     else (epoch + 1) * mbatch_count)

    summary_writer.add_scalars("Metrics/Train Accuracy vs Test Accuracy",
                               {"Train Accuracy" : train_results["accuracy"],
                                "Test Accuracy" : test_results["accuracy"]},
                               (epoch + 1) if not mbatch_group > 0
                                     else (epoch + 1) * mbatch_count)

    # 这里仍记录“加权F1”，与上面一致；你也可以改成记录 f1_at_050
    summary_writer.add_scalars("Metrics/Train F1 vs Test F1",
                               {"Train F1" : train_results["f1"],
                                "Test F1" : test_results["f1"]},
                               (epoch + 1) if not mbatch_group > 0
                                     else (epoch + 1) * mbatch_count)

    summary_writer.add_scalars("Metrics/Train Precision vs Test Precision",
                               {"Train Precision" : train_results["precision"],
                                "Test Precision" : test_results["precision"]},
                               (epoch + 1) if not mbatch_group > 0
                                     else (epoch + 1) * mbatch_count)

    summary_writer.add_scalars("Metrics/Train Recall vs Test Recall",
                               {"Train Recall" : train_results["recall"],
                                "Test Recall" : test_results["recall"]},
                               (epoch + 1) if not mbatch_group > 0
                                     else (epoch + 1) * mbatch_count)

    summary_writer.flush()

# ============================
# 4. The skeleton of the model training and validation program
# ============================
import os
from datetime import datetime  # ★ 新增
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torch.multiprocessing as mp
import re

best_f1 = 0.0
os.makedirs("checkpoints", exist_ok=True)

def get_next_checkpoint_path(folder="checkpoints"):
    existing = [f for f in os.listdir(folder) if f.startswith("checkpoint_") and f.endswith(".pth")]
    if not existing:
        next_idx = 1
    else:
        nums = [int(re.findall(r"checkpoint_(\d+)\.pth", f)[0]) for f in existing]
        next_idx = max(nums) + 1
    return os.path.join(folder, f"checkpoint_{next_idx}.pth")

class ResNet18_MultiLabel(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    def forward(self, x):
        return self.base_model(x)

class ResNet50_MultiLabel(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    def forward(self, x): 
        return self.base_model(x)
    
class ResNet101_MultiLabel(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.base_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    def forward(self, x): 
        return self.base_model(x)

# ★ 只微调 layer3、layer4、fc；其余冻结（含 BatchNorm 行为）
def freeze_all_but_l3l4_fc(model: nn.Module):
    """
    只微调 ResNet 的 layer3、layer4 与 fc；其余层全部冻结。
    兼容两种结构：
      1) 直接 torchvision resnet：model.layer3 / model.layer4 / model.fc
      2) 你的封装：model.base_model.layer3 / layer4 / fc
    """
    import torch.nn as nn

    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # 取 backbone（如果有 base_model 就用它，否则用模型本身）
    backbone = getattr(model, "base_model", model)

    # 安全检查
    for name in ("layer3", "layer4", "fc"):
        if not hasattr(backbone, name):
            raise AttributeError(f"Backbone missing attribute '{name}'. "
                                 f"Got {type(backbone).__name__} with attrs: {list(dict(backbone.named_children()).keys())}")

    # 解冻 layer3 / layer4 / fc
    for p in backbone.layer3.parameters():
        p.requires_grad = True
    for p in backbone.layer4.parameters():
        p.requires_grad = True
    for p in backbone.fc.parameters():
        p.requires_grad = True

    # 对已冻结的 BN 层：设为 eval 并禁用其参数梯度
    for m in backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False


def main():
    # ========== 超参数 ==========
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-3
    VAL_SPLIT = 0.1

    # ========== 本次运行的时间戳目录 ==========
    run_id   = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = os.path.join("checkpoints", run_id)
    tb_dir   = os.path.join("runs",        run_id)
    out_dir  = os.path.join("outputs",     run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)
    print(f"[Run] {run_id}\n- ckpts: {ckpt_dir}\n- tb: {tb_dir}\n- out: {out_dir}")

    # ========== 设备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # ========== 数据路径 ==========
    base_path = Path(".").resolve()
    img_dir = base_path / "ms-coco" / "images" / "train-resized"
    annotations_dir = base_path / "ms-coco" / "labels" / "train"

    # ========== 数据增强 ==========
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # === 仅用于获取总长度 ===
    full_dataset = COCOTrainImageDataset(img_dir, annotations_dir, transform=None)
    N = len(full_dataset)

    # === 随机划分（无 seed）===
    val_size = max(1, min(int(N * VAL_SPLIT), N - 1))
    train_size = N - val_size
    train_subset, val_subset = torch.utils.data.random_split(range(N), [train_size, val_size])
    train_indices = list(train_subset.indices)
    val_indices   = list(val_subset.indices)
    print(f"[Split] total={N}, train={len(train_indices)}, val={len(val_indices)}")

    # === 两个独立的 dataset，分别用不同的 transform ===
    train_dataset = COCOTrainImageDataset(img_dir, annotations_dir, transform=train_transform)
    val_dataset   = COCOTrainImageDataset(img_dir, annotations_dir, transform=transform)

    # 用相同索引切分
    train_set = torch.utils.data.Subset(train_dataset, train_indices)
    val_set   = torch.utils.data.Subset(val_dataset,   val_indices)

    # === DataLoader ===
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4,
                              pin_memory=True, prefetch_factor=2,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4,
                            pin_memory=True, prefetch_factor=2,
                            persistent_workers=True)

    # ========== 模型 ==========
    model = ResNet18_MultiLabel(num_classes=80).to(device)
    # model = ResNet50_MultiLabel(num_classes=80).to(device)
    # model = ResNet101_MultiLabel(num_classes=80).to(device)

    # ★ 只微调 layer3、layer4、fc
    freeze_all_but_l3l4_fc(model)

    # ========== 损失函数 & 优化器 ==========
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # ★ 只更新可训练参数
        lr=LEARNING_RATE, weight_decay=1e-5
    )

    # ========== 日志 ==========
    writer = SummaryWriter(log_dir=tb_dir)
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")

    # ========== 指标 CSV 路径 ==========
    metrics_csv_path = os.path.join(out_dir, "metrics.csv")

    # ========== 训练循环 ==========
    best_f1 = 0.0  # ★ 这里的 f1 指 micro-F1@0.50
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        if (epoch > 0) and (epoch % 20 == 0):
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5
            print("⚡ Learning rate halved to", optimizer.param_groups[0]["lr"])

        # —— 训练一个 epoch ——
        train_loop(train_loader, model, criterion, optimizer, device)

        # —— 计算“训练集”与“验证集”指标（阈值固定 0.5；确保 validation_loop 返回 results['f1']=micro-F1@0.50）——
        train_eval_result = validation_loop(train_loader, model, criterion,
                                            num_classes=80, device=device,
                                            multi_task=True, one_hot=True, class_metrics=False)
        val_eval_result   = validation_loop(val_loader,   model, criterion,
                                            num_classes=80, device=device,
                                            multi_task=True, one_hot=True, class_metrics=False)

        # —— 控制台打印（Train + Val）——
        print(f"Train | Loss: {train_eval_result['loss']:.4f}, "
              f"Acc: {train_eval_result['accuracy']:.4f}, "
              f"Prec: {train_eval_result['precision']:.4f}, "
              f"Rec: {train_eval_result['recall']:.4f}, "
              f"F1(micro@0.50): {train_eval_result['f1']:.4f}, "
              f"mAP: {train_eval_result['mAP']:.4f}")

        print(f"Val   | Loss: {val_eval_result['loss']:.4f}, "
              f"Acc: {val_eval_result['accuracy']:.4f}, "
              f"Prec: {val_eval_result['precision']:.4f}, "
              f"Rec: {val_eval_result['recall']:.4f}, "
              f"F1(micro@0.50): {val_eval_result['f1']:.4f}, "
              f"mAP: {val_eval_result['mAP']:.4f}")

        # —— TensorBoard（Train + Val）——
        lr_now = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/Train",        train_eval_result["loss"],    epoch)
        writer.add_scalar("Loss/Val",          val_eval_result["loss"],      epoch)
        writer.add_scalar("Accuracy/Train",    train_eval_result["accuracy"], epoch)
        writer.add_scalar("Accuracy/Val",      val_eval_result["accuracy"],   epoch)
        writer.add_scalar("Precision/Train",   train_eval_result["precision"], epoch)
        writer.add_scalar("Precision/Val",     val_eval_result["precision"],   epoch)
        writer.add_scalar("Recall/Train",      train_eval_result["recall"], epoch)
        writer.add_scalar("Recall/Val",        val_eval_result["recall"],   epoch)
        writer.add_scalar("F1/micro@0.50/Train", train_eval_result["f1"], epoch)   # ★
        writer.add_scalar("F1/micro@0.50/Val",   val_eval_result["f1"],   epoch)   # ★
        writer.add_scalar("mAP/Train",         train_eval_result["mAP"], epoch)
        writer.add_scalar("mAP/Val",           val_eval_result["mAP"],   epoch)
        writer.add_scalar("LR",                lr_now, epoch)

        # —— CSV（同一个 metrics.csv，追加两行：train / val）——
        append_metrics_csv(metrics_csv_path, epoch+1, "train", train_eval_result, lr_now)  # ★ res['f1']
        append_metrics_csv(metrics_csv_path, epoch+1, "val",   val_eval_result,   lr_now)  # ★ res['f1']

        # —— Checkpoint 与 best —— 
        checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Model checkpoint saved: {checkpoint_path}")

        if val_eval_result["f1"] > best_f1:  # ★ 用 micro-F1@0.50 选最佳
            best_f1 = val_eval_result["f1"]
            torch.save(model.state_dict(), best_model_path)
            print("✅ New best model saved with F1(micro@0.50):", best_f1)
        else:
            print("Model not improved (F1 micro@0.50):", val_eval_result["f1"])

    writer.close()

# ---------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
