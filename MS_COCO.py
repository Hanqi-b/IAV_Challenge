'''
# The MS COCO classification challenge

Teacher: Razmig Kéchichian
Students: Hanqi Lin, Alexandre Maratrat
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
from glob import glob
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

class COCOTrainImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotations_dir, max_images=None, transform=None):
        self.img_labels = sorted(glob("*.cls", root_dir=annotations_dir))
        if max_images:
            self.img_labels = self.img_labels[:max_images]
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

        # Preload all labels into memory as one-hot vectors (length=80)
        self.labels_cache = []
        for label_file in self.img_labels:
            path = os.path.join(annotations_dir, label_file)
            with open(path) as f:
                labels = [int(line) for line in f.readlines()]
            one_hot = torch.zeros(80).scatter_(0, torch.tensor(labels), value=1)
            self.labels_cache.append(one_hot)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, Path(self.img_labels[idx]).stem + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = self.labels_cache[idx]
        return image, labels

class COCOTestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, preload=False):
        self.img_list = sorted(glob("*.jpg", root_dir=img_dir))    
        self.img_dir = img_dir
        self.transform = transform
        self.preload = preload

        # Preload test images into RAM
        self.images_cache = None
        if preload:
            print(f"Preloading {len(self.img_list)} test images into RAM...")
            self.images_cache = []
            for fname in self.img_list:
                img_path = os.path.join(img_dir, fname)
                image = Image.open(img_path).convert("RGB")
                if transform:
                    image = transform(image)
                self.images_cache.append((image, Path(fname).stem))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.preload and self.images_cache is not None:
            return self.images_cache[idx]
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, Path(img_path).stem  # filename without extension

# Export the split to a CSV file
def export_split_csv(dataset: COCOTrainImageDataset, indices, out_csv_path: str):
    dirn = os.path.dirname(out_csv_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "labels"])  # labels are space-separated indices
        for idx in indices:
            stem = Path(dataset.img_labels[idx]).stem
            one_hot = dataset.labels_cache[idx]
            ids = torch.nonzero(one_hot, as_tuple=False).squeeze(1).cpu().tolist()
            w.writerow([stem, " ".join(map(str, ids))])

# Append per-epoch metrics to CSV (for plotting)
def append_metrics_csv(csv_path: str, epoch: int, split: str, res: dict, lr: float):
    header = ["epoch","split","loss","accuracy","precision","recall","f1_macro_050","mAP","lr"]
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
            f"{res['f1_macro_050']:.6f}",
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
        # Use AMP if a GradScaler is provided
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = net(inputs)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        # Aggregate mini-batch losses for smoother plots
        if mbatch_loss_group > 0 and (i+1) % mbatch_loss_group == 0:
            mbatch_losses.append(running_loss / mbatch_loss_group); running_loss = 0.0

    if mbatch_loss_group > 0:
        return mbatch_losses

import torch

def validation_loop(val_loader, net, criterion, num_classes, device,
                    multi_task=False, one_hot=False, class_metrics=False):
    """
    Fixed threshold = 0.5:
      - predictions = (sigmoid(outputs) > 0.5)
      - returns macro-F1@0.50
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

            # AMP at validation; guarded by device type for portability
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = net(images)
                total_loss += criterion(outputs, labels).item() * images.size(0)

            # Multi-label vs single-label switch
            if not multi_task:
                predictions = torch.zeros_like(outputs)
                predictions[torch.arange(outputs.shape[0]), torch.argmax(outputs, dim=1)] = 1.0
            else:
                predictions = (torch.sigmoid(outputs) > 0.5).float()

            # If labels are not one-hot, convert them to one-hot
            if not one_hot:
                labels_mat = torch.zeros_like(outputs, device=device)
                labels_mat[torch.arange(outputs.shape[0]), labels] = 1.0
                labels = labels_mat

            # Per-class TP/FP and supports
            tps = (predictions * labels).sum(dim=0)
            fps = (predictions * (1 - labels)).sum(dim=0)
            lbls = labels.sum(dim=0)

            class_tp += tps
            class_fp += fps
            class_total += lbls
            correct += tps.sum()  # note: this makes "accuracy" below effectively micro-recall

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())

    # Per-class precision/recall
    class_prec = class_tp / (class_tp + class_fp + 1e-8)
    class_recall = class_tp / (class_total + 1e-8)

    # Aggregate precision/recall with inverse-frequency weights (rare classes emphasized)
    freqs = class_total
    class_weights = 1. / (freqs + 1e-8)
    class_weights /= class_weights.sum()

    prec = (class_prec * class_weights).sum()
    recall = (class_recall * class_weights).sum()
    val_loss = total_loss / size

    # "accuracy" here is effectively micro-recall (TP / sum of positives)
    accuracy = correct / freqs.sum()

    # Micro-F1@0.50 on flattened predictions and labels
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    pred_bin_05 = (all_probs >= 0.5).astype(int)
    f1_macro_050 = f1_score(all_labels, pred_bin_05, average="macro", zero_division=0)

    # Macro-Average Precision (mAP) over classes
    map_score = average_precision_score(all_labels, all_probs, average="macro")

    def to_float(x):
        return x.item() if torch.is_tensor(x) else float(x)

    results = {
        "loss": to_float(val_loss),
        "accuracy": to_float(accuracy),
        "precision": to_float(prec),
        "recall": to_float(recall),
        #"f1_at_050": to_float(f1_at_050),
        "f1_macro_050": to_float(f1_macro_050),
        "mAP": to_float(map_score)
    }

    # per-class metrics payload
    if class_metrics:
        class_results = []
        for p, r in zip(class_prec, class_recall):
            f1_c = (0 if p == 0 and r == 0 else 2. / (1/(p+1e-8) + 1/(r+1e-8)))
            class_results.append({"f1": to_float(f1_c), "precision": to_float(p), "recall": to_float(r)})
        results = results, class_results

    return results

# ============================
# 3.3 Tensorboard logging (We didn't use this fonction)
# ============================

# ============================
# 4. The skeleton of the model training and validation program
# ============================
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

# Save all the checkpoints
def get_next_checkpoint_path(folder="checkpoints"):
    existing = [f for f in os.listdir(folder) if f.startswith("checkpoint_") and f.endswith(".pth")]
    if not existing:
        next_idx = 1
    else:
        nums = [int(re.findall(r"checkpoint_(\d+)\.pth", f)[0]) for f in existing]
        next_idx = max(nums) + 1
    return os.path.join(folder, f"checkpoint_{next_idx}.pth")

# 3 Different Resnet models
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
    def forward(self, x): return self.base_model(x)
    
class ResNet101_MultiLabel(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.base_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    def forward(self, x): return self.base_model(x)

def main():
    # ---------- Hyperparameters ----------
    BATCH_SIZE = 256
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    VAL_SPLIT = 0.1

    # ---------- Run directories (timestamped) ----------
    run_id   = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = os.path.join("checkpoints", run_id)
    tb_dir   = os.path.join("runs",        run_id)
    out_dir  = os.path.join("outputs",     run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)
    print(f"[Run] {run_id}\n- ckpts: {ckpt_dir}\n- tb: {tb_dir}\n- out: {out_dir}")

    # ---------- Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # ---------- Data paths ----------
    base_path = Path(".").resolve()
    img_dir = base_path / "ms-coco" / "images" / "train-resized"
    annotations_dir = base_path / "ms-coco" / "labels" / "train"

    # ---------- Data augmentation ----------
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

    # Use a temporary dataset (no transform) to get total length N
    full_dataset = COCOTrainImageDataset(img_dir, annotations_dir, transform=None)
    N = len(full_dataset)

    # ---------- Random split (no fixed seed) ----------
    VAL_SPLIT = 0.1
    val_size = max(1, min(int(N * VAL_SPLIT), N - 1))
    train_size = N - val_size

    train_subset, val_subset = torch.utils.data.random_split(range(N), [train_size, val_size])
    train_indices = list(train_subset.indices)
    val_indices   = list(val_subset.indices)

    print(f"[Split] total={N}, train={len(train_indices)}, val={len(val_indices)}")

    # Two independent datasets with different transforms (train vs val)
    train_dataset = COCOTrainImageDataset(img_dir, annotations_dir, transform=train_transform)
    val_dataset   = COCOTrainImageDataset(img_dir, annotations_dir, transform=transform)

    # Slice with the same indices to form train/val subsets
    train_set = torch.utils.data.Subset(train_dataset, train_indices)
    val_set   = torch.utils.data.Subset(val_dataset,   val_indices)

    # ---------- DataLoaders ----------
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4,
                              pin_memory=True, prefetch_factor=2,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4,
                            pin_memory=True, prefetch_factor=2,
                            persistent_workers=True)

    # ---------- Model ----------
    # To switch backbone, uncomment one of the following lines:
    model = ResNet18_MultiLabel(num_classes=80).to(device)
    # model = ResNet50_MultiLabel(num_classes=80).to(device)
    # model = ResNet101_MultiLabel(num_classes=80).to(device)

    # ---------- Loss & Optimizer ----------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # ---------- Logging ----------
    writer = SummaryWriter(log_dir=tb_dir)
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")

    # ---------- Metrics CSV path ----------
    metrics_csv_path = os.path.join(out_dir, "metrics.csv")

    # ---------- Training loop ----------
    best_f1 = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        if ((epoch % 30 == 0) and (epoch>0)):
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"]/2
                print("⚡ Learning rate manually reduced to", param_group["lr"])

        # Train one epoch
        train_loop(train_loader, model, criterion, optimizer, device)

        # Compute metrics on train/val (fixed threshold = 0.5)
        train_eval_result = validation_loop(train_loader, model, criterion,
                                            num_classes=80, device=device,
                                            multi_task=True, one_hot=True, class_metrics=False)

        val_eval_result   = validation_loop(val_loader,   model, criterion,
                                            num_classes=80, device=device,
                                            multi_task=True, one_hot=True, class_metrics=False)

        # Console logs (Train + Val)
        print(f"Train | Loss: {train_eval_result['loss']:.4f}, "
              f"Acc: {train_eval_result['accuracy']:.4f}, "
              f"Prec: {train_eval_result['precision']:.4f}, "
              f"Rec: {train_eval_result['recall']:.4f}, "
              f"Macro-F1@0.50: {train_eval_result['f1_macro_050']:.4f}, "
              f"mAP: {train_eval_result['mAP']:.4f}")

        print(f"Val   | Loss: {val_eval_result['loss']:.4f}, "
              f"Acc: {val_eval_result['accuracy']:.4f}, "
              f"Prec: {val_eval_result['precision']:.4f}, "
              f"Rec: {val_eval_result['recall']:.4f}, "
              f"Macro-F1@0.50: {val_eval_result['f1_macro_050']:.4f}, "
              f"mAP: {val_eval_result['mAP']:.4f}")

        # TensorBoard (Train + Val)
        lr_now = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/Train",    train_eval_result["loss"],    epoch)
        writer.add_scalar("Loss/Val",      val_eval_result["loss"],      epoch)
        writer.add_scalar("Accuracy/Train", train_eval_result["accuracy"], epoch)
        writer.add_scalar("Accuracy/Val",   val_eval_result["accuracy"],   epoch)
        writer.add_scalar("Precision/Train", train_eval_result["precision"], epoch)
        writer.add_scalar("Precision/Val",   val_eval_result["precision"],   epoch)
        writer.add_scalar("Recall/Train", train_eval_result["recall"], epoch)
        writer.add_scalar("Recall/Val",   val_eval_result["recall"],   epoch)
        writer.add_scalar("Macro-F1@0.50/Train", train_eval_result["f1_macro_050"], epoch)
        writer.add_scalar("Macro-F1@0.50/Val",   val_eval_result["f1_macro_050"],   epoch)
        writer.add_scalar("mAP/Train", train_eval_result["mAP"], epoch)
        writer.add_scalar("mAP/Val",   val_eval_result["mAP"],   epoch)
        writer.add_scalar("LR", lr_now, epoch)

        # CSV append (two rows per epoch: train / val)
        metrics_csv_path = os.path.join(out_dir, "metrics.csv")
        append_metrics_csv(metrics_csv_path, epoch+1, "train", train_eval_result, lr_now)
        append_metrics_csv(metrics_csv_path, epoch+1, "val",   val_eval_result,   lr_now)

        # Checkpoints and best model tracking
        checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Model checkpoint saved: {checkpoint_path}")

        if val_eval_result["f1_macro_050"] > best_f1:
            best_f1 = val_eval_result["f1_macro_050"]
            torch.save(model.state_dict(), best_model_path)
            print("✅ New best model saved with Macro-F1@0.50:", best_f1)
        else:
            print("Model not improved (Macro-F1@0.50):", val_eval_result["f1_macro_050"])

    writer.close()

# ---------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
