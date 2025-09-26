'''
# The MS COCO classification challenge

Razmig Kéchichian

This notebook defines the multi-class classification challenge on the [MS COCO dataset](https://cocodataset.org/). It defines the problem, sets the rules of organization and presents tools you are provided with to accomplish the challenge.


## 1. Problem statement

Each image has **several** categories of objects to predict, hence the difference compared to the classification problem we have seen on the CIFAR10 dataset where each image belonged to a **single** category, therefore the network loss function and prediction mechanism (only highest output probability) were defined taking this constraint into account.

We adapted the MS COCO dataset for the requirements of this challenge by, among other things, reducing the number of images and their dimensions to facilitate processing.

In the companion `ms-coco.zip` compressed directory you will find two sub-directories:
- `images`: which contains the images in train (65k) and test (~5k) subsets,
- `labels`: which lists labels for each of the images in the train subset only.

Each label file gives a list of class IDs that correspond to the class index in the following tuple:
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
'''
Your goal is to follow a **transfer learning strategy** in training and validating a network on **your own distribution of training data into training and a validation subsets**, then to **test it on the test subset** by producing a [JSON file](https://en.wikipedia.org/wiki/JSON) with content of the following format:

```
{
    "000000000139": [
        56,
        60,
        62
    ],
    "000000000285": [
        21,
    ],
    "000000000632": [
        57,
        59,
    73
    ],
    # other test images
}
```

In this file, the name (without extension) of each test image is associated with a list of class indices predicted by your network. Make sure that the JSON file you produce **follows this format strictly**.

You will submit your JSON prediction file to the following [online evaluation server and leaderboard](https://www.creatis.insa-lyon.fr/kechichian/ms-coco-classif-leaderboard.html), which will evaluate your predictions on test set labels, unavailable to you.

<div class="alert alert-block alert-danger"> <b>WARNING:</b> Use this server with <b>the greatest care</b>. A new submission with identical Participant or group name will <b>overwrite</b> the identically named submission, if one already exists, therefore check the leaderboard first. <b>Do not make duplicate leaderboard entries for your group</b>, keep track of your test scores privately. Also pay attention to upload only JSON files of the required format.<br>
</div>

The evaluation server calculates and returns mean performances over all classes, and optionally per class performances. Entries in the leaderboard are sorted by the F1 metric.

You can request an evaluation as many times as you want. It is up to you to specify the final evaluation by updating the leaderboard entry corresponding to your Participant or group name. This entry will be taken into account for grading your work.

It goes without saying that it is **prohibited** to use another distribution of the MS COCO database for training, e.g. the Torchvision dataset.


## 2. Organization

- Given the scope of the project, you will work in groups of 2. 
- Work on the challenge begins on IAV lab 3 session, that is on the **23rd of September**.
- Results are due 10 days later, that is on the **3rd of October, 18:00**. They comrpise:
    - a submission to the leaderboard,
    - a commented Python script (with any necessary modules) or Jupyter Notebook, uploaded on Moodle in the challenge repository by one of the members of the group.
    
    
## 3. Tools

In addition to the MS COCO annotated data and the evaluation server, we provide you with most code building blocks. Your task is to understand them and use them to create the glue logic, that is the main program, putting all these blocks together and completing them as necessary to implement a complete machine learning workflow to train and validate a model, and produce the test JSON file.

### 3.1 Custom `Dataset`s

We provide you with two custom `torch.utils.data.Dataset` sub-classes to use in training and testing.
'''
import os
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

        # ✅ 提前加载所有标签到内存
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

        # ✅ 直接用缓存里的标签
        labels = self.labels_cache[idx]

        return image, labels

class COCOTestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, preload=False):
        self.img_list = sorted(glob("*.jpg", root_dir=img_dir))    
        self.img_dir = img_dir
        self.transform = transform
        self.preload = preload

        # ✅ 可选：预先加载所有图像到内存
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
        return image, Path(img_path).stem  # filename w/o extension
    
'''
### 3.2 Training and validation loops

The following are two general-purpose classification train and validation loop functions to be called inside the epochs for-loop with appropriate argument settings.

Pay particular attention to the `validation_loop()` function's arguments `multi_task`, `th_multi_task` and `one_hot`.
'''

from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, average_precision_score

def train_loop(train_loader, net, criterion, optimizer, device,
               scaler=None, mbatch_loss_group=-1):
    net.train()
    running_loss = 0.0
    mbatch_losses = []

    # ✅ tqdm 进度条
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", unit="batch")

    for i, (inputs, labels) in progress_bar:
        # ✅ 异步传输到 GPU
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # ✅ AMP 混合精度
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = net(inputs)
            loss = criterion(outputs, labels)

        if scaler is not None:  # 混合精度
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # 普通 FP32
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        # ✅ 更新 tqdm 显示
        progress_bar.set_postfix(loss=loss.item())

        if mbatch_loss_group > 0 and (i+1) % mbatch_loss_group == 0:
            mbatch_losses.append(running_loss / mbatch_loss_group)
            running_loss = 0.0

    if mbatch_loss_group > 0:
        return mbatch_losses



import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm
import torch

def validation_loop(val_loader, net, criterion, num_classes, device,
                    multi_task=False, th_multi_task=0.5, one_hot=False, class_metrics=False):
    net.eval()
    total_loss = 0.0
    correct = 0
    size = len(val_loader.dataset)

    # ✅ 直接用 tensor 代替 Python dict，加速
    class_total = torch.zeros(num_classes, device=device)
    class_tp = torch.zeros(num_classes, device=device)
    class_fp = torch.zeros(num_classes, device=device)

    # ✅ 新增：收集所有概率和标签，用来做阈值扫描 & mAP
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", unit="batch"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ✅ 混合精度推理
            with torch.amp.autocast("cuda"):
                outputs = net(images)
                total_loss += criterion(outputs, labels).item() * images.size(0)

            # === multi-label predictions ===
            if not multi_task:    
                predictions = torch.zeros_like(outputs)
                predictions[torch.arange(outputs.shape[0]), torch.argmax(outputs, dim=1)] = 1.0
            else:
                predictions = (outputs > th_multi_task).float()

            if not one_hot:
                labels_mat = torch.zeros_like(outputs, device=device)
                labels_mat[torch.arange(outputs.shape[0]), labels] = 1.0
                labels = labels_mat

            # ✅ 向量化统计
            tps = (predictions * labels).sum(dim=0)
            fps = (predictions * (1 - labels)).sum(dim=0)
            lbls = labels.sum(dim=0)

            class_tp += tps
            class_fp += fps
            class_total += lbls
            correct += tps.sum()

            # ✅ 收集概率和标签 (sigmoid → numpy)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())

    # === 计算整体指标 (原逻辑) ===
    class_prec = class_tp / (class_tp + class_fp + 1e-8)
    class_recall = class_tp / (class_total + 1e-8)

    freqs = class_total
    class_weights = 1. / (freqs + 1e-8)
    class_weights /= class_weights.sum()

    prec = (class_prec * class_weights).sum()
    recall = (class_recall * class_weights).sum()
    f1 = 2. / (1/prec + 1/recall + 1e-8)
    val_loss = total_loss / size
    accuracy = correct / freqs.sum()

    # === 新增：阈值扫描 + mAP ===
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    best_thr, best_f1 = 0.5, 0
    for thr in np.linspace(0.1, 0.9, 17):  # 0.1, 0.15, ... 0.9
        pred_bin = (all_probs >= thr).astype(int)
        f1_tmp = f1_score(all_labels, pred_bin, average="micro", zero_division=0)
        if f1_tmp > best_f1:
            best_f1, best_thr = f1_tmp, thr

    map_score = average_precision_score(all_labels, all_probs, average="macro")

    # ✅ 安全转换函数，兼容 float 和 tensor
    def to_float(x):
        return x.item() if torch.is_tensor(x) else float(x)

    results = {
        "loss": to_float(val_loss),
        "accuracy": to_float(accuracy),
        "f1": to_float(f1),          # 原有的加权 F1
        "precision": to_float(prec),
        "recall": to_float(recall),
        "best_f1": best_f1,          # 新增：阈值优化后 F1
        "best_thr": best_thr,        # 新增：最佳阈值
        "mAP": map_score             # 新增：mAP
    }

    # === 每类指标（可选）===
    if class_metrics:
        class_results = []
        for p, r in zip(class_prec, class_recall):
            f1_c = (0 if p == 0 and r == 0 else 2. / (1/(p+1e-8) + 1/(r+1e-8)))
            class_results.append({
                "f1": to_float(f1_c),
                "precision": to_float(p),
                "recall": to_float(r)
            })
        results = results, class_results

    return results

'''
### 3.3 Tensorboard logging (optional)

Evaluation metrics and losses produced by the `validation_loop()` function on train and validation data can be logged to a [Tensorboard `SummaryWriter`](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) which allows you to observe training graphically via the following function:
'''

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

    if train_class_results and test_class_results:
        for i in range(len(train_class_results)):
            summary_writer.add_scalars(f"Class Metrics/{class_names[i]}/Train F1 vs Test F1",
                                       {"Train F1" : train_class_results[i]["f1"],
                                        "Test F1" : test_class_results[i]["f1"]},
                                       (epoch + 1) if not mbatch_group > 0
                                             else (epoch + 1) * mbatch_count)

            summary_writer.add_scalars(f"Class Metrics/{class_names[i]}/Train Precision vs Test Precision",
                                       {"Train Precision" : train_class_results[i]["precision"],
                                        "Test Precision" : test_class_results[i]["precision"]},
                                       (epoch + 1) if not mbatch_group > 0
                                             else (epoch + 1) * mbatch_count)

            summary_writer.add_scalars(f"Class Metrics/{class_names[i]}/Train Recall vs Test Recall",
                                       {"Train Recall" : train_class_results[i]["recall"],
                                        "Test Recall" : test_class_results[i]["recall"]},
                                       (epoch + 1) if not mbatch_group > 0
                                             else (epoch + 1) * mbatch_count)
    summary_writer.flush()
'''
## 4. The skeleton of the model training and validation program

Your main program should have more or less the following sections and control flow:
'''

import os
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torch.multiprocessing as mp
import re

# === 在 main 训练循环开始之前初始化 best_f1 ===
best_f1 = 0.0
os.makedirs("checkpoints", exist_ok=True)

# === 辅助函数：获取下一个 checkpoint 文件名 ===
def get_next_checkpoint_path(folder="checkpoints"):
    existing = [f for f in os.listdir(folder) if f.startswith("checkpoint_") and f.endswith(".pth")]
    if not existing:
        next_idx = 1
    else:
        nums = [int(re.findall(r"checkpoint_(\d+)\.pth", f)[0]) for f in existing]
        next_idx = max(nums) + 1
    return os.path.join(folder, f"checkpoint_{next_idx}.pth")

# ====== 你的 Dataset, train_loop, validation_loop, update_graphs 都放在这里 ======
# (假设你已经从上面的定义复制过来了，这里不再重复)

# ---------------------------------------------------
# 自定义网络
class ResNet50_MultiLabel(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        # 加载 ImageNet 预训练权重
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # 替换最后一层
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# ---------------------------------------------------
def main():
    # ========== 超参数 ==========
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-3
    VAL_SPLIT = 0.2

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

    dataset = COCOTrainImageDataset(img_dir, annotations_dir, transform=transform)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = transform


    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4,
                              pin_memory=True, prefetch_factor=2,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4,
                            pin_memory=True, prefetch_factor=2,
                            persistent_workers=True)

    # ========== 模型 ==========
    model = ResNet50_MultiLabel(num_classes=80).to(device)

    # ========== 损失函数 & 优化器 ==========
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # ========== 日志 ==========
    best_model_path = "checkpoints/best_resnet50.pth"
    best_f1 = 0.0
    writer = SummaryWriter("runs/exp1")

    # ========== 训练循环 ==========
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        if epoch == 50:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 2e-4
                print("⚡ Learning rate manually reduced to", param_group["lr"])

        train_loop(train_loader, model, criterion, optimizer, device)

        train_eval_result = validation_loop(train_loader, model, criterion,
                                            num_classes=80, device=device,
                                            multi_task=True, one_hot=True, class_metrics=False)
        val_eval_result = validation_loop(val_loader, model, criterion,
                                          num_classes=80, device=device,
                                          multi_task=True, one_hot=True, class_metrics=False)

        print(f"Train | Loss: {train_eval_result['loss']:.4f}, "
              f"Acc: {train_eval_result['accuracy']:.4f}, "
              f"F1: {train_eval_result['f1']:.4f}, "
              f"Prec: {train_eval_result['precision']:.4f}, "
              f"Rec: {train_eval_result['recall']:.4f}," 
              f"F1(best thr={train_eval_result['best_thr']:.2f}): {train_eval_result['best_f1']:.4f}, "
              f"mAP: {train_eval_result['mAP']:.4f}")

        print(f"Val   | Loss: {val_eval_result['loss']:.4f}, "
              f"Acc: {val_eval_result['accuracy']:.4f}, "
              f"F1: {val_eval_result['f1']:.4f}, "
              f"Prec: {val_eval_result['precision']:.4f}, "
              f"Rec: {val_eval_result['recall']:.4f}, "
              f"F1(best thr={train_eval_result['best_thr']:.2f}): {train_eval_result['best_f1']:.4f}, "
              f"mAP: {train_eval_result['mAP']:.4f}")

        update_graphs(writer, epoch, train_eval_result, val_eval_result)

        # === 在每个 epoch 结束后保存 ===
        # 保存当前 epoch 的 checkpoint（始终保留）
        checkpoint_path = get_next_checkpoint_path("checkpoints")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Model checkpoint saved: {checkpoint_path}")

        # 判断是否刷新 best_model
        if val_eval_result["f1"] > best_f1:
            best_f1 = val_eval_result["f1"]
            best_model_path = os.path.join("checkpoints", "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print("✅ New best model saved with F1:", best_f1)
        else:
            print("Model not improved (F1):", val_eval_result["f1"])

    writer.close()

# ---------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
