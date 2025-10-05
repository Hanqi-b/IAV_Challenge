import os, json
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
from torch import nn, optim
from PIL import Image

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights

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

# 如果你已在别处定义了 COCOTestImageDataset，可复用；这里按你之前的实现：
class COCOTestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, preload=False):
        self.img_dir = Path(img_dir)
        # 只收集 .jpg（需要别的后缀可自行加）
        self.img_list = sorted([p for p in self.img_dir.glob("*.jpg")])
        self.transform = transform
        self.preload = preload

        if len(self.img_list) == 0:
            raise FileNotFoundError(f"No .jpg found in: {self.img_dir}")

        # 可选：预加载
        self.images_cache = None
        if preload:
            self.images_cache = []
            for p in self.img_list:
                im = Image.open(p).convert("RGB")
                if self.transform:
                    im = self.transform(im)
                self.images_cache.append((im, p.stem))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.preload and (self.images_cache is not None):
            return self.images_cache[idx]

        p = self.img_list[idx]
        im = Image.open(p).convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, p.stem

def generate_predictions_json(model,
                              img_dir,
                              out_json_path,
                              device,
                              threshold: float = 0.5,
                              batch_size: int = 64,
                              num_workers: int = 4,
                              preload: bool = False):

    """
    对 img_dir 下所有 .jpg 进行多标签预测（阈值=0.5），导出 JSON：
    {
        "000000000139": [56, 60, 62],
        "000000000285": [21],
        ...
    }
    """
    # 与验证集一致的变换（确定性）
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = COCOTestImageDataset(img_dir=img_dir, transform=test_transform, preload=preload)
    loader  = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=num_workers,
                         pin_memory=True,
                         prefetch_factor=2 if num_workers > 0 else None,
                         persistent_workers=(num_workers > 0))

    model.eval()
    results = {}

    # 关闭梯度、用 AMP 推理
    autocast_enabled = (device.type == "cuda")
    with torch.inference_mode():
        for images, stems in tqdm(loader, desc="Predicting", unit="batch"):
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                logits = model(images)
                probs  = torch.sigmoid(logits)

            # 二值化（阈值=0.5），逐图转为类别 id 列表
            preds_bin = (probs >= threshold)

            for i, stem in enumerate(stems):
                idxs = torch.nonzero(preds_bin[i], as_tuple=False).squeeze(1).cpu().tolist()
                # 若需要按类别 ID 升序（一般无所谓，但更整洁）
                idxs = list(map(int, sorted(idxs)))
                results[stem] = idxs

    # 写 JSON（合法、无注释、无尾逗号）
    os.makedirs(Path(out_json_path).parent, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Predictions saved to: {out_json_path}  (images={len(results)})")


# ======== 用法示例（在训练脚本结尾或单独脚本中调用）========
if __name__ == "__main__":
    # 1) 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 构建与你训练一致的模型并加载权重
    model = ResNet18_MultiLabel(num_classes=80).to(device)
    ckpt_path = r"checkpoints\20251005-153257\best_model.pth"   # ← 改成你的权重路径
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    # 3) 指定图片目录与输出 JSON 路径
    base_path = Path(".").resolve()
    # 例如：测试集目录或你要验证的图片集目录
    img_dir = base_path / "ms-coco" / "images" / "test-resized"  # ← 按你的实际目录改
    out_json = base_path / "outputs" / "20251005-153257" / "predictions.json"

    # 4) 生成 JSON（阈值=0.5）
    generate_predictions_json(model,
                              img_dir=img_dir,
                              out_json_path=str(out_json),
                              device=device,
                              threshold=0.5,
                              batch_size=64,
                              num_workers=4,
                              preload=False)
