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
    """ResNet-18 backbone for multi-label classification (80 classes)."""
    def __init__(self, num_classes=80):
        super().__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    def forward(self, x):
        return self.base_model(x)

class ResNet50_MultiLabel(nn.Module):
    """ResNet-50 backbone for multi-label classification (80 classes)."""
    def __init__(self, num_classes=80):
        super().__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    def forward(self, x): return self.base_model(x)
    
class ResNet101_MultiLabel(nn.Module):
    """ResNet-101 backbone for multi-label classification (80 classes)."""
    def __init__(self, num_classes=80):
        super().__init__()
        self.base_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    def forward(self, x): return self.base_model(x)

class COCOTestImageDataset(Dataset):
    """
    Simple test dataset:
      - Collects *.jpg files under img_dir
      - Optional preloading into RAM
      - Returns (image_tensor, image_stem)
    """
    def __init__(self, img_dir, transform=None, preload=False):
        self.img_dir = Path(img_dir)
        # Collect only .jpg; extend if you need more suffixes
        self.img_list = sorted([p for p in self.img_dir.glob("*.jpg")])
        self.transform = transform
        self.preload = preload

        if len(self.img_list) == 0:
            raise FileNotFoundError(f"No .jpg found in: {self.img_dir}")

        # Optional: preload into RAM
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
        return im, p.stem  # filename without extension

def generate_predictions_json(model,
                              img_dir,
                              out_json_path,
                              device,
                              threshold: float = 0.5,
                              batch_size: int = 64,
                              num_workers: int = 4,
                              preload: bool = False):
    """
    Multi-label inference over all *.jpg under `img_dir` (threshold=0.5) and export to JSON:
      {
          "000000000139": [56, 60, 62],
          "000000000285": [21],
          ...
      }
    """
    # Deterministic transform aligned with validation preprocessing
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

    # Disable grads; use AMP during inference
    autocast_enabled = (device.type == "cuda")
    with torch.inference_mode():
        for images, stems in tqdm(loader, desc="Predicting", unit="batch"):
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                logits = model(images)
                probs  = torch.sigmoid(logits)

            # Binarize (threshold=0.5) and convert to class-id lists per image
            preds_bin = (probs >= threshold)

            for i, stem in enumerate(stems):
                idxs = torch.nonzero(preds_bin[i], as_tuple=False).squeeze(1).cpu().tolist()
                # Optional: sort class ids ascending for readability
                idxs = list(map(int, sorted(idxs)))
                results[stem] = idxs

    # Write JSON (valid JSON, no comments/trailing commas)
    os.makedirs(Path(out_json_path).parent, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Predictions saved to: {out_json_path}  (images={len(results)})")


# ======== Usage example (call at the end of a training script or standalone) ========
if __name__ == "__main__":
    # 1) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Build the same model as used in training, then load weights
    model = ResNet18_MultiLabel(num_classes=80).to(device)
    # To switch backbone, use one of:
    # model = ResNet50_MultiLabel(num_classes=80).to(device)
    # model = ResNet101_MultiLabel(num_classes=80).to(device)
    ckpt_path = r"checkpoints\20251005-214956\best_model.pth"   # <-- set your checkpoint path
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    # 3) Image directory and output JSON path
    base_path = Path(".").resolve()
    # e.g., a test set directory or any image folder you want to evaluate
    img_dir = base_path / "ms-coco" / "images" / "test-resized"  # <-- adjust to your real path
    out_json = base_path / "outputs" / "20251005-214956" / "predictions.json"

    # 4) Run inference and export JSON (threshold=0.5)
    generate_predictions_json(model,
                              img_dir=img_dir,
                              out_json_path=str(out_json),
                              device=device,
                              threshold=0.5,
                              batch_size=128,
                              num_workers=4,
                              preload=False)
