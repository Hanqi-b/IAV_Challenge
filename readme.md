# Environment Configuration

## 1) Get the course Conda environment

Download the `env_pytorch_yolo.yml` Conda environment spec from the course page:

* **URL:** `https://moodle.insa-lyon.fr/course/view.php?id=10077`

Create the environment:

```bash
conda env create -f env_pytorch_yolo.yml
conda activate env_pytorch_yolo
```

---

## 2) Unpack the repository and required files into one folder

Create a working directory and extract the code **and** the course-provided files there. Example:


```bash
mkdir -p ~/work/coco-multilabel
cd ~/work/coco-multilabel
# clone here
# unzip here
```

---

## 3) Expected directory layout

Your folder **should look like this** (paths may vary slightly; do not include binaries or images in the submission ZIP):
```
MS_COCO_Challenge/
├─ readme.md
├─ MS_COCO.py
├─ MS_COCO_test.py
└─ ms-coco/                    
   ├─ images/
   │  ├─ train-resized/
   │  └─ test-resized/
   └─ labels/
      └─ train/            
```

# How to Run

## 1) Train (one command)

```bash
# go to the project root
cd MS_COCO_Challenge

# start training with the script’s defaults
python MS_COCO.py
```

Notes:

* The script reads images from `ms-coco/images/train-resized` and labels from `ms-coco/labels/train`.
* A new timestamped run folder is created on each launch under `checkpoints/<run_id>`, `outputs/<run_id>`, and `runs/<run_id>`.
* The best model is selected by **Macro-F1@0.50** on the validation split and saved as `best_model.pth`.

## 2) Expected folder layout after training

Example with `run_id = 20251005-011302`:

```
MS_COCO_Challenge/
├─checkpoints
│  ├─20251005-011302
│  │  ├─epoch_001.pth
│  │  ├─epoch_002.pth
│  │  └─best_model.pth
├─ms-coco
│  ├─images
│  │  ├─test-resized
│  │  └─train-resized
│  └─labels
│      └─train
├─outputs
│  └─20251005-011302
│     └─metrics.csv
└─runs
    └─20251005-011302
       └─ (TensorBoard event files)
```

## 3) Live visualization with TensorBoard

You can monitor training while it runs, or review it afterward:

```bash
# still at the project root
tensorboard --logdir runs
```

## 4) Evaluate / Export predictions JSON

**Goal:** produce `predictions.json` at a specified path, containing per-image predicted class IDs.

### 4.1 Edit paths at the top of `MS_COCO_test.py`

Open `MS_COCO_test.py` and set these three variables to your local paths:

```python
# required: best checkpoint to load
CKPT_PATH = r"checkpoints\20251005-011302\best_model.pth"  

# required: output JSON target (parent dirs will be created if missing)
OUT_JSON = "outputs/20251005-011302/predictions.json"
```

### 4.2 Run evaluation and write JSON

```bash
# from the project root
python MS_COCO_test.py
```

This writes `predictions.json` to `OUT_JSON`.
The export uses a probability threshold of `0.5` for multilabel binarization (same as during training evaluation). Adjust threshold, batch size, or workers inside `MS_COCO_test.py` if needed.

```
MS_COCO_Challenge/
├─checkpoints
│  ├─20251005-011302
├─ms-coco
│  ├─images
│  │  ├─test-resized
│  │  └─train-resized
│  └─labels
│      └─train
├─outputs
│  └─20251005-011302
└─runs
    └─20251005-011302
```
---
# Code Configuration

We train a multilabel classifier with:

* **Backbone:** `ResNet18` 
* **Batch size:** `256`
* **Strategy:** end-to-end **fine-tuning**. 
* **Threshold for binarization:** `0.35`
* **Initial LR:** `1e-3`, **halved every 30 epoch**

### Backbone and batch

We compared three backbones—**ResNet18**, **ResNet50**, and **ResNet101**—under the same training recipe .

* **ResNet18** performed best **in our setup** and could run with a **batch size of 256** on our machine.
* **ResNet101** was the most resource-intensive: it could only run with a **batch size of 32**, and its **per-epoch training time was ~4–5×** that of ResNet18.
* Based on **micro-F1**, the observed ranking was: **ResNet18 > ResNet50 > ResNet101**.

### Strategy

We adopt full **end-to-end fine-tuning** of ResNet-18 initialized from ImageNet.  
We also experimented with freezing the backbone and unfreezing only **layer3 and layer4**, or only the **last stage**, but these variants did not surpass full fine-tuning in our setup. 

### Threshold

We initially used a fixed decision threshold of **0.50**. Mid-project we learned a simple post-hoc strategy: **sweep a range of thresholds on the validation split and select the one that maximizes F1**. We did not integrate this into the training loop, but running a small sweep after training and then applying the best validation threshold (τ*) at inference yielded higher F1 in our tests. In our experiments, τ* typically settled around **0.35**, which is what we use in the main results.

### Code Acceleration

* **Preload the training set to RAM:** We modified the dataset class to support preloading images. This removes I/O bottlenecks and stabilizes throughput.

* **Parallel CPU–GPU pipeline:** The **GPU** trains (forward/backward + updates) while the **CPU**—via a multi-process `DataLoader`—**reads, decodes, augments, and prefetches** the next batch, reducing I/O stalls and boosting throughput.


