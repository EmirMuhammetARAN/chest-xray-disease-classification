# Architecture Diagram

```
INPUT IMAGE (224x224x3)
         |
         v
┌─────────────────────────────────────┐
│   EfficientNetB0 Backbone           │
│   (ImageNet pretrained)             │
│                                     │
│   • 237 trainable layers            │
│   • Compound scaling (d/w/r)        │
│   • MBConv blocks (Inverted ResNet) │
│   • Squeeze-Excitation attention    │
│   • Swish activation                │
└─────────────────────────────────────┘
         |
         v (1280 features)
┌─────────────────────────────────────┐
│   Global Average Pooling            │
└─────────────────────────────────────┘
         |
         v (1280 → 512)
┌─────────────────────────────────────┐
│   Dense(512) + ReLU                 │
│   Dropout(0.3)                      │
└─────────────────────────────────────┘
         |
         v (512 → 256)
┌─────────────────────────────────────┐
│   Dense(256) + ReLU                 │
│   Dropout(0.2)                      │
└─────────────────────────────────────┘
         |
         v (256 → 15)
┌─────────────────────────────────────┐
│   Dense(15) + Sigmoid               │
│   (Multi-label output)              │
└─────────────────────────────────────┘
         |
         v
  [15 Disease Probabilities]
  
  Atelectasis, Cardiomegaly, 
  Consolidation, Edema, Effusion,
  Emphysema, Fibrosis, Hernia,
  Infiltration, Mass, Nodule,
  Pleural_Thickening, Pneumonia,
  Pneumothorax, No Finding
```

---

## Training Pipeline

```
NIH ChestX-ray14 Dataset (112,120 images)
         |
         v
┌──────────────────────────────────────┐
│  Data Preprocessing                  │
│  • Resize to 224x224                 │
│  • Normalize (0-1)                   │
│  • Patient-level split (80/20)       │
└──────────────────────────────────────┘
         |
         v
┌──────────────────────────────────────┐
│  Class Balancing                     │
│  • Oversampling (min 2000 samples)   │
│  • Class weights (inverse frequency)  │
└──────────────────────────────────────┘
         |
         v
┌──────────────────────────────────────┐
│  Data Augmentation                   │
│  • Horizontal flip (p=0.5)           │
│  • Rotation (±10°)                   │
│  • Zoom (±10%)                       │
│  • Brightness (±10%)                 │
└──────────────────────────────────────┘
         |
         v
┌──────────────────────────────────────┐
│  Training Configuration              │
│  • Loss: Focal Loss (α=0.25, γ=2.0) │
│  • Optimizer: Adam (LR=1e-4)         │
│  • Batch size: 32                    │
│  • Mixed precision: FP16             │
│  • Epochs: 50 (early stop at 46)     │
└──────────────────────────────────────┘
         |
         v
┌──────────────────────────────────────┐
│  Evaluation & Threshold Optimization │
│  • ROC-AUC per disease               │
│  • F1-score per disease              │
│  • Optimal thresholds (max F1)       │
│  • Test-Time Augmentation (TTA)      │
└──────────────────────────────────────┘
         |
         v
    FINAL MODEL
    (AUC 0.784)
```

---

## Inference Pipeline

```
INPUT: Chest X-Ray Image
         |
         v
┌──────────────────────────────────────┐
│  Preprocessing                       │
│  • Convert to RGB                    │
│  • Resize to 224x224                 │
│  • Normalize (÷255)                  │
└──────────────────────────────────────┘
         |
         v
    ┌───────────────────────────┐
    │   TTA Enabled?            │
    └───────────────────────────┘
         |                |
         NO              YES
         |                |
         v                v
    ┌─────────┐   ┌──────────────────┐
    │ Single  │   │ 6 Augmentations: │
    │Predict  │   │ • Original       │
    └─────────┘   │ • HFlip          │
         |        │ • Brightness ±   │
         |        │ • Average preds  │
         |        └──────────────────┘
         |                |
         v                v
┌──────────────────────────────────────┐
│  Post-processing                     │
│  • Apply optimal thresholds          │
│  • Filter low-confidence predictions │
│  • Sort by probability               │
└──────────────────────────────────────┘
         |
         v
    ┌───────────────────────────┐
    │ Grad-CAM Enabled?         │
    └───────────────────────────┘
         |                |
         NO              YES
         |                |
         v                v
    ┌─────────┐   ┌──────────────────┐
    │ Return  │   │ Generate heatmap │
    │ Results │   │ for top 3 preds  │
    └─────────┘   │ • Last conv layer│
         |        │ • Overlay on img │
         |        └──────────────────┘
         |                |
         v                v
OUTPUT: Disease Predictions + Grad-CAM
```

---

## Model Architecture Details

| Component | Details |
|-----------|---------|
| **Input** | 224x224x3 RGB image |
| **Backbone** | EfficientNetB0 (237 layers, 5.3M params) |
| **Base Model Training** | Full fine-tuning (all layers trainable) |
| **Pooling** | Global Average Pooling |
| **Hidden Layer 1** | Dense(512) + ReLU + Dropout(0.3) |
| **Hidden Layer 2** | Dense(256) + ReLU + Dropout(0.2) |
| **Output Layer** | Dense(15) + Sigmoid (multi-label) |
| **Total Parameters** | ~5.8M (all trainable) |
| **Mixed Precision** | FP16 (2x faster training) |

---

## Performance by Disease

| Disease | AUC | Precision | Recall | F1-Score | Training Samples |
|---------|-----|-----------|--------|----------|------------------|
| **Edema** | 0.884 | 0.52 | 0.85 | 0.65 | 2,000 |
| **Cardiomegaly** | 0.865 | 0.48 | 0.82 | 0.61 | 2,776 |
| **Effusion** | 0.852 | 0.44 | 0.80 | 0.57 | 13,317 |
| **Mass** | 0.824 | 0.41 | 0.79 | 0.54 | 5,782 |
| **Pneumothorax** | 0.815 | 0.39 | 0.78 | 0.52 | 5,302 |
| **Consolidation** | 0.803 | 0.37 | 0.76 | 0.50 | 4,667 |
| **Pneumonia** | 0.792 | 0.35 | 0.75 | 0.48 | 2,000 |
| **Atelectasis** | 0.781 | 0.33 | 0.74 | 0.46 | 11,559 |
| **Nodule** | 0.770 | 0.31 | 0.73 | 0.43 | 6,331 |
| **Infiltration** | 0.752 | 0.28 | 0.71 | 0.40 | 19,894 |
| **Pleural Thickening** | 0.741 | 0.26 | 0.69 | 0.38 | 3,385 |
| **Emphysema** | 0.721 | 0.23 | 0.67 | 0.34 | 2,516 |
| **Fibrosis** | 0.695 | 0.20 | 0.64 | 0.30 | 2,000 |
| **No Finding** | 0.690 | 0.18 | 0.62 | 0.28 | 60,361 |
| **Hernia** | 0.612 | 0.10 | 0.55 | 0.17 | 227 |
| **MEAN** | **0.784** | **0.34** | **0.73** | **0.46** | - |

**Key Observations:**
- Best performance on **edema, cardiomegaly, effusion** (large, clear visual patterns)
- Worst performance on **hernia** (very few training samples: 227)
- High recall (73%) but low precision (34%) → **Many false positives**
- Trade-off intentional: Better to catch diseases than miss them

---

## Comparison with Baseline

| Model | Architecture | AUC (Mean) | Year | Notes |
|-------|--------------|------------|------|-------|
| **This Model** | EfficientNetB0 | **0.784** | 2025 | Full fine-tuning, Focal Loss, TTA |
| Wang et al. | ResNet-50 | 0.740 | 2017 | Original NIH paper, baseline |
| Rajpurkar et al. | DenseNet-121 | 0.841 | 2018 | CheXNet (external test set) |
| Irvin et al. | DenseNet-121 | 0.763 | 2019 | CheXpert (different dataset) |

**Improvement over baseline:** +5.9% (0.784 vs 0.740)

---

## Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) shows **where the model looks** when making predictions.

**How it works:**
1. Extract activations from last convolutional layer
2. Compute gradients of predicted class w.r.t. activations
3. Weight activations by gradients
4. Generate heatmap (red = important, blue = not important)
5. Overlay heatmap on original X-ray

**Example:**
```
Original X-Ray          Grad-CAM Heatmap        Overlay
    ┌─────┐                 ┌─────┐             ┌─────┐
    │     │                 │🔴🔴 │             │🔥🔥 │
    │  ◯  │  ───────────>   │🔴🔴 │  ───────>   │ ◯🔥 │
    │     │                 │     │             │     │
    └─────┘                 └─────┘             └─────┘
  (Chest X-Ray)        (Model attention)    (Interpretable)
```

**Benefits:**
- **Trust:** See if model looks at the right regions
- **Debugging:** Detect spurious correlations (e.g., text artifacts)
- **Clinical value:** Radiologists can validate model's reasoning

---

## Technical Stack

- **Deep Learning:** TensorFlow 2.10, Keras API
- **Architecture:** EfficientNetB0 (Tan & Le, 2019)
- **Loss Function:** Focal Loss (Lin et al., 2017)
- **Visualization:** Grad-CAM (Selvaraju et al., 2017)
- **Deployment:** Gradio 4.0, Hugging Face Spaces
- **Hardware:** GPU (mixed precision FP16)

---

## Future Improvements

### High Priority (1-2 weeks)
- [ ] **External validation** on CheXpert or MIMIC-CXR
- [ ] **Ensemble models** (EfficientNetB0 + DenseNet121)
- [ ] **Multi-view support** (frontal + lateral X-rays)

### Medium Priority (1 month)
- [ ] **Grad-CAM++** or **Score-CAM** for better localization
- [ ] **Bounding box regression** for disease localization
- [ ] **Report generation** (findings → radiology report)

### Low Priority (3+ months)
- [ ] **3D models** for CT scans
- [ ] **Weakly-supervised segmentation** (disease masks)
- [ ] **Multi-task learning** (age/sex/smoking status prediction)

---

## References

1. **Wang et al. (2017)** - ChestX-ray8: Hospital-scale Chest X-ray Database  
   https://arxiv.org/abs/1705.02315

2. **Tan & Le (2019)** - EfficientNet: Rethinking Model Scaling for CNNs  
   https://arxiv.org/abs/1905.11946

3. **Lin et al. (2017)** - Focal Loss for Dense Object Detection  
   https://arxiv.org/abs/1708.02002

4. **Selvaraju et al. (2017)** - Grad-CAM: Visual Explanations from Deep Networks  
   https://arxiv.org/abs/1610.02391

5. **Rajpurkar et al. (2018)** - CheXNet: Radiologist-Level Pneumonia Detection  
   https://arxiv.org/abs/1711.05225
