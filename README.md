# Multi-Label Chest X-Ray Disease Classification

**Deep learning system for automated detection of 15 thoracic diseases from chest X-ray images using EfficientNetB0 with advanced training techniques.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📊 Performance

| Metric | Value | Benchmark (Wang et al. 2017) |
|--------|-------|------------------------------|
| **Mean AUC** | **0.784** | 0.740 |
| **Improvement** | **+5.9%** | Baseline |
| **Top Disease (Edema)** | **0.884 AUC** | - |
| **Recall (Medical Priority)** | **80.3%** | - |

**Real Talk:** This isn't radiologist-level (CheXNet: 0.841 AUC), but it beats the original ChestX-ray14 paper. For a 3rd-year undergrad project, this is solid work. The dataset has 10-20% label noise (NLP-extracted, not radiologist-verified), which caps performance.

---

<p align="center">
    <img src="githubImages/per_class_auc.png" alt="Per-Class AUC Bar Chart" width="600"/>
    <br>
    <b>Figure 1. Per-Class AUC Breakdown (Edema at top, Nodule at bottom)</b>
</p>

<p align="center">
    <img src="githubImages/recall_precision_tradeoff.png" alt="Recall vs Precision Trade-off" width="600"/>
    <br>
    <b>Figure 2. Recall vs Precision Trade-off for All Diseases</b>
</p>


<p align="center">
    <img src="githubImages/train_test_distribution.png" alt="Train/Test Disease Distribution" width="800"/>
    <br>
    <b>Figure 3. Train vs Test Set Disease Distribution</b>
</p>

<p align="center">
    <img src="githubImages/gradcam_examples.png" alt="Grad-CAM Heatmaps" width="400"/>
    <br>
    <b>Figure 4. Grad-CAM Heatmaps for Model Explainability</b>
</p>

<p align="center">
    <img src="githubImages/roc_top6.png" alt="ROC Curves Top 6 Diseases" width="700"/>
    <br>
    <b>Figure 5. ROC Curves for Top 6 Diseases by AUC</b>
</p>

<p align="center">
    <img src="githubImages/prc_top6.png" alt="Precision-Recall Curves Top 6 Diseases" width="700"/>
    <br>
    <b>Figure 6. Precision-Recall Curves for Top 6 Diseases</b>
</p>

## 🎯 Dataset

**ChestX-ray14 (NIH Clinical Center)**
- 112,120 frontal-view chest X-ray images
- 30,805 unique patients
- 15 disease classes (multi-label)
- **Download:** [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC)

**Diseases:** Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax, No Finding

**⚠️ Dataset Issues (Be Aware):**
- Labels extracted via NLP from radiology reports → 10-20% noise
- Extreme class imbalance (Hernia: 110 samples vs No Finding: 60K)
- Multi-label complexity (avg 1.5 diseases per image)

---

## 🏗️ Architecture

```
Input (224x224x3)
    ↓
EfficientNetB0 (ImageNet pretrained)
    ├── All 237 layers trainable (full fine-tuning)
    └── Mixed Precision (FP16) for speed
    ↓
Global Average Pooling
    ↓
Dense(512, ReLU) → Dropout(0.3)
    ↓
Dense(256, ReLU) → Dropout(0.2)
    ↓
Dense(15, Sigmoid) [Multi-label output]
```

**Why This Works:**
- **EfficientNetB0:** SOTA efficiency (5.3M params, 0.39B FLOPs)
- **Full fine-tuning:** Medical imaging ≠ ImageNet → adapt all layers
- **Mixed precision:** 30-40% speedup, no accuracy loss

📖 **[See detailed architecture diagrams and training pipeline →](ARCHITECTURE.md)**

---

## 🔧 Training Strategy

### **1. Focal Loss (Lin et al. 2020)**
```python
focal_loss = BinaryFocalCrossentropy(alpha=0.25, gamma=2.0)
```
**Why:** Handles extreme class imbalance better than BCE. Focuses on hard-to-classify samples (rare diseases).

### **2. Balanced Oversampling**
- Rare diseases (Hernia: 110 → 2000 samples) oversampled
- Prevents model from ignoring minority classes
- **Trade-off:** Increased training time (+4%), but +12% AUC on rare diseases

### **3. Class Weights**
- Soft weighting (50% reduction factor) to avoid overfitting rare classes
- Complements Focal Loss for balanced learning

### **4. Medical-Appropriate Augmentation**
```python
- Horizontal flip (anatomically valid)
- Brightness ±10% (X-ray exposure variation)
- Contrast ±10% (detector sensitivity)
- Random zoom 0.9-1.0 (positioning variation)
```
**No rotation:** Chest X-rays have fixed orientation (heart on left).

### **5. Test-Time Augmentation (TTA)**
- 6 predictions per image (1 original + 5 augmented)
- Average predictions → +0.6% AUC boost
- **Cost:** 6x inference time (use for critical cases only)

### **6. Threshold Optimization**
- Default 0.5 → Optimized 0.2-0.45 per disease
- Target: 80% recall (medical priority)
- **Result:** False positives increase, but missing diseases is worse

---

## 📈 Results Breakdown

### **Top Performing Diseases:**
| Disease | AUC | Recall | Precision | Why Good? |
|---------|-----|--------|-----------|-----------|
| Edema | 0.884 | 80% | 43% | Clear radiological features |
| Cardiomegaly | 0.865 | 80% | 39% | Large, distinct heart silhouette |
| Effusion | 0.852 | 82% | 46% | High prevalence (2.5K samples) |

### **Worst Performing Diseases:**
| Disease | AUC | Recall | Precision | Why Bad? |
|---------|-----|--------|-----------|----------|
| Hernia | 0.612 | 75% | 18% | Only 110 samples (extreme rarity) |
| Pneumonia | 0.698 | 79% | 22% | Overlaps with Infiltration (label noise) |
| Nodule | 0.704 | 78% | 28% | Small, subtle features |

### **Honest Assessment:**
- **AUC 0.78** is good for noisy labels, but not clinic-ready
- **80% recall** is appropriate for screening (catch diseases early)
- **40% precision** means high false positives (radiologist review needed)
- This is a **screening tool**, not a diagnostic system

---

## ⚠️ Limitations (Critical)

### **1. False Positive Rate (The Elephant in the Room)**
- **Precision: 40-45%** → 55-60% false positives
- **Why:** Low thresholds (0.2-0.4) to maximize recall
- **Clinical impact:** Radiologist must review all positives (intended use)

### **2. Dataset Label Noise**
- ChestX-ray14 uses NLP extraction (not radiologist-verified)
- Estimated 10-20% mislabeling rate
- Some "diseases" are actually descriptions (e.g., "No Finding")

### **3. Class Imbalance Persists**
- Even with oversampling, rare diseases underperform
- Hernia (110 samples) vs No Finding (60K) → 500x difference
- Model biased toward common diseases

### **4. No External Validation**
- Trained and tested on same hospital (NIH Clinical Center)
- Performance will drop on external datasets (domain shift)
- Real-world deployment requires multi-site validation

### **5. Not Radiologist-Level**
- CheXNet (2017): 0.841 AUC with DenseNet-121
- This model: 0.784 AUC with EfficientNetB0
- **Gap:** 5.7% AUC → Needs more data, better labels, or ensemble

---

## 🚀 Live Demo

**Try it online:** [🤗 Hugging Face Space](https://huggingface.co/spaces/emiraran/chest-xray-classification)

Upload a chest X-ray and get instant predictions! No setup required.

---

## 💻 Local Usage

### **Installation**
```bash
pip install -r requirements.txt
```

### **Quick Inference (No Grad-CAM)**
```bash
python demo.py images/00000001_000.png
```

### **Full Inference (With Grad-CAM)**
```bash
python demo_with_gradcam.py images/00000001_000.png
# Output: Disease predictions + gradcam_*.png heatmaps
```

### **Programmatic Usage**
```python
from demo import ChestXRayPredictor

# Initialize predictor
predictor = ChestXRayPredictor(
    model_path='best_model_final.h5',
    thresholds_path='optimal_thresholds.pkl',
    label_encoder_path='label_encoder.pkl'
)

# Get predictions
results = predictor.predict('sample_xray.png', use_tta=False)
    for disease, idx in label_encoder.items():
        prob = probs[idx]
        threshold = thresholds[disease]
        if prob >= threshold:
            results.append({
                'disease': disease,
                'probability': f"{prob:.1%}",
                'confidence': 'HIGH' if prob > threshold + 0.1 else 'MEDIUM'
            })
    
    return sorted(results, key=lambda x: float(x['probability'].strip('%')), reverse=True)

# Example
predictions = predict_xray('sample_xray.png')
for p in predictions:
    print(f"{p['disease']:<20} {p['probability']:>6}  [{p['confidence']}]")
```

---

## 📁 Project Structure

```
chest-xray-classification/
├── chest_xray_analysis.ipynb      # Main notebook (training + evaluation)
├── README.md                      # This file
├── ARCHITECTURE.md                # Detailed architecture diagrams & pipeline
├── .gitignore                     # Ignore large files
├── requirements.txt               # Python dependencies
├── demo.py                        # Local inference script
├── demo_with_gradcam.py           # Local demo with Grad-CAM visualization
├── gradcam_utils.py               # Grad-CAM implementation
├── app.py                         # Gradio web interface for HF Spaces
├── best_model_final.h5            # Model weights (included)
├── best_model.h5                  # Alternative/training checkpoint (included)
├── optimal_thresholds.pkl         # Disease-specific thresholds (included)
├── label_encoder.pkl              # Disease name mapping (included)
└── images/                        # Example images / input samples (small subset)
```

**Note:** This repository includes pre-trained model artifacts for convenience. If you prefer to retrain from scratch, use the notebook to generate your own weights.

---

## 🔬 Technical Details

### **Training Configuration**
```yaml
Epochs: 50 (early stopping at epoch 46)
Batch Size: 64
Learning Rate: 1e-5 (reduced to 3.1e-7 via ReduceLROnPlateau)
Optimizer: Adam
Loss: Binary Focal Crossentropy (α=0.25, γ=2.0)
Mixed Precision: FP16
Training Time: ~3 hours (NVIDIA RTX GPU)
```

### **Data Split**
- **Patient-level split** (not image-level) to prevent data leakage
- Train: 89,826 images (24,644 patients)
- Test: 22,294 images (6,161 patients)
- **Why patient-level?** Same patient may have multiple X-rays → prevent memorization

### **Callbacks**
- **ModelCheckpoint:** Save best val_auc model
- **ReduceLROnPlateau:** Halve LR if val_loss plateaus (patience=5)
- **EarlyStopping:** Stop if val_auc plateaus (patience=10)

---

## 🎨 Grad-CAM Visualization

**NEW!** See where the model looks when making predictions:

```bash
# Generate Grad-CAM heatmaps for top 3 predictions
python demo_with_gradcam.py images/00000001_000.png

# Output: gradcam_edema.png, gradcam_cardiomegaly.png, gradcam_effusion.png
```

**What is Grad-CAM?**
- Gradient-weighted Class Activation Mapping
- Shows important regions for each disease prediction
- Red = model focuses here, Blue = model ignores
- **Use case:** Validate model isn't using spurious correlations (e.g., text artifacts)

**Reference:** Selvaraju et al. (2017) - [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

---

## 📚 References

1. **Wang et al. (2017)** - ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks  
   [Paper](https://arxiv.org/abs/1705.02315) | [Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)

2. **Rajpurkar et al. (2017)** - CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays  
   [Paper](https://arxiv.org/abs/1711.05225)

3. **Tan & Le (2019)** - EfficientNet: Rethinking Model Scaling for CNNs  
   [Paper](https://arxiv.org/abs/1905.11946)

4. **Selvaraju et al. (2017)** - Grad-CAM: Visual Explanations from Deep Networks  
   [Paper](https://arxiv.org/abs/1610.02391)

4. **Lin et al. (2020)** - Focal Loss for Dense Object Detection  
   [Paper](https://arxiv.org/abs/1708.02002)

---

## 🎓 For Recruiters / Academic Review

### **What's Good:**
✅ Beats published benchmark (+5.9% AUC)  
✅ SOTA techniques (Focal Loss, TTA, Mixed Precision, Full Fine-Tuning)  
✅ Medical-aware design (recall priority, patient-level split)  
✅ Comprehensive evaluation (ROC, PR curves, confusion matrices)  
✅ Honest limitation discussion (no BS marketing)  

### **What's Missing (Acknowledgment):**
❌ External validation (single hospital data)  
❌ Radiologist comparison (no ground truth verification)  
❌ Grad-CAM visualization (explainability)  
❌ Ensemble methods (single model only)  
❌ Production deployment (no API, no containerization)  

### **Suitable For:**
- 🎓 Undergraduate/Graduate ML coursework
- 📝 Academic paper (with external validation)
- 💼 Portfolio project for ML engineer roles
- 🏥 Research prototype (NOT clinical deployment)

### **NOT Suitable For:**
- ❌ Clinical decision-making (FDA/CE approval required)
- ❌ Standalone diagnosis (must be radiologist-assisted)
- ❌ Real-time emergency screening (inference time ~200ms per image)

---

## 🤝 Contributing

This is an academic project. If you find issues or have improvements:
1. Fork the repo
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

**Dataset License:** NIH ChestX-ray14 dataset is public domain (U.S. Government work). Please cite the original paper if you use this work.

---

## 🙏 Acknowledgments

- NIH Clinical Center for ChestX-ray14 dataset
- Original paper authors (Wang et al., 2017)
- TensorFlow team for EfficientNet implementation
- Medical imaging community for open research

---

## 📧 Contact

**Author:** Emir Muhammet Aran  
**Institution:** Computer Engineering Student  
**GitHub:** [github.com/EmirMuhammetARAN](https://github.com/EmirMuhammetARAN)

---

## ⚡ Quick Start

Windows PowerShell (recommended):

```powershell
# 1. Clone repo
git clone https://github.com/EmirMuhammetARAN/chest-xray-disease-classification.git
cd chest-xray-disease-classification

# 2. Create and activate virtual environment (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the Gradio app (local demo)
python app.py

# 5. Quick inference examples
python demo.py                # runs the included demo flow
python demo_with_gradcam.py images/00000001_000.png  # generate Grad-CAM images
```

Unix / WSL / macOS:

```bash
# 1. Clone repo
git clone https://github.com/EmirMuhammetARAN/chest-xray-disease-classification.git
cd chest-xray-disease-classification

# 2. Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run notebook or demos
jupyter notebook chest_xray_analysis.ipynb
python demo.py
python demo_with_gradcam.py images/00000001_000.png
```

Notes:
- Model artifact files present in repo: `best_model_final.h5`, `best_model.h5`, `optimal_thresholds.pkl`, `label_encoder.pkl`.
- If you prefer to retrain the model yourself, open `chest_xray_analysis.ipynb` and follow the training cells (GPU recommended).

## ✅ Reproducibility & Tests

- The primary training pipeline is in `chest_xray_analysis.ipynb`. Run the notebook in order to reproduce preprocessing, training, and threshold selection.
- For deterministic runs: fix random seeds (`numpy`, `tensorflow`, `python`), run on the same TF/CUDA versions, and use the same data split (patient-level split is required).
- Add tests under `test/` for small utility functions and model I/O. Keep CI fast by mocking heavy model calls.

## 🛠 Troubleshooting

- Missing model files: ensure `best_model_final.h5`, `optimal_thresholds.pkl`, and `label_encoder.pkl` are in the project root. `demo.py` will print missing files if any are absent.
- GPU errors: verify your CUDA/cuDNN versions match the installed TensorFlow version. Use `pip show tensorflow` and consult TensorFlow release notes.
- Slow inference: enable mixed precision or run with reduced TTA (`use_tta=False`). For batch inference, use larger batch sizes where memory permits.
- Grad-CAM fails: some model builds use different last-conv layer names. `gradcam_utils.get_last_conv_layer_name()` attempts to find a compatible layer; update `LAST_CONV_LAYER` in `app.py` if necessary.

## 📚 How to Cite

If you use this work in academic research, please cite the original ChestX-ray14 dataset paper (Wang et al., 2017) and this repository in the methods section.

Example citation:

```
Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. 2017.
Repository: EmirMuhammetARAN/chest-xray-disease-classification (GitHub).
```

## 🔒 Privacy & Data

- The ChestX-ray14 dataset is public-domain (U.S. government data). Do not commit patient-identifiable data to this repository.
- When sharing results, follow the original dataset license and acknowledge the NIH dataset.


---

**Last Updated:** December 2025  
**Status:** ✅ Training complete | 📊 AUC 0.784 | 🎓 Academic project

---

## 🔥 Honest Takeaway

**This model works, but it's not magic.**

- It beats the 2017 baseline → Good engineering
- It has 60% false positives → Needs radiologist review
- It costs $0.50/1000 images (GPU inference) → Economical screening
- It's NOT FDA-approved → Research only

**Use case:** Pre-screen X-rays → flag suspicious cases → radiologist reviews positives.  
**Don't use for:** Standalone diagnosis, emergency triage, legal liability scenarios.

**Bottom line:** Solid ML engineering with realistic expectations. That's how you build trust in AI.
