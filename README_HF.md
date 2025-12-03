---
title: Chest X-Ray Disease Classification
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - medical-imaging
  - computer-vision
  - multi-label-classification
  - chest-xray
  - deep-learning
  - tensorflow
  - efficientnet
short_description: 15 thoracic diseases detection (AUC 0.784)
---

# Chest X-Ray Disease Classification 🏥

**Automated detection of 15 thoracic diseases from chest X-ray images using deep learning.**

## 🎯 Performance

- **Mean AUC:** 0.784 (beats 2017 baseline by +5.9%)
- **Recall:** 80.3% (medical priority - catch diseases early)
- **Architecture:** EfficientNetB0 with full fine-tuning
- **Dataset:** NIH ChestX-ray14 (112,120 images)

## 🔬 Model Details

**Training:**
- Focal Loss for class imbalance
- Balanced sampling (oversampling rare diseases)
- Test-Time Augmentation (TTA)
- Mixed Precision (FP16)
- Patient-level train/test split

**Best Performing Diseases:**
- Edema: 0.884 AUC
- Cardiomegaly: 0.865 AUC  
- Effusion: 0.852 AUC

**🔥 NEW! Grad-CAM Visualization:**
- Enable the checkbox to see **where the model looks**
- Red regions = High attention (important for prediction)
- Blue regions = Low attention (ignored by model)
- Helps validate the model isn't using spurious features

## ⚠️ Limitations

**IMPORTANT:** This is a research prototype. NOT for clinical diagnosis.

- **High false positive rate (60%)** by design to maximize recall
- Dataset has label noise (NLP-extracted from reports)
- Single-site training (NIH) - may not generalize
- Requires radiologist review for all predictions
- NOT FDA-approved or clinically validated

## 📊 Use Case

**Intended:** Screening tool to flag suspicious X-rays for radiologist review  
**NOT intended:** Standalone diagnosis, emergency triage, legal liability scenarios

## 🔗 Resources

- **Dataset:** [NIH ChestX-ray14 on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Code:** [GitHub Repository](https://github.com/emirmuhammmetaran/chest-xray-classification)
- **Paper:** [Wang et al. 2017 - ChestX-ray14 Dataset](https://arxiv.org/abs/1705.02315)

## 📄 Citation

```bibtex
@article{wang2017chestxray14,
  title={Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2097--2106},
  year={2017}
}
```

---

**Author:** Emir Muhammet Aran | **Institution:** Computer Engineering Student  
**Last Updated:** December 2025
