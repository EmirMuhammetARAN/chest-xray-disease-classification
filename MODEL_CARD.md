# Model Card — Chest X-Ray Disease Classification (EfficientNetB0)

Model: `best_model_final.h5`
Owner: Emir Muhammet Aran (GitHub: EmirMuhammetARAN)
License: MIT

## Short description
A multi-label chest X‑ray classifier trained on the NIH ChestX-ray14 dataset to detect 15 thoracic findings (e.g., Edema, Effusion, Pneumothorax). Model architecture: EfficientNetB0 (full fine-tuning) with custom dense heads and mixed precision training.

## Intended use
- Screening / triage aid: highlight likely abnormal studies for radiologist prioritization.
- Research and educational purposes: reproducible pipeline for multi-label chest X‑ray classification.

## Not intended use
- Not for clinical diagnosis or treatment decisions.
- Not approved for clinical deployment (no regulatory clearance).

## Factors and caveats
- Training data: NIH ChestX-ray14 (NLP-extracted labels; estimated 10–20% label noise).
- Patient population: Single hospital (NIH Clinical Center); performance may degrade on external sites due to domain shift.
- Image types: Frontal view chest X‑rays only. Lateral views not supported.

## Metrics
- Mean AUC (reported): 0.784 (per README)
- Example per-class AUCs: Edema 0.884, Cardiomegaly 0.865, Effusion 0.852.
- Thresholds: disease-specific thresholds saved in `optimal_thresholds.pkl` (used to convert probabilities to binary positives).

## Training data
- Dataset: NIH ChestX-ray14 (112k frontal X‑ray images)
- Split: patient-level train/test split (prevents leakage across images of same patient)
- Preprocessing: resize to 224×224, normalization to [0,1], medical-aware augmentations (horizontal flip, brightness/contrast, zoom)

## Evaluation procedure
- Primary metric: per-class AUC (ROC), mean AUC reported across 15 classes
- Test-time augmentation (TTA) used optionally for inference (6 augmentations)

## Ethical considerations and limitations
- Label noise: Many labels come from report-level NLP and are not radiologist-verified, which reduces the reliability of per-class labels.
- False positives: To prioritize recall, thresholds produce a relatively high false positive rate (precision ~40%), meaning radiologist review is required for positives.
- External validity: The model was evaluated on NIH data only — validate on local data before clinical use.

## How to use (quick)
```bash
# Run Gradio app locally
python app.py

# Single image inference
python demo.py

# Grad-CAM visualization
python demo_with_gradcam.py images/00000001_000.png
```

## Model files
- `best_model_final.h5` — weights used by demo scripts
- `optimal_thresholds.pkl` — disease-specific thresholds
- `label_encoder.pkl` — mapping disease name -> index

## Contact
For questions, open an issue or contact the repository owner: https://github.com/EmirMuhammetARAN
