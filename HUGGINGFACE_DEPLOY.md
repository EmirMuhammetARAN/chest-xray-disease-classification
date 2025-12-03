# Hugging Face Deployment Guide

## 🚀 How to Deploy to Hugging Face Spaces

### **Step 1: Create Hugging Face Account**
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up / Log in
3. Go to your profile → **Spaces** → **Create new Space**

---

### **Step 2: Configure Space**

**Space Settings:**
- **Space name:** `chest-xray-classification`
- **License:** MIT
- **SDK:** Gradio
- **Hardware:** CPU Basic (Free) - Sufficient for inference
- **Visibility:** Public (or Private)

---

### **Step 3: Prepare Files**

**Files to upload:**
```
chest-xray-classification/
├── app.py                      # Main Gradio app with Grad-CAM (REQUIRED)
├── gradcam_utils.py            # Grad-CAM implementation (REQUIRED)
├── requirements.txt            # Dependencies (rename from requirements_hf.txt)
├── README.md                   # Space description (rename from README_HF.md)
├── best_model_final.h5         # Model weights (REQUIRED)
├── optimal_thresholds.pkl      # Thresholds (REQUIRED)
├── label_encoder.pkl           # Label mapping (REQUIRED)
└── examples/                   # Sample X-rays (optional)
    ├── normal.png
    └── pneumonia.png
```

**⚠️ IMPORTANT:**
- Rename `requirements_hf.txt` → `requirements.txt`
- Rename `README_HF.md` → `README.md`
- Model files MUST be included (Git LFS will handle large files)
- **NEW:** `gradcam_utils.py` is required for Grad-CAM visualization

---

### **Step 4: Upload via Git (Recommended)**

```bash
# 1. Clone your space
git clone https://huggingface.co/spaces/emiraran/chest-xray-classification
cd chest-xray-classification

# 2. Install Git LFS (for large model files)
git lfs install

# 3. Track large files
git lfs track "*.h5"
git lfs track "*.pkl"
git add .gitattributes

# 4. Copy files
cp ../app.py .
cp ../gradcam_utils.py .
cp ../requirements_hf.txt requirements.txt
cp ../README_HF.md README.md
cp ../best_model_final.h5 .
cp ../optimal_thresholds.pkl .
cp ../label_encoder.pkl .

# 5. Commit and push
git add .
git commit -m "Initial commit: Chest X-Ray Classifier"
git push
```

---

### **Step 5: Alternative - Web Upload**

If Git LFS doesn't work:

1. Go to your Space on Hugging Face
2. Click **Files** tab → **Add file** → **Upload files**
3. Drag & drop:
   - `app.py`
   - `gradcam_utils.py` (NEW!)
   - `requirements.txt` (renamed!)
   - `README.md` (renamed!)
   - `best_model_final.h5` (may take time - ~500MB)
   - `optimal_thresholds.pkl`
   - `label_encoder.pkl`
4. Click **Commit changes**

**Note:** Large files (>10MB) require Git LFS. If upload fails, use Git method.

---

### **Step 6: Wait for Build**

- Space will automatically build (takes 2-5 minutes)
- Check **Logs** tab for errors
- Once "Running", your app is live! 🎉

---

### **Step 7: Test & Share**

**Test:**
1. Upload a chest X-ray image
2. Click "Analyze X-Ray"
3. Verify results display correctly

**Share:**
- **Direct link:** `https://huggingface.co/spaces/emiraran/chest-xray-classification`
- **Embed:** Copy embed code from Space settings
- **LinkedIn/Portfolio:** Link to your Space!

---

## 🐛 Troubleshooting

### **Issue: Model file too large**
```bash
# Solution: Use Git LFS
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add best_model_final.h5
git commit -m "Add model with LFS"
git push
```

### **Issue: Out of memory**
- **Solution:** Upgrade to CPU Upgrade ($0/month) or GPU (paid)
- Or reduce model size (use model quantization)

### **Issue: Dependencies not installing**
```python
# Check requirements.txt has correct versions
gradio>=4.0.0
tensorflow>=2.10.0,<2.11.0
numpy>=1.23.0,<1.24.0
Pillow>=9.5.0
```

### **Issue: App crashes on prediction**
- Check Logs tab for errors
- Verify model files loaded correctly
- Test locally first: `python app.py`

---

## 💡 Tips

**Performance:**
- CPU Basic (free) is sufficient (~2-3 sec per image)
- TTA increases time to ~10 sec (5x predictions)
- GPU not needed for single-image inference

**Cost:**
- CPU Basic: **FREE** ✅
- CPU Upgrade (2 vCPU): $0.03/hour (~$22/month if always on)
- GPU: $0.60/hour+ (overkill for this use case)

**Best Practices:**
1. Test locally before deploying (`python app.py`)
2. Add example images to `/examples/` folder
3. Update README with your name, university, GitHub link
4. Share on LinkedIn with demo link! 🚀

---

## 📊 What to Put on LinkedIn

```
🚀 Just deployed my chest X-ray disease classifier to Hugging Face!

🔗 Try it live: https://huggingface.co/spaces/YOURNAME/chest-xray-classification

📊 Key Features:
• Detects 15 thoracic diseases from X-rays
• 0.784 AUC (beats 2017 baseline)
• 80% recall (medical priority)
• Instant predictions via Gradio UI

🛠️ Tech Stack:
• TensorFlow + EfficientNetB0
• Focal Loss + Test-Time Augmentation
• Deployed on HF Spaces (free tier!)

Code & notebook: github.com/yourname/chest-xray-classification

#MachineLearning #HealthcareAI #DeepLearning #HuggingFace
```

---

## ✅ Checklist

Before deploying:
- [ ] Rename `requirements_hf.txt` → `requirements.txt`
- [ ] Rename `README_HF.md` → `README.md`
- [ ] Update `app.py` with your name/university
- [ ] Update `README.md` with your GitHub link
- [ ] Test locally: `python app.py`
- [ ] Copy model files (`best_model_final.h5`, `.pkl` files)
- [ ] Create Hugging Face Space
- [ ] Upload files (Git LFS or web)
- [ ] Wait for build (check Logs)
- [ ] Test live demo
- [ ] Share on LinkedIn! 🎉

---

**Good luck! 🚀**
