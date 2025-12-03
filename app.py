"""
Chest X-Ray Disease Classification - Hugging Face Demo
=======================================================

Multi-label classification of 15 thoracic diseases from chest X-rays.

Author: Emir Muhammet Aran
Model: EfficientNetB0 (AUC 0.784)
Dataset: NIH ChestX-ray14
"""

import gradio as gr
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from gradcam_utils import generate_gradcam_for_top_predictions, get_last_conv_layer_name


# ============================================================================
# MODEL LOADING
# ============================================================================

def build_model(num_classes=15):
    """Rebuild EfficientNetB0 architecture"""
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB0
    
    IMG_SIZE = 224
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        pooling='avg'
    )
    
    x = base_model.output
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', dtype='float32')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Load model components
print("Loading model...")
model = build_model(num_classes=15)
model.load_weights('best_model_final.h5')

with open('optimal_thresholds.pkl', 'rb') as f:
    optimal_thresholds = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("✅ Model loaded successfully!")


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_xray(image, use_tta=False):
    """
    Predict diseases from chest X-ray image.
    
    Args:
        image: PIL Image or numpy array
        use_tta: Use Test-Time Augmentation (slower but more accurate)
    
    Returns:
        HTML formatted results
    """
    try:
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize and normalize
        image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        # Predict
        if use_tta:
            # Test-Time Augmentation (5 predictions)
            predictions = []
            predictions.append(model.predict(img_array, verbose=0)[0])
            
            for _ in range(4):
                # Random horizontal flip
                aug_img = tf.image.random_flip_left_right(img_array)
                aug_img = tf.image.random_brightness(aug_img, max_delta=0.1)
                aug_img = tf.clip_by_value(aug_img, 0.0, 1.0)
                predictions.append(model.predict(aug_img.numpy(), verbose=0)[0])
            
            probs = np.mean(predictions, axis=0)
        else:
            probs = model.predict(img_array, verbose=0)[0]
        
        # Apply thresholds and format results
        results = []
        for disease, idx in label_encoder.items():
            prob = float(probs[idx])
            threshold = optimal_thresholds[disease]
            
            if prob >= threshold:
                confidence_score = min((prob - threshold) / (1 - threshold), 1.0)
                confidence = 'HIGH' if confidence_score > 0.5 else 'MEDIUM'
                
                results.append({
                    'disease': disease,
                    'probability': prob,
                    'confidence': confidence
                })
        
        # Sort by probability
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        
        # Generate Grad-CAM for top 3 predictions if enabled
        gradcam_images = None
        if use_tta and results:  # Use TTA checkbox to toggle Grad-CAM
            try:
                last_conv_layer = get_last_conv_layer_name(model)
                gradcam_images = generate_gradcam_for_top_predictions(
                    image, model, results, label_encoder, top_k=min(3, len(results)), 
                    last_conv_layer_name=last_conv_layer
                )
            except Exception as e:
                print(f"Grad-CAM generation failed: {e}")
                gradcam_images = None
        
        # Format output
        if not results:
            html_output = """
            <div style="padding: 20px; background: #d4edda; border: 2px solid #28a745; border-radius: 10px;">
                <h2 style="color: #155724; margin-top: 0;">✅ NO ABNORMALITIES DETECTED</h2>
                <p style="color: #155724;">All disease probabilities are below the optimized thresholds.</p>
                <p style="color: #666; font-size: 0.9em; margin-bottom: 0;">
                    <strong>Note:</strong> This model prioritizes recall (80%), so low-probability findings are filtered out.
                </p>
            </div>
            """
        else:
            html_output = f"""
            <div style="padding: 20px; background: #fff3cd; border: 2px solid #ffc107; border-radius: 10px;">
                <h2 style="color: #856404; margin-top: 0;">⚠️ {len(results)} POTENTIAL FINDING(S) DETECTED</h2>
                <div style="margin: 15px 0;">
            """
            
            for i, r in enumerate(results, 1):
                prob_pct = f"{r['probability'] * 100:.1f}%"
                conf_color = '#28a745' if r['confidence'] == 'HIGH' else '#ffc107'
                
                html_output += f"""
                <div style="padding: 12px; margin: 8px 0; background: white; border-left: 4px solid {conf_color}; border-radius: 5px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold; font-size: 1.1em;">{i}. {r['disease']}</span>
                        <span style="background: {conf_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em;">
                            {r['confidence']}
                        </span>
                    </div>
                    <div style="margin-top: 8px;">
                        <span style="color: #666;">Probability: </span>
                        <span style="font-weight: bold; color: #333;">{prob_pct}</span>
                    </div>
                </div>
                """
            
            html_output += """
                </div>
            </div>
            """
        
        # Add disclaimer
        html_output += """
        <div style="margin-top: 20px; padding: 15px; background: #f8d7da; border: 2px solid #f5c6cb; border-radius: 10px;">
            <h3 style="color: #721c24; margin-top: 0; font-size: 1em;">⚠️ IMPORTANT DISCLAIMER</h3>
            <p style="color: #721c24; margin: 8px 0; font-size: 0.9em;">
                <strong>This is a research prototype. NOT for clinical diagnosis.</strong>
            </p>
            <ul style="color: #721c24; margin: 8px 0; font-size: 0.85em; padding-left: 20px;">
                <li>Model achieves 0.784 AUC (80% recall, 40% precision)</li>
                <li>High false positive rate by design (prioritizes catching diseases)</li>
                <li>Dataset has 10-20% label noise (NLP-extracted labels)</li>
                <li>Always consult a qualified radiologist for medical diagnosis</li>
            </ul>
        </div>
        """
        
        # Return both HTML and Grad-CAM images
        if gradcam_images:
            return html_output, gradcam_images[0][1], gradcam_images[1][1] if len(gradcam_images) > 1 else None, gradcam_images[2][1] if len(gradcam_images) > 2 else None
        else:
            return html_output, None, None, None
        
    except Exception as e:
        error_html = f"""
        <div style="padding: 20px; background: #f8d7da; border: 2px solid #f5c6cb; border-radius: 10px;">
            <h2 style="color: #721c24; margin-top: 0;">❌ ERROR</h2>
            <p style="color: #721c24;">Failed to process image: {str(e)}</p>
            <p style="color: #666; font-size: 0.9em;">
                Please ensure the image is a valid chest X-ray (PNG/JPEG format).
            </p>
        </div>
        """
        return error_html, None, None, None


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS
custom_css = """
#component-0 {
    max-width: 900px;
    margin: auto;
}
.output-html {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
"""

# Example images (optional - add if you have sample X-rays)
examples = [
    # ["examples/normal.png"],
    # ["examples/pneumonia.png"],
]

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Chest X-Ray Disease Classifier") as demo:
    gr.Markdown(
        """
        # 🏥 Chest X-Ray Disease Classification
        
        **Multi-label detection of 15 thoracic diseases using EfficientNetB0**
        
        Upload a frontal chest X-ray image to detect potential abnormalities.
        
        **Performance:** Mean AUC 0.784 | 80% Recall | Trained on 112K X-rays (NIH ChestX-ray14)
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Chest X-Ray",
                type="pil",
                height=400
            )
            
            tta_checkbox = gr.Checkbox(
                label="Enable Grad-CAM Visualization",
                value=False,
                info="Show where the model looks (enables TTA for better accuracy)"
            )
            
            predict_btn = gr.Button(
                "🔍 Analyze X-Ray",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            output_html = gr.HTML(
                label="Results",
                elem_classes="output-html"
            )
    
    # Grad-CAM visualizations
    with gr.Row(visible=True):
        gradcam_1 = gr.Image(label="🔥 Grad-CAM #1 (Top Prediction)", type="pil")
        gradcam_2 = gr.Image(label="🔥 Grad-CAM #2", type="pil")
        gradcam_3 = gr.Image(label="🔥 Grad-CAM #3", type="pil")
    
    # Examples section (if you have sample images)
    if examples:
        gr.Examples(
            examples=examples,
            inputs=image_input,
            outputs=output_html,
            fn=predict_xray,
            cache_examples=False
        )
    
    gr.Markdown(
        """
        ---
        
        ## 📊 About This Model
        
        **Architecture:** EfficientNetB0 with full fine-tuning (237 layers)  
        **Training:** Focal Loss + Balanced Sampling + Mixed Precision (FP16)  
        **Dataset:** NIH ChestX-ray14 (112,120 images from 30,805 patients)  
        
        **Detected Diseases (15 classes):**
        - Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion
        - Emphysema, Fibrosis, Hernia, Infiltration, Mass
        - Nodule, Pleural Thickening, Pneumonia, Pneumothorax, No Finding
        
        **Performance by Disease:**
        - Best: Edema (0.884 AUC), Cardiomegaly (0.865 AUC), Effusion (0.852 AUC)
        - Worst: Hernia (0.612 AUC - only 110 training samples)
        
        **Limitations:**
        - High false positive rate (60%) by design to maximize recall
        - Dataset has label noise (NLP-extracted from reports)
        - Single-site training (NIH) - may not generalize to other hospitals
        - NOT FDA-approved or clinically validated
        
        ---
        
        ## 🔗 Links
        
        - **Dataset:** [NIH ChestX-ray14 on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
        - **Code:** [GitHub Repository](https://github.com/emirmuhammmetaran/chest-xray-classification)
        - **Paper:** [Wang et al. 2017](https://arxiv.org/abs/1705.02315)
        
        ---
        
        **Built by:** Emir Muhammet Aran | **Institution:** Computer Engineering Student  
        **Last Updated:** December 2025
        """
    )
    
    # Connect button to prediction function
    predict_btn.click(
        fn=predict_xray,
        inputs=[image_input, tta_checkbox],
        outputs=[output_html, gradcam_1, gradcam_2, gradcam_3]
    )

# Launch app
if __name__ == "__main__":
    demo.launch()
