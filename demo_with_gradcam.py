"""
Local Demo Script with Grad-CAM Visualization
==============================================

Test the model locally with Grad-CAM before deploying to Hugging Face.

Usage:
    python demo_with_gradcam.py path/to/xray.png
"""

import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
from gradcam_utils import generate_gradcam_for_top_predictions, get_last_conv_layer_name


def build_model(num_classes=15):
    """Rebuild EfficientNetB0 architecture"""
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB0

    IMG_SIZE = 224
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model = EfficientNetB0(
        include_top=False, weights=None, input_tensor=inputs, pooling="avg"
    )
    x = base_model.output
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", dtype="float32")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def predict_with_gradcam(image_path):
    """Run prediction with Grad-CAM visualization"""

    print("\n" + "=" * 70)
    print("🏥 CHEST X-RAY DISEASE CLASSIFICATION WITH GRAD-CAM")
    print("=" * 70 + "\n")

    # Load model
    print("[1/5] Loading model...")
    model = build_model(num_classes=15)
    model.load_weights("best_model_final.h5")
    print("✅ Model loaded")

    # Load thresholds and label encoder
    print("[2/5] Loading thresholds...")
    with open("optimal_thresholds.pkl", "rb") as f:
        optimal_thresholds = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("✅ Thresholds loaded")

    # Load and preprocess image
    print(f"[3/5] Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    print("✅ Image preprocessed")

    # Run prediction
    print("[4/5] Running inference...")
    probs = model.predict(img_array, verbose=0)[0]

    # Get results above threshold
    results = []
    for disease, idx in label_encoder.items():
        prob = float(probs[idx])
        threshold = optimal_thresholds[disease]
        if prob >= threshold:
            results.append(
                {"disease": disease, "probability": prob, "threshold": threshold}
            )

    results = sorted(results, key=lambda x: x["probability"], reverse=True)
    print("✅ Inference complete")

    # Generate Grad-CAM
    if results:
        print(
            f"[5/5] Generating Grad-CAM for top {min(3, len(results))} predictions..."
        )
        try:
            last_conv_layer = get_last_conv_layer_name(model)
            gradcam_images = generate_gradcam_for_top_predictions(
                image,
                model,
                results,
                label_encoder,
                top_k=min(3, len(results)),
                last_conv_layer_name=last_conv_layer,
            )

            # Save Grad-CAM images
            for i, (disease, grad_img, prob) in enumerate(gradcam_images, 1):
                output_name = f"gradcam_{disease.replace(' ', '_').lower()}.png"
                grad_img.save(output_name)
                print(f"   ✅ Saved: {output_name}")

            print("✅ Grad-CAM generation complete")
        except Exception as e:
            print(f"❌ Grad-CAM failed: {e}")
    else:
        print("[5/5] No abnormalities detected - skipping Grad-CAM")

    # Print results
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70 + "\n")

    if not results:
        print("✅ NO ABNORMALITIES DETECTED")
        print("   All disease probabilities are below optimized thresholds.\n")
    else:
        print(f"⚠️  {len(results)} POTENTIAL FINDING(S) DETECTED:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['disease']}")
            print(f"   Probability: {r['probability']*100:.1f}%")
            print(f"   Threshold:   {r['threshold']*100:.1f}%")
            print(f"   Confidence:  {'HIGH' if r['probability'] > 0.7 else 'MEDIUM'}")
            print()

    print("=" * 70)
    print("⚠️  DISCLAIMER: Research prototype - NOT for clinical diagnosis")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\n❌ Usage: python demo_with_gradcam.py <path_to_xray_image>\n")
        print("Example:")
        print("  python demo_with_gradcam.py images/00000001_000.png\n")
        sys.exit(1)

    predict_with_gradcam(sys.argv[1])
