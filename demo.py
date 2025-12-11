"""
Chest X-Ray Disease Classification - Inference Demo
====================================================

Simple demo for predicting diseases from chest X-ray images.

DISCLAIMER: This is a research prototype. NOT for clinical use.
Always consult a radiologist for medical diagnosis.
"""

import tensorflow as tf
import numpy as np
import pickle
import os
from pathlib import Path


class ChestXRayPredictor:
    """
    Chest X-ray disease predictor using trained EfficientNetB0 model.

    Usage:
        predictor = ChestXRayPredictor('best_model_final.h5',
                                       'optimal_thresholds.pkl',
                                       'label_encoder.pkl')
        results = predictor.predict('sample.png', use_tta=False)
    """

    def __init__(self, weights_path, thresholds_path, encoder_path):
        """
        Initialize predictor.

        Args:
            weights_path: Path to model weights (.h5 file)
            thresholds_path: Path to optimal thresholds (.pkl file)
            encoder_path: Path to label encoder (.pkl file)
        """
        # Load label encoder
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        # Load optimal thresholds
        with open(thresholds_path, "rb") as f:
            self.optimal_thresholds = pickle.load(f)

        # Build and load model
        self.model = self._build_model()
        self.model.load_weights(weights_path)

        print(f"✅ Model loaded successfully!")
        print(f"   Diseases: {len(self.label_encoder)}")
        print(f"   Weights: {weights_path}")

    def _build_model(self):
        """Rebuild model architecture (must match training)"""
        from tensorflow.keras import layers
        from tensorflow.keras.applications import EfficientNetB0

        IMG_SIZE = 224
        NUM_CLASSES = len(self.label_encoder)

        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        base_model = EfficientNetB0(
            include_top=False,
            weights=None,  # We'll load weights manually
            input_tensor=inputs,
            pooling="avg",
        )

        x = base_model.output
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="sigmoid", dtype="float32")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def preprocess_image(self, image_path):
        """Load and preprocess X-ray image"""
        img = tf.io.read_file(image_path)

        # Support PNG/JPEG
        if image_path.lower().endswith(".png"):
            img = tf.image.decode_png(img, channels=3)
        else:
            img = tf.image.decode_jpeg(img, channels=3)

        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, 0)  # Add batch dimension
        return img

    def predict(self, image_path, use_tta=False, verbose=True):
        """
        Predict diseases from chest X-ray.

        Args:
            image_path: Path to X-ray image (PNG or JPEG)
            use_tta: Use Test-Time Augmentation (slower but more accurate)
            verbose: Print results

        Returns:
            List of dict with disease predictions (sorted by probability)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Preprocess
        img = self.preprocess_image(image_path)

        # Predict
        if use_tta:
            # Test-Time Augmentation (6 predictions)
            predictions = []

            # Original
            predictions.append(self.model.predict(img, verbose=0)[0])

            # 5 augmented versions
            for _ in range(5):
                aug_img = tf.image.random_flip_left_right(img)
                aug_img = tf.image.random_brightness(aug_img, max_delta=0.1)
                aug_img = tf.clip_by_value(aug_img, 0.0, 1.0)
                predictions.append(self.model.predict(aug_img, verbose=0)[0])

            probs = np.mean(predictions, axis=0)
            std = np.std(predictions, axis=0)
        else:
            probs = self.model.predict(img, verbose=0)[0]
            std = None

        # Apply thresholds
        results = []
        for disease, idx in self.label_encoder.items():
            prob = float(probs[idx])
            threshold = self.optimal_thresholds[disease]
            prediction = "POSITIVE" if prob >= threshold else "NEGATIVE"

            # Only include positive predictions
            if prob >= threshold:
                confidence_score = min((prob - threshold) / (1 - threshold), 1.0)
                confidence = "HIGH" if confidence_score > 0.5 else "MEDIUM"

                results.append(
                    {
                        "disease": disease,
                        "probability": prob,
                        "threshold": threshold,
                        "prediction": prediction,
                        "confidence": confidence,
                        "std": float(std[idx]) if std is not None else None,
                    }
                )

        # Sort by probability
        results = sorted(results, key=lambda x: x["probability"], reverse=True)

        # Print results
        if verbose:
            self._print_results(image_path, results, use_tta)

        return results

    def _print_results(self, image_path, results, use_tta):
        """Pretty print results"""
        print("\n" + "=" * 80)
        print("CHEST X-RAY ANALYSIS RESULTS")
        print("=" * 80)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Method: {'TTA (6 predictions)' if use_tta else 'Single prediction'}")
        print("-" * 80)

        if not results:
            print("✅ NO ABNORMALITIES DETECTED")
            print("   (All disease probabilities below threshold)")
        else:
            print(f"⚠️  {len(results)} POTENTIAL FINDING(S) DETECTED")
            print("-" * 80)
            print(
                f"{'Disease':<25} {'Probability':<12} {'Confidence':<12} {'Threshold':<10}"
            )
            print("-" * 80)

            for r in results:
                prob_str = f"{r['probability']:.1%}"
                print(
                    f"{r['disease']:<25} {prob_str:<12} {r['confidence']:<12} {r['threshold']:.3f}"
                )

        print("-" * 80)
        print("⚠️  DISCLAIMER: This is a research prototype.")
        print("   NOT for clinical diagnosis. Consult a radiologist.")
        print("=" * 80 + "\n")


def demo():
    """Run demo with sample image"""
    print("Chest X-Ray Disease Classification - Demo")
    print("=" * 80)

    # Check files
    required_files = [
        "best_model_final.h5",
        "optimal_thresholds.pkl",
        "label_encoder.pkl",
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing files: {missing}")
        print("\nTo generate these files:")
        print("1. Run the training notebook: chest_xray_analysis.ipynb")
        print("2. Execute the final cell to save model weights")
        return

    # Initialize predictor
    predictor = ChestXRayPredictor(
        weights_path="best_model_final.h5",
        thresholds_path="optimal_thresholds.pkl",
        encoder_path="label_encoder.pkl",
    )

    # Demo image
    demo_image = "images/00000013_005.png"  # Replace with your test image

    if not os.path.exists(demo_image):
        print(f"\n❌ Demo image not found: {demo_image}")
        print("Please specify a valid chest X-ray image path.")
        return

    # Predict (without TTA for speed)
    print("\n🔍 Running inference (fast mode)...")
    results = predictor.predict(demo_image, use_tta=False)

    # Predict with TTA (more accurate but slower)
    print("\n🔍 Running inference with TTA (accurate mode)...")
    results_tta = predictor.predict(demo_image, use_tta=True)


if __name__ == "__main__":
    demo()
