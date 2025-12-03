"""
Grad-CAM Implementation for Chest X-Ray Classification
========================================================

Visualizes which regions of the X-ray the model focuses on when making predictions.

Reference: Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep Networks
"""

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a given image and prediction.
    
    Args:
        img_array: Preprocessed image (batch_size, height, width, channels)
        model: Trained Keras model
        last_conv_layer_name: Name of last convolutional layer
        pred_index: Target class index (if None, uses predicted class)
    
    Returns:
        heatmap: Normalized heatmap (0-1 range)
    """
    # Create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient of the output neuron with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 & 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        img: Original PIL Image or numpy array
        heatmap: Grad-CAM heatmap (0-1 range)
        alpha: Transparency of heatmap overlay (0-1)
        colormap: OpenCV colormap (default: JET - red=hot, blue=cold)
    
    Returns:
        superimposed_img: PIL Image with heatmap overlay
    """
    # Convert PIL to numpy if needed
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    
    return Image.fromarray(superimposed_img)


def generate_gradcam_for_disease(image, model, disease_name, label_encoder, 
                                  last_conv_layer_name='top_conv', img_size=224):
    """
    Generate Grad-CAM visualization for a specific disease prediction.
    
    Args:
        image: PIL Image
        model: Trained model
        disease_name: Name of disease to visualize
        label_encoder: Disease name -> index mapping
        last_conv_layer_name: Name of last conv layer in EfficientNetB0
        img_size: Input image size
    
    Returns:
        overlaid_image: PIL Image with Grad-CAM overlay
        heatmap: Raw heatmap array
    """
    # Preprocess image
    img_resized = image.convert('RGB').resize((img_size, img_size))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    # Get disease index
    disease_idx = label_encoder[disease_name]
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, disease_idx)
    
    # Overlay on original image
    overlaid_image = overlay_heatmap_on_image(img_resized, heatmap, alpha=0.4)
    
    return overlaid_image, heatmap


def generate_gradcam_for_top_predictions(image, model, predictions, label_encoder, 
                                          top_k=3, last_conv_layer_name='top_conv'):
    """
    Generate Grad-CAM for top K predicted diseases.
    
    Args:
        image: PIL Image
        model: Trained model
        predictions: List of prediction dicts from main app
        label_encoder: Disease name -> index mapping
        top_k: Number of top predictions to visualize
        last_conv_layer_name: Name of last conv layer
    
    Returns:
        gradcam_images: List of (disease_name, overlaid_image, probability) tuples
    """
    gradcam_images = []
    
    # Sort predictions by probability
    sorted_preds = sorted(predictions, key=lambda x: x['probability'], reverse=True)[:top_k]
    
    for pred in sorted_preds:
        disease_name = pred['disease']
        probability = pred['probability']
        
        # Generate Grad-CAM
        overlaid_img, _ = generate_gradcam_for_disease(
            image, model, disease_name, label_encoder, last_conv_layer_name
        )
        
        gradcam_images.append((disease_name, overlaid_img, probability))
    
    return gradcam_images


def get_last_conv_layer_name(model):
    """
    Automatically find the last convolutional layer in the model.
    
    For EfficientNetB0, it's typically 'top_conv' or the last Conv2D layer.
    
    Args:
        model: Keras model
    
    Returns:
        layer_name: Name of last conv layer
    """
    # Try common names first
    common_names = ['top_conv', 'block7a_project_conv', 'conv_head']
    for name in common_names:
        try:
            model.get_layer(name)
            return name
        except:
            pass
    
    # Search backwards for Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    
    raise ValueError("No convolutional layer found in model!")
