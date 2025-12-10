import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os

# ============================================================================
# 1. MODEL VE AYARLAR
# ============================================================================

def build_model(num_classes=15):
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

CLASS_NAMES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'No Finding',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax'
]

print("Model hazırlanıyor...")
model = build_model(num_classes=len(CLASS_NAMES))
model_filename = "best_model_final.h5"

try:
    model.load_weights(model_filename)
    print("✅ Ağırlıklar yüklendi.")
except Exception as e:
    print(f"❌ HATA: {e}")

LAST_CONV_LAYER = "top_activation"

# ============================================================================
# 2. GRAD-CAM MOTORU
# ============================================================================

def get_img_array(image_pil, size):
    img_array = np.array(image_pil.resize(size))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except:
        last_conv_layer_name = "block7a_project_conv" # Alternatif katman
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_gradcam_plot(image_pil, heatmap, title):
    img = np.array(image_pil.resize((224, 224)))
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(superimposed_img)
    ax.set_title(title, fontsize=11, fontweight='bold', color='white', backgroundcolor='darkred')
    ax.axis('off')
    return fig

# ============================================================================
# 3. TAHMİN (SANSÜRSÜZ MOD)
# ============================================================================

def predict_xray(image):
    if image is None:
        return None, None, None, None, None

    # Görüntü İşleme
    image_pil = Image.fromarray(image).convert("RGB")
    img_array = get_img_array(image_pil, size=(224, 224))
    
    # Tahmin
    predictions = model.predict(img_array)[0]
    
    # 1. Bar Grafiği İçin Sözlük
    confidences = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    
    # 2. Detaylı Tablo İçin Dataframe (Tüm 0.0001'leri gösterir)
    df = pd.DataFrame(list(confidences.items()), columns=["Hastalık", "Olasılık"])
    df["Olasılık"] = df["Olasılık"].apply(lambda x: f"%{x*100:.4f}") # 4 hane detay
    df = df.sort_values(by="Olasılık", ascending=False, key=lambda x: x.str.strip('%').astype(float))
    
    # 3. Grad-CAM İçin Seçim (Filtresiz)
    # "No Finding" hariç en yüksek 3 skoru al, oranına bakmaksızın!
    active_diseases = []
    for i, score in enumerate(predictions):
        disease_name = CLASS_NAMES[i]
        if disease_name != "No Finding": # Sadece temiz'i çıkar
            active_diseases.append((disease_name, score, i))
            
    # Sırala ve ilk 3'ü al
    active_diseases.sort(key=lambda x: x[1], reverse=True)
    top_3 = active_diseases[:3]
    
    plots = [None, None, None]
    
    try:
        for idx, (name, score, class_idx) in enumerate(top_3):
            # Heatmap oluştur
            heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER, pred_index=class_idx)
            # Başlık (Çok düşükse bile göster)
            plot_title = f"{name}: %{score*100:.2f}"
            plots[idx] = generate_gradcam_plot(image_pil, heatmap, plot_title)
    except Exception as e:
        print(f"Grad-CAM hatası: {e}")

    return confidences, df, plots[0], plots[1], plots[2]

# ============================================================================
# 4. ARAYÜZ (TABLOLU)
# ============================================================================

examples_list = []
if os.path.exists("example_1.jpg"): examples_list.append(["example_1.jpg"])

with gr.Blocks(theme=gr.themes.Soft(), title="Medical AI Analysis") as demo:
    
    gr.Markdown("# 🩻 Detaylı Göğüs Röntgeni Analizi")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Röntgen Yükle", type="numpy")
            predict_btn = gr.Button("🔍 DETAYLI ANALİZ", variant="primary")
            
            # Hepsini gösteren tablo
            output_table = gr.Dataframe(label="Tüm Sonuçlar (En Düşükler Dahil)", headers=["Hastalık", "Olasılık"], interactive=False)

        with gr.Column(scale=1):
            # Görsel Bar Grafik
            output_labels = gr.Label(num_top_classes=15, label="Özet Durum")
            
            gr.Markdown("### 🧠 Şüphelenilen Bölgeler (Grad-CAM)")
            gr.Markdown("_Model 'Temiz' dese bile, en ufak şüphe duyduğu bölgeler:_")
            with gr.Row():
                plot1 = gr.Plot(label="Şüphe 1")
                plot2 = gr.Plot(label="Şüphe 2")
                plot3 = gr.Plot(label="Şüphe 3")

    if examples_list:
        gr.Examples(examples=examples_list, inputs=input_image)

    predict_btn.click(
        fn=predict_xray, 
        inputs=input_image, 
        outputs=[output_labels, output_table, plot1, plot2, plot3]
    )

if __name__ == "__main__":
    demo.launch()