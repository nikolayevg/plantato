import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import json
import html
import supervision as sv
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Конфигурация
MODEL_ID = '/home/nick/deeplearning/DIPLOMA/project/plantato/checkpoints/var_2'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    return processor, model

processor, model = load_model()

def plot_bbox(image, data):
   # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Remove the axis ticks and labels
    ax.axis('off')

    # Show the plot
    return plt

def render_results(image: Image.Image, response):
    """Визуализация результатов с bounding boxes"""
    try:
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
        image = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image.copy(), detections)
        image = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image, detections)
        return image
    except Exception as e:
        st.error(f"Ошибка визуализации: {e}")
        return image
    
# Интерфейс
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
st.title("🌱 Анализ заболеваний листьев")
uploaded_file = st.file_uploader("Загрузите изображение листа", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Исходное изображение", use_column_width=True)

    if st.button("Анализировать"):
        with st.spinner("Обработка..."):
            # Ключевое изменение: передаем ТОЛЬКО <OD> без дополнительного текста
            inputs = processor(text="<OD>", images=image, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
            
            generated_text = processor.batch_decode(generated_ids, skip_special_ytokens=False)[0]
            result = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)
        
        st.subheader("Результат:")
        print(result)
        if isinstance(result, dict) and len(result['<OD>']['bboxes']) != 0:
            annotated_img = render_results(image, result)
            st.image(annotated_img, caption="Обнаруженные заболевания", use_column_width=True)
            st.json(result['<OD>']['labels'])  # Показываем raw-ответ модели
        else:
            st.warning("Модель не обнаружила заболеваний.")