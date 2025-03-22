import streamlit as st
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import numpy as np
import matplotlib.pyplot as plt

# Load Mask2Former fine-tuned on ADE20k semantic segmentation
st.title("Mask2Former Semantic Segmentation")
st.write("Upload an image to perform semantic segmentation using Mask2Former.")

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")

def segment_image(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def visualize_segmentation(image: Image.Image, segmentation_map):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_map, cmap="jet", alpha=0.7)
    plt.axis("off")
    plt.title("Segmented Image")
    
    st.pyplot(plt)

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Segment Image"):
        st.write("Processing the image...")
        segmentation_map = segment_image(image)
        visualize_segmentation(image, segmentation_map.numpy())

# Option to test with a sample image
if st.button("Use Sample Image"):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    st.image(image, caption="Sample Image", use_column_width=True)
    
    st.write("Processing the image...")
    segmentation_map = segment_image(image)
    visualize_segmentation(image, segmentation_map.numpy())
