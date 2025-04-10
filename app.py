import streamlit as st
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import numpy as np
import matplotlib.pyplot as plt

# Load Mask2Former fine-tuned on ADE20k semantic segmentation
st.set_page_config(
    page_title="Mask2Former Semantic Segmentation",
    layout="centered",
    page_icon="🧠"
    )

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")


@st.cache_resource(show_spinner=False)
def load_model():
    """Load Segformer model and processor with appropriate device."""
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    return processor, model, device


def segment_image(image: Image.Image, processor, model):
    """Run inference and return segmentation labels."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def visualize_segmentation(image: Image.Image, segmentation_map):
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_map, cmap="jet", alpha=0.7)
    plt.axis("off")
    plt.title("Segmented Image")
    
    st.pyplot(plt)


def sidebar_profile():
    # Sidebar info with custom profile section
    st.sidebar.title("ℹ️ About")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <style>
            .custom-sidebar {
                display: flex;
                flex-direction: column;
                align-items: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                width: 650px;
                padding: 10px;
            }
            .profile-container {
                display: flex;
                flex-direction: row;
                align-items: flex-start;
                width: 100%;
            }
            .profile-image {
                width: 200px;
                height: auto;
                border-radius: 15px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
                margin-right: 15px;
            }
            .profile-details {
                font-size: 14px;
                width: 100%;
            }
            .profile-details h3 {
                margin: 0 0 10px;
                font-size: 18px;
                color: #333;
            }
            .profile-details p {
                margin: 10px 0;
                display: flex;
                align-items: center;
            }
            .profile-details a {
                text-decoration: none;
                color: #1a73e8;
            }
            .profile-details a:hover {
                text-decoration: underline;
            }
            .icon-img {
                width: 18px;
                height: 18px;
                margin-right: 6px;
            }
        </style>

        <div class="custom-sidebar">
            <div class="profile-container">
                <img class="profile-image" src="https://res.cloudinary.com/dwhfxqolu/image/upload/v1744014185/pnhnaejyt3udwalrmnhz.jpg" alt="Profile Image">
                <div class="profile-details">
                    <h3>👨‍💻 Developed by:<br> Tahir Abbas Shaikh</h3>
                    <p>
                        <img class="icon-img" src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail">
                        <strong>Email:</strong> <a href="mailto:tahirabbasshaikh555@gmail.com">tahirabbasshaikh555@gmail.com</a>
                    </p>
                    <p>📍 <strong>Location:</strong> Sukkur, Sindh, Pakistan</p>
                    <p>
                        <img class="icon-img" src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" alt="GitHub">
                        <strong>GitHub:</strong> <a href="https://github.com/Tahir-Abbas-555" target="_blank">Tahir-Abbas-555</a>
                    </p>
                    <p>
                        <img class="icon-img" src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace">
                        <strong>HuggingFace:</strong> <a href="https://huggingface.co/Tahir5" target="_blank">Tahir5</a>
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------- MAIN UI ----------------------

def main():
    # Set page config FIRST
    
    st.title("🎯 Mask2Former Semantic Segmentation")

    st.markdown(
        """
        Upload an image, and this app will perform **semantic segmentation** using 
        the [facebook/mask2former-swin-large-ade-semantic](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic) model. 
        This powerful model is trained on the ADE20K dataset and can segment a wide variety of scenes and objects with high accuracy.
        """
    )

    # Load the model only once
    processor, model, device = load_model()

    # File uploader
    uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with st.spinner("🧠 Processing with Segformer..."):
            labels = segment_image(image, processor, model)

        with col2:
            st.markdown("#### 🖼️ Segmentation Output")
            visualize_segmentation(image, labels)

        st.success("✅ Segmentation completed successfully!")

    else:
        st.info("Please upload an image to start face parsing.")

# ---------------------- LAUNCH APP ----------------------

if __name__ == "__main__":
    sidebar_profile()
    main()