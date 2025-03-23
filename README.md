# Mask2Former Semantic Segmentation

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue)](https://huggingface.co/spaces/Tahir5/mask2former-streamlit-app)

## Overview
Mask2Former Semantic Segmentation is a Streamlit-based web application that utilizes the Mask2Former model fine-tuned on the ADE20k dataset to perform semantic segmentation on images. Users can upload their own images or use a sample image to visualize segmentation results.

## Features
- **Upload an Image**: Users can upload images in JPG, PNG, or JPEG format.
- **Perform Segmentation**: Uses Facebook's Mask2Former model for high-quality semantic segmentation.
- **Visualize Results**: Displays both the original image and the segmented image with color mapping.
- **Sample Image Support**: Users can test the model with a predefined sample image.

## Technologies Used
- **Python**
- **Streamlit**
- **Hugging Face Transformers**
- **PyTorch**
- **PIL (Pillow)**
- **Matplotlib**
- **NumPy**
- **Requests**

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Tahir-Abbas-555/mask2former-streamlit-app.git
   cd mask2former-streamlit-app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the app in your browser (default: `http://localhost:8501`).
2. Upload an image or use the sample image.
3. Click "Segment Image" to process and visualize the results.

## Live Demo
Try the app online without installation: [Mask2Former Streamlit App](https://huggingface.co/spaces/Tahir5/mask2former-streamlit-app)

## Screenshots
| Original Image | Segmented Image |
|---------------|----------------|
| ![Original](https://via.placeholder.com/400) | ![Segmented](https://via.placeholder.com/400) |

## License
This project is licensed under the MIT License.

## Author
[Tahir Abbas Shaikh](https://github.com/Tahir-Abbas-555)

