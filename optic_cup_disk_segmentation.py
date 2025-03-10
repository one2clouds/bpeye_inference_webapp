import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import streamlit as st 
from PIL import Image
from torchvision.transforms import transforms


def process_image(image, processor, model):
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the input image
    inputs = processor(image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu()

    # Upsample the logits to match the input image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )

    # Get the predicted segmentation
    pred_disc_cup = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)

    return pred_disc_cup


def Optic_Disc_Cup_Segmentation():
    st.title("Optic Cup Disc Segmentation")

    processor = AutoImageProcessor.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
    model = SegformerForSemanticSegmentation.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")

    html_temp = """
    <div style="background-color:#40B3DF;padding:10px">
    <h2 style="color:white;text-align:center;"> Optic Cup Disc Segmentation </h2>
    </div>
    <style>
    .stButton>button {
        background: linear-gradient(135deg, #6a5acd, #4caf50); /* Soft purple to green */
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 18px;
        box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.2);
        border: none;
    }
    </style>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_arr = np.array(image)

        pred_disc_cup = process_image(image_arr, processor, model)

        # Display the input image, predicted segmentation, and original result picture
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        axes[0].imshow(image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        axes[1].imshow(pred_disc_cup, cmap='gray')
        axes[1].set_title('Predicted Segmentation')
        axes[1].axis('off')

        st.pyplot(fig, use_container_width=False)