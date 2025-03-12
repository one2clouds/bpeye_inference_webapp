import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import streamlit as st 
from PIL import Image
from torchvision.transforms import transforms
from streamlit_image_comparison import image_comparison
import io


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

def fig2img(fig): 
    buf = io.BytesIO() 
    fig.savefig(buf) 
    buf.seek(0) 
    img = Image.open(buf) 
    return img 


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

        fig, axes = plt.subplots(figsize=(8, 6))
        axes.imshow(image)
        axes.axis('off')
        fig.tight_layout(pad=0)
        # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # axes.margins(x=0, y=0)
        image = fig2img(fig) 

        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes.imshow(pred_disc_cup, cmap='gray')
        # st.pyplot(fig, use_container_width=False)
        axes.axis('off')
        fig.tight_layout(pad=0)
        # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # axes.margins(x=0, y=0)
        segmentation = fig2img(fig)

        # st.image(image)
        # st.image(segmentation)

        image_comparison(
            img1=image,
            img2=segmentation,
        )