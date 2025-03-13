import torch.nn as nn 
import torch 
import torchvision
from torchvision.transforms import transforms
from glob import glob 
from torchvision.transforms import Lambda, Compose, ToPILImage
from preprocessing.crop_transform_pad_images import read_image, crop_nonzero, pad_to_largest_square
import streamlit as st 
from PIL import Image 
from torchvision.models import ResNet50_Weights
from src.models.res_net_module import Res_Net
from optic_cup_disk_segmentation import Optic_Disc_Cup_Segmentation
import glob
from some_backgrounds_glaucomic_features import some_backgrounds

def predict_disease(img_path, img_transform, my_transforms, model, device): 

    preprocessed_img = img_transform(img_path)

    preprocessed_transformed_image = my_transforms(preprocessed_img)

    preprocessed_transformed_image = preprocessed_transformed_image.unsqueeze(0).to(device)
    # print(preprocessed_transformed_image.shape)

    outputs = nn.Sigmoid()(model(preprocessed_transformed_image))
    confidence_score, predicted_test = torch.max(outputs, 1)

    predicted_label = "Referable Glaucoma" if predicted_test.item() == 1 else "Non-Referable Glaucoma"

    return predicted_label, confidence_score.item()*100



def Glaucoma_Classification():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ['NRG', 'RG']


    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    # model = Res_Net(len(classes))
    torch.serialization.add_safe_globals([Res_Net])
    checkpoint = torch.load('./glaucoma_resnet_airogs_focal/epoch_018.ckpt', weights_only=False)
    state_dict = {k.replace('net.model.', ''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)

    my_transforms = transforms.Compose([
        # transforms.ToPILImage(), 
        transforms.ToTensor(), # convert 0-255 to 0-1 and from np to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Resize(size=(512,512),antialias=True)
    ])

    img_transform = Compose([
    # Lambda(read_image), # because we have already converted our image to tensor
    Lambda(crop_nonzero),
    Lambda(pad_to_largest_square),
    ToPILImage(),
    ])


    st.title("Glaucoma Classification")
    html_temp = """
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
        max_width, max_height = 300, 300
        image_tensor = transforms.PILToTensor()(image)

        image.thumbnail(size = (max_width, max_height))
        st.image(image, caption="", use_container_width=False)

    result=""
    if st.button("Predict", key="predict_button_1"):
        if uploaded_file is None:
            st.warning("Please Upload an image from above to get predictions.")
        else:
            predicted_label, confidence_percentage = predict_disease(image_tensor, img_transform, my_transforms, model, device)
            
            # Define styles for different cases
            if predicted_label == "Referable Glaucoma":
                st.markdown(
                    f"""
                    <div style="background-color: #ffcccc; color: #666666;
                        padding: 15px; border-radius: 10px; font-size: 18px; font-weight: bold; text-align: center; border: 2px solid red; "> <b>{predicted_label} and Confidence Score : {confidence_percentage:.2f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif predicted_label == "Non-Referable Glaucoma":
                st.markdown(
                    f"""
                    <div style=" background-color: #ccffcc; color: #666666; padding: 15px; border-radius: 10px; font-size: 18px; font-weight: bold; text-align: center; border: 2px solid green; "> <b>{predicted_label} and Confidence Score : {confidence_percentage:.2f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    some_backgrounds()

    print("Hello, model loaded")



def AMD_Classification():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ['AMD', 'Non-AMD']


    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    # model = Res_Net(len(classes))
    torch.serialization.add_safe_globals([Res_Net])
    checkpoint = torch.load('./amd_resnet_airogs/epoch_018.ckpt', weights_only=False)
    state_dict = {k.replace('net.model.', ''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)

    my_transforms = transforms.Compose([
        # transforms.ToPILImage(), 
        transforms.ToTensor(), # convert 0-255 to 0-1 and from np to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Resize(size=(512,512),antialias=True)
    ])

    img_transform = Compose([
    # Lambda(read_image), # because we have already converted our image to tensor
    Lambda(crop_nonzero),
    Lambda(pad_to_largest_square),
    ToPILImage(),
    ])

    st.title("AMD Classification")
    html_temp = """
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
        max_width, max_height = 300, 300
        image_tensor = transforms.PILToTensor()(image)

        image.thumbnail(size = (max_width, max_height))
        st.image(image, caption="", use_container_width=False)

    
    result=""
    if st.button("Predict", key="predict_button_2"):
        if uploaded_file is None:
            st.warning("Please Upload an image from above to get predictions.")
        else:
            predicted_label, confidence_percentage = predict_amd_disease(image_tensor, img_transform, my_transforms, model, device)
            
            # Define styles for different cases
            if predicted_label == "AMD":
                st.markdown(
                    f"""
                    <div style="background-color: #ffcccc; color: #666666;
                        padding: 15px; border-radius: 10px; font-size: 18px; font-weight: bold; text-align: center; border: 2px solid red; "> <b>{predicted_label} and Confidence Score : {confidence_percentage:.2f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif predicted_label == "Non-AMD":
                st.markdown(
                    f"""
                    <div style=" background-color: #ccffcc; color: #666666; padding: 15px; border-radius: 10px; font-size: 18px; font-weight: bold; text-align: center; border: 2px solid green; "> <b>{predicted_label} and Confidence Score : {confidence_percentage:.2f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    st.set_page_config(page_title="NAAMII - BPEye WebApp")

    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)



    st.sidebar.header("Select the Task")
    menu = ["Glaucoma Classification", "AMD Classification", "Optic Cup Disc Segmentation"]
    choice = st.sidebar.selectbox(label='Menu', options=menu)

    if choice == "Glaucoma Classification":
        Glaucoma_Classification()
    if choice == "AMD Classification":
        AMD_Classification()
    if choice == "Optic Cup Disc Segmentation":
        Optic_Disc_Cup_Segmentation()







    







