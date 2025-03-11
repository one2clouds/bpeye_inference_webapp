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
    checkpoint = torch.load('/home/shirshak/inference_BPEye_Project_2024/glaucoma_resnet_airogs_focal/epoch_018.ckpt', weights_only=False)
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

    st.title("Background")
    st.write("Effects of Glaucoma")
    st.write("Damaged caused by Glaucoma can't be reversed. If detected early, vision loss can be prevented.")    
    image = Image.open("./images/effects_of_glaucoma.png")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("For Glaucoma, focusing primaily on the Optic Cup Disc areas")
    image = Image.open("./images/Normal_vs_Zoomed.png")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Large Cup")
    st.write("Optic Cup enlarges relative to the optic disc")    
    image = Image.open("./images/Large_Cup.png")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Appearance neuro-retinal rim superiorly & inferiorly")
    st.write("Neuro-retinal Rim : Pinkish Tissue surrounding optic cup.")    
    image = Image.open("./images/Neuroretinal_Rim.png")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Nasalization (Nasal) Displacement of vessel trunk")
    st.write("Nasal = Central retinal artery/vein trunk shifts toward the nasal side of the optic disc")    
    st.write("As optic cup gets larger, blood vessels pushed to nasal side")    
    image = Image.open("./images/effects_of_glaucoma.png")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Baring of the circum-linear vessel superiorly and inferiorly")
    st.write("Circumlinear vessels : tiny blood vessels around optic disc")
    st.write("Baring : vessel appear displaced along optic cup")
    image = Image.open("./images/effects_of_glaucoma.png")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Disc Hemorrhage(s)")
    st.write("A small bleed on the optic disc surface because of pressure inside eye (Intraocular Pressure")
    st.write("Diagnostic criteria for glaucoma may not be met at the time of disc hemorrhage observation")
    st.write("Some results show many eyes may eventually progress to glaucoma after disc hemorrhage")
    image = Image.open("./images/effects_of_glaucoma.png")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    print("Hello, model loaded")



def AMD_Classification():
    st.title("AMD Classification")
    html_temp = """
    <div style="background-color:#40B3DF;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Glaucoma CLassification App </h2>
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
    menu = ["Glaucoma CLassification", "AMD Classification", "Optic Cup Disc Segmentation"]
    choice = st.sidebar.selectbox(label='Menu', options=menu)

    if choice == "Glaucoma CLassification":
        Glaucoma_Classification()
    if choice == "AMD Classification":
        AMD_Classification()
    if choice == "Optic Cup Disc Segmentation":
        Optic_Disc_Cup_Segmentation()







    







