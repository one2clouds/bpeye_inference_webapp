import streamlit as st
from PIL import Image


def some_backgrounds():
    st.title("Background")
    st.write("Effects of Glaucoma")
    image = Image.open("./images/effects_of_glaucoma.png")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("For Glaucoma, focusing primaily on the Optic Cup Disc areas")
    image = Image.open("./images/Normal_vs_Zoomed.png")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Large Cup")
    st.write("Optic Cup enlarges relative to the optic disc")    
    image = Image.open("./images/Large_Cup.png")
    st.image(image, caption="Large Cup", use_container_width=True)

    st.write("Appearance neuro-retinal rim superiorly & inferiorly")
    st.write("Neuro-retinal Rim : Pinkish Tissue surrounding optic cup.")    
    image = Image.open("./images/Neuroretinal_Rim.png")
    st.image(image, caption="Appearance neuro-retinal rim superiorly & inferiorly", use_container_width=True)

    st.write("Nasalization (Nasal) Displacement of vessel trunk")
    st.write("Nasal = Central retinal artery/vein trunk shifts toward the nasal side of the optic disc")    
    st.write("As optic cup gets larger, blood vessels pushed to nasal side")    
    image = Image.open("./images/effects_of_glaucoma.png")
    st.image(image, caption="Nasalization (Nasal) Displacement of vessel trunk", use_container_width=True)

    st.write("Baring of the circum-linear vessel superiorly and inferiorly")
    st.write("Circumlinear vessels : tiny blood vessels around optic disc")
    st.write("Baring : vessel appear displaced along optic cup")
    image = Image.open("./images/effects_of_glaucoma.png")
    st.image(image, caption="Baring of the circum-linear vessel superiorly and inferiorly", use_container_width=True)

    st.write("Disc Hemorrhage(s)")
    st.write("A small bleed on the optic disc surface because of pressure inside eye (Intraocular Pressure")
    st.write("Diagnostic criteria for glaucoma may not be met at the time of disc hemorrhage observation")
    st.write("Some results show many eyes may eventually progress to glaucoma after disc hemorrhage")
    image = Image.open("./images/effects_of_glaucoma.png")
    st.image(image, caption="Disc Hemorrhage(s)", use_container_width=True)

    st.write("Retinal Nerve Fiber Layer (RNFL) defect superiorly and inferiorly")
    st.write("Dark, wedge-shaped areas radiating from the optic disc.")
    st.write("Thinning of RNFL in the upper (superiorly) andlower (inferiorly) retinal regions")
    image = Image.open("./images/effects_of_glaucoma.png")
    st.image(image, caption="Retinal Nerve Fiber Layer (RNFL) defect superiorly and inferiorly", use_container_width=True)

    st.write("Laminar Dots")
    st.write("The lamina cribrosa (mesh-like structure of tissues in eye) is usually not vividly visible in healthy eyes.")
    st.write("Pressure inside eye (Intraocular Pressure) causes mechanical stress on the lamina cribrosa")
    st.write("It then, stretches lamina cribrosa, thinning surrounding nerve tissues, making it visible as laminar dots")
    image = Image.open("./images/effects_of_glaucoma.png")
    st.image(image, caption="Fundus image optic disk with multiple lamina cribrosa pores (left) Blue dots highlighting the location of the pores of left image (right)", use_column_width=True)
