import streamlit as st
from PIL import Image


def some_backgrounds():
    for _ in range(30):
        st.write("")

    max_width, max_height = 400, 300

    st.title("Understanding the Effects of Glaucoma")
    st.write("---")

    st.header("1. Overview")
    image = Image.open("./images/effects_of_glaucoma.png")
    image.thumbnail(size = (max_width, max_height))
    st.image(image, caption="Effects of Glaucoma", use_container_width=False)
    st.write("---")

    st.header("2. Normal vs Glaucoma")
    st.markdown("""
                - Glaucoma affects the **Optic Cup Disc** areas, so focus primarily on those areas
                - Notice the differences in the **Optic Cup Disc** areas between normal and glaucomatous eyes.
                """)
    image = Image.open("./images/Normal_vs_Zoomed.png")
    image.thumbnail(size = (max_width, max_height))
    st.image(image, caption="Normal vs Zoomed Optic Cup Disc", use_container_width=False)


    st.header("3. Key Effects")

    # --------------------------------------------------------

    st.subheader("Large Cup")
    st.markdown("""
            - Optic Cup enlarges relative to the optic disc
            """)
    image = Image.open("./images/Large_Cup.png")
    image.thumbnail(size = (max_width, max_height))
    st.image(image, caption="Large Cup", use_container_width=False)
    st.write("---")

    # --------------------------------------------------------

    st.subheader("Appearance neuro-retinal rim superiorly & inferiorly")
    st.markdown("""
                - Neuro-retinal Rim : Pinkish Tissue surrounding optic cup
                """)
    image = Image.open("./images/Neuroretinal_Rim.png")
    image.thumbnail(size = (max_width, max_height))
    st.image(image, caption="Appearance neuro-retinal rim superiorly & inferiorly", use_container_width=False)
    st.write("---")

    # --------------------------------------------------------

    st.subheader("Nasalization (Nasal) Displacement of vessel trunk")
    st.markdown("""
                - Nasal : Central retinal artery/vein trunk shifts toward the nasal side of the optic disc
                - As optic cup gets larger, blood vessels pushed to nasal side
                """)
    image = Image.open("./images/Nasalization.png")
    image.thumbnail(size = (max_width, max_height))
    st.image(image, caption="Nasalization (Nasal) Displacement of vessel trunk", use_container_width=False)
    st.write("---")

    # --------------------------------------------------------

    st.subheader("Baring of the circum-linear vessel superiorly and inferiorly")
    st.markdown("""
                - Circumlinear vessels : tiny blood vessels around optic disc
                - Baring : vessel appear displaced along optic cup
                """)
    image = Image.open("./images/Baring_Circum_Linear_Vessels.png")
    image.thumbnail(size = (max_width, max_height))
    st.image(image, caption="Baring of the circum-linear vessel superiorly and inferiorly", use_container_width=False)
    st.write("---")

    # --------------------------------------------------------

    st.subheader("Disc Hemorrhage(s)")
    st.markdown("""
                - A small bleed on the optic disc surface because of pressure inside eye (Intraocular Pressure)
                - Diagnostic criteria for glaucoma may not be met at the time of disc hemorrhage observation
                - Some results show many eyes may eventually progress to glaucoma after disc hemorrhage.
                """)
    image = Image.open("./images/Disc_Hemmorrhage.png")
    image.thumbnail(size = (max_width, max_height))
    st.image(image, caption="Disc Hemorrhage(s)", use_container_width=False)
    st.write("---")

    # --------------------------------------------------------

    st.subheader("Retinal Nerve Fiber Layer (RNFL) defect superiorly and inferiorly")
    st.markdown("""
                - Dark, wedge-shaped areas radiating from the optic disc
                - Thinning of RNFL in the upper (superiorly) andlower (inferiorly) retinal regions
                """)
    image = Image.open("./images/RNFL_defect.png")
    image.thumbnail(size = (max_width, max_height))
    st.image(image, caption="Retinal Nerve Fiber Layer (RNFL) defect superiorly and inferiorly", use_container_width=False)
    st.write("---")

    # --------------------------------------------------------

    st.subheader("Laminar Dots")
    st.markdown("""
                - The lamina cribrosa (mesh-like structure of tissues in eye) is usually not vividly visible in healthy eyes
                - Pressure inside eye (Intraocular Pressure) causes mechanical stress on the lamina cribrosa
                - It then, stretches lamina cribrosa, thinning surrounding nerve tissues, making it visible as laminar dots
                """)
    image = Image.open("./images/Laminar_Dots.png")
    image.thumbnail(size = (max_width, max_height))
    st.image(image, caption="Fundus image optic disk with multiple lamina cribrosa pores (left) Blue dots highlighting the location of the pores of left image (right)", use_container_width=False)
    st.write("---")

    # --------------------------------------------------------
