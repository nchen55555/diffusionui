import streamlit as st

st.write("# Welcome to a Live Demo of the Experimental Generative AI Models for Tile Generation! 👋")

st.sidebar.success("Select a model to run above.")

st.markdown(
    """
    **👈 Select a model** to experiment with recreating some images!
    ### Steps
    - Input an image url into the first form to generate a prompt
    - Customize the prompt or directly copy and paste it into the second form for image regeneration
    - For more information about the different models, check out this [presentation](https://drive.google.com/file/d/1cGuGW_FV0nmErdlhsaOx9vsiANP-3bf0/view?usp=drive_link)
    ### Overview
    - Model A is Realistic Vision V1.4 CUDA Blip Base (the best overall model)
    - Model B is Realistic Vision V1.4 CUDA Git Large Coco (the best disregarding NSFW scores)
    - Model C is Photoreal 2.0 CUDA Git Large Coco (the best disregarding any quality assesments)
"""
)

st.markdown(":green[Created By Nicole Chen]")
