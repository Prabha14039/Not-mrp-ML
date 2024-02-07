import streamlit as st
import os
from PIL import Image
#from model import predict_and_display, save_uploaded_file

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())  
        return 1
    except:
        return 0
    
# File upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # Predict and display
        image, purchase_link = predict_and_display(os.path.join("uploads", uploaded_file.name))
        
        # Display the predicted image
        st.image(image)

        # Display the purchase link
        st.write("Purchase Link:", purchase_link)
    else:
        st.error("Some error occurred during file upload. Please try again.")
