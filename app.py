import streamlit as st
import os
import torch
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path


# Function for saving images and making predictions
def save_image(uploaded_file):
    if uploaded_file is not None:
        # Create the 'images' directory if it doesn't exist
        os.makedirs("images", exist_ok=True)
        
        save_path = os.path.join("images", "input.jpeg")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Image saved to {save_path}")

        # Load the model
        model = torch.load(Path('artifacts/06_19_2024_22_47_31/model_training/model.pt'))

        # Define the transformations
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # Open the image and apply the transformations
        image = Image.open(Path('images/input.jpeg'))
        input_tensor = trans(image)

        # Adjust the tensor shape if needed
        input_tensor = input_tensor.view(1, 1, 224, 224).repeat(1, 3, 1, 1)

        # Make the prediction
        output = model(input_tensor)
        prediction = int(torch.max(output.data, 1)[1].numpy())

        # Display the prediction
        if prediction == 0:
            st.text_area(label="Prediction:", value="Normal", height=100)
        elif prediction == 1:
            st.text_area(label="Prediction:", value="PNEUMONIA", height=100)


if __name__ == "__main__":
    st.title("Xray lung classifier")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    save_image(uploaded_file)