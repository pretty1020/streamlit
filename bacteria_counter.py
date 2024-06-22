import streamlit as st
import cv2
import numpy as np
from skimage import measure, color
import pandas as pd

# Function to count bacteria
def count_bacteria(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to get a binary image
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

    # Label connected regions of the binary image
    labeled_image = measure.label(binary_image, connectivity=2, background=0)
    num_features = labeled_image.max()

    # Convert labeled image to RGB for better visualization
    labeled_image_rgb = color.label2rgb(labeled_image, bg_label=0)

    return num_features, labeled_image_rgb, labeled_image

# Streamlit App
st.title("Bacteria Counter from Microscope Image")

# Documentation and Help Section
st.sidebar.title("Documentation and Help")
st.sidebar.markdown("""
This application allows you to upload a microscope image and automatically counts the number of bacteria present.

### How to use the application
1. **Upload an Image**: Click on the 'Browse files' button to upload a microscope image (supported formats: jpg, png, jpeg).
2. **View Results**: The application will display the uploaded image, count the number of bacteria, and show a labeled image with detected bacteria.
3. **Download Results**: You can download the labeled image and a CSV file containing the sizes of detected bacteria.



### Contact
For any issues or questions, please contact Javi :)
""")

# Upload an image
uploaded_file = st.file_uploader("Upload a microscope image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Microscope Image", use_column_width=True)

    # Count bacteria
    num_bacteria, labeled_image, labeled_image_int = count_bacteria(image)
    st.write(f"Number of bacteria detected: {num_bacteria}")

    # Display labeled image
    st.image(labeled_image, caption="Labeled Bacteria", use_column_width=True)

    # Provide option to download the labeled image
    labeled_image_bgr = cv2.cvtColor((labeled_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    result_image_path = "labeled_image.jpg"
    cv2.imwrite(result_image_path, labeled_image_bgr)
    with open(result_image_path, "rb") as file:
        st.download_button("Download Labeled Image", file, file_name="labeled_image.jpg")

    # Calculate and display size of each bacterium
    regions = measure.regionprops(labeled_image_int)
    bacteria_sizes = [region.area for region in regions]
    st.write("Sizes of detected bacteria (in pixels):", bacteria_sizes)

    # Provide option to download the sizes as CSV
    sizes_df = pd.DataFrame(bacteria_sizes, columns=["Bacterium Size (pixels)"])
    sizes_csv = sizes_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Sizes as CSV", sizes_csv, file_name="bacteria_sizes.csv")
