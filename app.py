
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Retinal Image Enhancement",
    page_icon="👁️",
    layout="wide"
)

st.title("Retinal Image Enhancement Using Python")
st.write("Upload a retinal image and enhance it using different image processing techniques.")

uploaded_file = st.file_uploader(
    "Upload a retinal image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

technique = st.selectbox(
    "Select Enhancement Technique",
    ["CLAHE", "Histogram Equalization", "Gamma Correction", "Gaussian Filtering"]
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_image)

    if technique == "CLAHE":
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        result = cv2.cvtColor(
            cv2.merge((enhanced_l, a, b)),
            cv2.COLOR_LAB2RGB
        )

    elif technique == "Histogram Equalization":
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = cv2.equalizeHist(y)
        result = cv2.cvtColor(
            cv2.merge((y, cr, cb)),
            cv2.COLOR_YCrCb2RGB
        )

    elif technique == "Gamma Correction":
        gamma = 0.5
        table = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)]
        ).astype("uint8")
        result = cv2.LUT(image, table)

    else:
        result = cv2.GaussianBlur(image, (3, 3), 0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Enhanced Image")
        st.image(result, use_container_width=True)

    output = BytesIO()
    Image.fromarray(result).save(output, format="PNG")

    st.download_button(
        "Download Enhanced Image",
        data=output.getvalue(),
        file_name="enhanced_retinal_image.png",
        mime="image/png"
    )
