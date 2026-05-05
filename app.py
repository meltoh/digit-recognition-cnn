
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas

from predict import predict_digit
from utils.preprocessing import get_preprocessing_steps


st.set_page_config(
    page_title="Digit Recognition System",
    page_icon="🔢",
    layout="wide"
)


def load_css(file_path):
    with open(file_path) as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


load_css("styles/global.css")


def show_top_3_chart(top_3):
    for item in top_3:
        st.markdown(
            f"""
            <div class="metric-row">
                {item["digit"]} ({item["word"]}) — {item["confidence"]:.2f}%
                <div class="bar-wrap">
                    <div class="bar-fill" style="width: {item["confidence"]}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Handwritten Digit Recognition System</div>
        <div class="app-subtitle">
            A CNN-based system for recognizing single handwritten digits from 0 to 9.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


left_col, right_col = st.columns([1, 1], gap="large")

image = None


with left_col:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Input</div>
            <div class="card-text">
                Draw one digit clearly or upload a handwritten digit image.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    input_method = st.radio(
        "Input method",
        ["Draw digit", "Upload image"],
        horizontal=True
    )

    if input_method == "Draw digit":
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=20,
            stroke_color="black",
            background_color="white",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
        )

        if canvas_result.image_data is not None:
            image_array = canvas_result.image_data.astype(np.uint8)
            image = Image.fromarray(image_array).convert("RGB")

    else:
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", use_column_width=True)


with right_col:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Recognition Result</div>
            <div class="card-text">
                The model returns the predicted digit, word label, and confidence score.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    predict_clicked = st.button("Run Recognition", use_container_width=True)

    if image is None:
        st.info("Draw or upload a digit to begin.")

    elif predict_clicked:
        result = predict_digit(image)

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="predicted-digit">{result["digit"]}</div>
                <div class="predicted-word">{result["word"]}</div>
                <div class="confidence">Confidence: {result["confidence"]:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if result["confidence"] < 70:
            st.markdown(
                """
                <div class="warning-box">
                    Low confidence prediction. The input may be unclear or different from the training style.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="success-box">
                    Prediction confidence is within an acceptable range.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="section-title">Top 3 Predictions</div>',
            unsafe_allow_html=True
        )

        show_top_3_chart(result["top_3"])

        st.markdown(
            '<div class="section-title">Preprocessing Preview</div>',
            unsafe_allow_html=True
        )

        steps = get_preprocessing_steps(image)

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.image(steps["original"], caption="Original", use_column_width=True)

        with p2:
            st.image(steps["grayscale"], caption="Grayscale", use_column_width=True)

        with p3:
            st.image(steps["thresholded"], caption="Thresholded", use_column_width=True)

        with p4:
            st.image(steps["final"], caption="Final 28×28", use_column_width=True)


st.markdown(
    """
    <div class="card">
        <div class="card-title">Processing Pipeline</div>
        <div class="card-text">
            Input image → grayscale conversion → thresholding → crop and center → 28×28 resizing → CNN prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        Handwritten Digit Recognition System
    </div>
    """,
    unsafe_allow_html=True
)
