import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas

from pixelation import process_digit, render_pixel_grid_with_values
from canvas_utils import canvas_json_or_rgba_to_grayscale


st.set_page_config(page_title="Digit Pixelation Visualizer", layout="wide")
st.title("Digit Pixelation Visualizer")

st.markdown("Draw 0–9, then see the same MNIST-style 32×32 pixel grid.")

clear_clicked = st.button("Clear All", use_container_width=True)

# One independent canvas and result column per digit.
columns = st.columns(10)

for digit, column in enumerate(columns):
    with column:
        st.markdown(f"**{digit}**")

        canvas_key = f"digit-canvas-{digit}"
        result_key = f"digit-result-{digit}"

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=20,
            stroke_color="rgb(255, 255, 255)",
            background_color="rgb(0, 0, 0)",
            background_image=None,
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            point_display_radius=0,
            key=f"{canvas_key}-{'clear' if clear_clicked else 'active'}",
            display_toolbar=False,
            initial_drawing=None,
        )

        st.markdown("<div style='text-align:center; font-size:22px;'>↓</div>", unsafe_allow_html=True)
        st.caption("raw → pixelated")

        if canvas_result.image_data is not None:
            gray = canvas_json_or_rgba_to_grayscale(canvas_result.image_data)
            pixel_array = process_digit(canvas_result.image_data)
        else:
            pixel_array = np.zeros((32, 32), dtype=np.uint8)

        grid_image = render_pixel_grid_with_values(pixel_array, upscale_to=320)
        st.image(grid_image, caption=f"Digit {digit}", use_container_width=True)
