import streamlit as st


def center_text(text, font_size=False):
    if not font_size:
        return st.markdown(f"<div style='text-align: center;'>{text}</div>", unsafe_allow_html=True)
    else:
        return st.markdown(
            f"<div style='text-align: center; font-size: {font_size}px;'>{text}</div>", unsafe_allow_html=True
        )
