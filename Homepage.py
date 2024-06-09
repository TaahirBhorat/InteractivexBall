import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

logo_url = 'pics/output-onlinepngtools (1).png'
st.sidebar.image(logo_url)


# Display the image and text above the sidebar
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-top: -10px;'>
        <h2 style='margin-bottom: 0;'>xBall</h2>
    </div>
    """,
    unsafe_allow_html=True
)
st.set_page_config(
    page_title="xBall Interactive",
    page_icon="⚽️"
)

st.title("Main Page")

st.write('Here find our Interactive tools and analysis to aid in player, recruitment, match analtysis and playstyle')

