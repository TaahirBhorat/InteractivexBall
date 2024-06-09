import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

logo_url = 'pics/output-onlinepngtools (1).png'

st.set_page_config(
    page_title="xBall Interactive",
    page_icon="⚽️"
)

st.sidebar.image(logo_url)
st.title("Main Page")

st.write('Here find our Interactive tools and analysis to aid in player, recruitment, match analtysis and playstyle')

