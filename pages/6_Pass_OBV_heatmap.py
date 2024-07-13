### maybe one plot with all dribbles, and the other one showing most popular zones

import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from statsbombpy import sb
logo_url = 'pics/output-onlinepngtools (1).png'
st.sidebar.image(logo_url)
creds = {"user":"daylesolomon@gmail.com", "passwd": "qIRf28g8"}


# Display the image and text above the sidebar
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-top: -10px;'>
        <h2 style='margin-bottom: 0;'>xBall</h2>
    </div>
    """,
    unsafe_allow_html=True
)
if 'tops' not in st.session_state:
    st.session_state['tops'] = ["Big Things Coming"]
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False


def calculate_heatmap_OBV_frequencies(data, grid_size_x, grid_size_y, pitch_length=104, pitch_width=68):
    """
    Calculate the frequencies for the heatmap.
    """
    heatmap = np.zeros((grid_size_y, grid_size_x))
    cell_size_x = pitch_length / grid_size_x
    cell_size_y = pitch_width / grid_size_y

    for _, row in data.iterrows():
        
        cell_x = min(int(row['start_x'] // cell_size_x), grid_size_x - 1)
        cell_y = min(int(row['start_y'] // cell_size_y), grid_size_y - 1)

        if row['obv_total_net']>0:
            heatmap[cell_y, cell_x] += row['obv_total_net']

    return heatmap

def create_soccer_pitch_with_boxes(ax, pitch_length=104, pitch_width=68):
    """
    Create a soccer pitch layout with detailed markings (including boxes) on the given Matplotlib axis.
    """
    pitch_lines = 'lightgrey'
    # Pitch Outline & Centre Line
    ax.plot([0, 0], [0, pitch_width], color=pitch_lines) # Left sideline
    ax.plot([0, pitch_length], [pitch_width, pitch_width], color=pitch_lines) # Top goal line
    ax.plot([pitch_length, pitch_length], [pitch_width, 0], color=pitch_lines) # Right sideline
    ax.plot([pitch_length, 0], [0, 0], color=pitch_lines) # Bottom goal line
    ax.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color=pitch_lines) # Halfway line

    # Left Penalty Area
    penalty_area_left_x = 16.5
    penalty_area_left_y = 22
    ax.plot([0, penalty_area_left_x], [pitch_width / 2 + penalty_area_left_y, pitch_width / 2 + penalty_area_left_y], color=pitch_lines) # Top line
    ax.plot([penalty_area_left_x, penalty_area_left_x], [pitch_width / 2 + penalty_area_left_y, pitch_width / 2 - penalty_area_left_y], color=pitch_lines) # Right line
    ax.plot([penalty_area_left_x, 0], [pitch_width / 2 - penalty_area_left_y, pitch_width / 2 - penalty_area_left_y], color=pitch_lines) # Bottom line

    # Right Penalty Area
    penalty_area_right_x = pitch_length - 16.5
    ax.plot([pitch_length, penalty_area_right_x], [pitch_width / 2 + penalty_area_left_y, pitch_width / 2 + penalty_area_left_y], color=pitch_lines) # Top line
    ax.plot([penalty_area_right_x, penalty_area_right_x], [pitch_width / 2 + penalty_area_left_y, pitch_width / 2 - penalty_area_left_y], color=pitch_lines) # Left line
    ax.plot([penalty_area_right_x, pitch_length], [pitch_width / 2 - penalty_area_left_y, pitch_width / 2 - penalty_area_left_y], color=pitch_lines) # Bottom line

    # Left 6-yard Box
    six_yard_box_left_x = 5.5
    six_yard_box_left_y = 9.16
    ax.plot([0, six_yard_box_left_x], [pitch_width / 2 + six_yard_box_left_y, pitch_width / 2 + six_yard_box_left_y], color=pitch_lines) # Top line
    ax.plot([six_yard_box_left_x, six_yard_box_left_x], [pitch_width / 2 + six_yard_box_left_y, pitch_width / 2 - six_yard_box_left_y], color=pitch_lines) # Right line
    ax.plot([six_yard_box_left_x, 0], [pitch_width / 2 - six_yard_box_left_y, pitch_width / 2 - six_yard_box_left_y], color=pitch_lines) # Bottom line

    # Right 6-yard Box
    six_yard_box_right_x = pitch_length - 5.5
    ax.plot([pitch_length, six_yard_box_right_x], [pitch_width / 2 + six_yard_box_left_y, pitch_width / 2 + six_yard_box_left_y], color=pitch_lines) # Top line
    ax.plot([six_yard_box_right_x, six_yard_box_right_x], [pitch_width / 2 + six_yard_box_left_y, pitch_width / 2 - six_yard_box_left_y], color=pitch_lines) # Left line
    ax.plot([six_yard_box_right_x, pitch_length], [pitch_width / 2 - six_yard_box_left_y, pitch_width / 2 - six_yard_box_left_y], color=pitch_lines) # Bottom line

    return ax

def plot_heatmap_with_pitch(heatmap, title, grid_size_x, grid_size_y, pitch_length=104, pitch_width=68):
    """
    Plot the heatmap with soccer pitch markings.
    """
    # heatmap = heatmap_doku_start
    fig, ax = plt.subplots(figsize=(30, 20))
    ax = create_soccer_pitch_with_boxes(ax, pitch_length, pitch_width)

    # Set dark background for the figure and axis
    fig.patch.set_facecolor('black')
    #ax.set_facecolor('black')
    # Rescale the heatmap to the size of the pitch
    heatmap_rescaled = np.zeros((grid_size_y, grid_size_x))
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            heatmap_rescaled[i, j] = heatmap[int(i * heatmap.shape[0] / grid_size_y), int(j * heatmap.shape[1] / grid_size_x)]

    # appply gaussian kernel smoothing
    sig = 0.9
    # heatmap_doku_start = gaussian_filter(heatmap_doku_start, sigma=sig)
    heatmap_rescaled = gaussian_filter(heatmap_rescaled, sigma=sig)
    
    # Plot the heatmap
    ax.imshow(np.flipud(heatmap_rescaled), extent=(0, pitch_length, 0, pitch_width), interpolation='nearest', cmap='magma', alpha=0.8)
    #ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    return fig 




st.title('Pass OBV Heatmap')

data_folder = 'data/leagues'

# List all files in the data folder
files = os.listdir(data_folder)

# Streamlit selectbox
st.subheader('Choose Base Player')
selected_file = st.selectbox("Select a League", files)
df = pd.read_csv(os.path.join(data_folder, selected_file), low_memory=False)
# df = pd.read_csv('data/leagues/South Africa_PSL_2324.csv', low_memory=False)

# Selecting the players
player_choices = df['player'].unique()
selected_player = st.selectbox("Select a Player", player_choices)
#selected_player = 'Kimvuidi Keikie Karim'

# country, league, season = extract_file_info(selected_file)
# player_position = getPosition(country, league, season, selected_player)

# Filter the original DataFrame based on both league and player
# selected_player = 'Jeremy Doku'
data_doku = df[(df['player'] == selected_player) & (df['type']=='Pass')]

grid_size_x = 16
grid_size_y = 10
# Calculate heatmap frequencies
heatmap_doku_start = calculate_heatmap_OBV_frequencies(data_doku, grid_size_x, grid_size_y)



plt1 = plot_heatmap_with_pitch(heatmap_doku_start,"",grid_size_x, grid_size_y)
# Display each plot in a separate column

st.pyplot(plt1)



