import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statsbombpy
from scipy.ndimage import gaussian_filter
from statsbombpy import sb
if 'tops' not in st.session_state:
    st.session_state['tops'] = ["Big Things Coming"]
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

# Calculate heatmap function for ball receipts
def calculate_heatmap_frequencies(data, grid_size_x, grid_size_y, pitch_length=104, pitch_width=68):
    """
    Calculate the frequencies for the heatmap.
    """
    heatmap = np.zeros((grid_size_y, grid_size_x))
    cell_size_x = pitch_length / grid_size_x
    cell_size_y = pitch_width / grid_size_y

    for _, row in data.iterrows():
        cell_x = min(int(row['start_x'] // cell_size_x), grid_size_x - 1)
        cell_y = min(int(row['start_y'] // cell_size_y), grid_size_y - 1)
        heatmap[cell_y, cell_x] += 1

    return heatmap

def create_soccer_pitch_with_boxes(ax, pitch_length=104, pitch_width=68):
    """
    Create a soccer pitch layout with detailed markings (including boxes) on the given Matplotlib axis.
    """
    # Pitch Outline & Centre Line
    ax.plot([0, 0], [0, pitch_width], color="black") # Left sideline
    ax.plot([0, pitch_length], [pitch_width, pitch_width], color="black") # Top goal line
    ax.plot([pitch_length, pitch_length], [pitch_width, 0], color="black") # Right sideline
    ax.plot([pitch_length, 0], [0, 0], color="black") # Bottom goal line
    ax.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color="black") # Halfway line

    # Left Penalty Area
    penalty_area_left_x = 16.5
    penalty_area_left_y = 22
    ax.plot([0, penalty_area_left_x], [pitch_width / 2 + penalty_area_left_y, pitch_width / 2 + penalty_area_left_y], color="black") # Top line
    ax.plot([penalty_area_left_x, penalty_area_left_x], [pitch_width / 2 + penalty_area_left_y, pitch_width / 2 - penalty_area_left_y], color="black") # Right line
    ax.plot([penalty_area_left_x, 0], [pitch_width / 2 - penalty_area_left_y, pitch_width / 2 - penalty_area_left_y], color="black") # Bottom line

    # Right Penalty Area
    penalty_area_right_x = pitch_length - 16.5
    ax.plot([pitch_length, penalty_area_right_x], [pitch_width / 2 + penalty_area_left_y, pitch_width / 2 + penalty_area_left_y], color="black") # Top line
    ax.plot([penalty_area_right_x, penalty_area_right_x], [pitch_width / 2 + penalty_area_left_y, pitch_width / 2 - penalty_area_left_y], color="black") # Left line
    ax.plot([penalty_area_right_x, pitch_length], [pitch_width / 2 - penalty_area_left_y, pitch_width / 2 - penalty_area_left_y], color="black") # Bottom line

    # Left 6-yard Box
    six_yard_box_left_x = 5.5
    six_yard_box_left_y = 9.16
    ax.plot([0, six_yard_box_left_x], [pitch_width / 2 + six_yard_box_left_y, pitch_width / 2 + six_yard_box_left_y], color="black") # Top line
    ax.plot([six_yard_box_left_x, six_yard_box_left_x], [pitch_width / 2 + six_yard_box_left_y, pitch_width / 2 - six_yard_box_left_y], color="black") # Right line
    ax.plot([six_yard_box_left_x, 0], [pitch_width / 2 - six_yard_box_left_y, pitch_width / 2 - six_yard_box_left_y], color="black") # Bottom line

    # Right 6-yard Box
    six_yard_box_right_x = pitch_length - 5.5
    ax.plot([pitch_length, six_yard_box_right_x], [pitch_width / 2 + six_yard_box_left_y, pitch_width / 2 + six_yard_box_left_y], color="black") # Top line
    ax.plot([six_yard_box_right_x, six_yard_box_right_x], [pitch_width / 2 + six_yard_box_left_y, pitch_width / 2 - six_yard_box_left_y], color="black") # Left line
    ax.plot([six_yard_box_right_x, pitch_length], [pitch_width / 2 - six_yard_box_left_y, pitch_width / 2 - six_yard_box_left_y], color="black") # Bottom line

    return ax

def plot_heatmap_with_pitch(heatmap, title, grid_size_x, grid_size_y, pitch_length=104, pitch_width=68):
    """
    Plot the heatmap with soccer pitch markings.
    """
    fig, ax = plt.subplots(figsize=(30, 20))
    ax = create_soccer_pitch_with_boxes(ax, pitch_length, pitch_width)

    # Rescale the heatmap to the size of the pitch
    heatmap_rescaled = np.zeros((grid_size_y, grid_size_x))
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            heatmap_rescaled[i, j] = heatmap[int(i * heatmap.shape[0] / grid_size_y), int(j * heatmap.shape[1] / grid_size_x)]

    # appply gaussian kernel smoothing
    sig = 0.9
    # heatmap_doku_start = gaussian_filter(heatmap_doku_start, sigma=sig)
    heatmap_rescaled = gaussian_filter(heatmap_rescaled, sigma=sig)

    ax.imshow(np.flipud(heatmap_rescaled), extent=(0, pitch_length, 0, pitch_width), interpolation='nearest', cmap='magma', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('Pitch Length')
    ax.set_ylabel('Pitch Width')
    return fig 




# List of file names
data_folder = 'data/leagues'

# List all files in the data folder
file_names = os.listdir(data_folder)

# Adjusted function to match file names
def match_to_file(row, file_names):
    # Extract values from the row
    competition_name = row['country_name']
    league_name = row['competition_name']  # Assuming this is the correct column for league name
    season_name = row['season_name']

    # Adjust season_name for year range format
    if '/' in season_name:
        season = ''.join(season_name.split('/')[0][-2:] + season_name.split('/')[1][-2:])
    else:  # Handle single year format
        season = season_name[-2:]

    # Create a pattern to match
    pattern = f"{competition_name}_{league_name}_{season}.csv"

    # Check if the pattern matches any of the file names
    return pattern in file_names



st.title('Player Comparison')

data_folder = 'data/leagues'

# List all files in the data folder
files = os.listdir(data_folder)

# Streamlit selectbox
st.subheader('Choose Base Player')
selected_file = st.selectbox("Select a League", files)
df = pd.read_csv(os.path.join(data_folder, selected_file))
# df = pd.read_csv('data/England_Premier League_2324.csv', low_memory=False)
# Selecting the players
player_choices = df['player'].unique()
selected_player = st.selectbox("Select a Player", player_choices)

# Filter the original DataFrame based on both league and player
data_doku = df[(df['player'] == selected_player) & (df['type']=='Ball Receipt*')]


# data_doku = player_df[player_df.type_name =="dribble"] spadl
#data_doku = player_df[player_df.type =="Ball Receipt*"]


grid_size_x = 16
grid_size_y = 11
# Calculate heatmap frequencies
heatmap_doku_start = calculate_heatmap_frequencies(data_doku, grid_size_x, grid_size_y)

plt1 = plot_heatmap_with_pitch(heatmap_doku_start, '', grid_size_x, grid_size_y)

# Display each plot in a separate column
st.header("Ball Receipts")
st.pyplot(plt1)
