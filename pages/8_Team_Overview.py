import streamlit as st
import pandas as pd
import numpy as np
import mplsoccer
import matplotlib.pyplot as pl
data = pd.read_csv("data/leagues/fullseason (1).csv").iloc[:,1:]
data['player_season_obv_shot_90'].fillna(0, inplace=True)
data['player_season_obv_90'] = data['player_season_obv_90']  - data['player_season_obv_shot_90']

st.title("PSL 23/24 Team overview")

team_options = data["team_name"].unique()
selected_team = st.selectbox('Select a team:', team_options)
def shorten_name(name):
    parts = name.split()
    if len(parts) > 2:
        return f"{parts[0]} {parts[-1]}"
    return name
def shorten_name(name):
    parts = name.split()
    if len(parts) > 1:
        return f"{parts[0][0]}. {parts[-1]}"
    return name

max_games = 28
selected_games = st.selectbox('Select the minimum number of games:', list(range(1, 21)))

# def map_filter_rank(df, min_90s_played):
#     # Define the mapping dictionary
#     position_map = {
#         'Center Forward': 'CF', 
#         'Left Wing': 'LW',
#         'Right Wing': 'RW',
#         'Left Back': 'LB',
#         'Right Back': 'RB',
#         'Center Midfield': 'M',
#         'Left Midfield': 'LW',
#         'Left Midfielder': 'LW',
#         'Right Midfield': 'RW',
#         'Right Midfielder': 'RW',
#         'Defensive Midfielder': 'DM',
#         'Attacking Midfielder': 'M',
#         'Center Back': 'CB',
#         'Goalkeeper': 'GK',
#         'Left Centre Back': 'CB',
#         'Right Centre Back': 'CB',
#         'Left Defensive Midfielder': 'DM',
#         'Right Defensive Midfielder': 'DM',
#         'Centre Attacking Midfielder': 'M',
#         'Centre Defensive Midfielder': 'DM',
#         'Left Wing Back': 'LB',
#         'Right Wing Back': 'RB',
#         'Right Forward': 'RW',
#         'Left Forward': 'LW',
#         'Centre Forward': 'CF',
#         'Left Centre Midfielder': 'M',
#         'Left Centre Forward': 'CF',
#         'Right Centre Forward': 'CF',
#         'Right Midfielder': 'RW',
#         'Right Centre Midfielder': 'M',
#         'Left Attacking Midfielder': 'M',
#         'Right Attacking Midfielder': 'M'
#     }

#     # Map the primary_position to position
#     df['position'] = df['primary_position'].map(position_map)

#     # Handle any NaN values (if any remain)
#     df.dropna(subset=['position'], inplace=True)

#     # Filter out players based on min_90s_played
#     df_filtered = df[df['player_season_90s_played'] >= min_90s_played].copy()

#     # Create the position_rank column based on player_season_obv_90 within each position, except for CF
#     df_filtered['position_rank'] = df_filtered.groupby('position').apply(
#         lambda x: x['player_season_np_xg_90'].rank(ascending=False, method='min') if x.name == 'CF'
#         else x['player_season_obv_90'].rank(ascending=False, method='min')
#     ).reset_index(level=0, drop=True)

#     # Calculate FIFA_score (1-100 scale) within each position
#     df_filtered['FIFA_score'] = df_filtered.groupby('position')['position_rank'].transform(
#         lambda x: 100 * (1 - (x - 1) / (x.max() - 1))
#     )
#     df_filtered['player_name'] = df_filtered['player_name'].apply(shorten_name)

#     return df_filtered

import pandas as pd

def map_filter_rank(df, min_90s_played):
    # Define the mapping dictionary
    position_map = {
        'Center Forward': 'CF',
        'Left Wing': 'LW',
        'Right Wing': 'RW',
        'Left Back': 'LB',
        'Right Back': 'RB',
        'Center Midfield': 'M',
        'Left Midfield': 'LW',
        'Left Midfielder': 'LW',
        'Right Midfield': 'RW',
        'Right Midfielder': 'RW',
        'Defensive Midfielder': 'DM',
        'Attacking Midfielder': 'M',
        'Center Back': 'CB',
        'Goalkeeper': 'GK',
        'Left Centre Back': 'CB',
        'Right Centre Back': 'CB',
        'Left Defensive Midfielder': 'DM',
        'Right Defensive Midfielder': 'DM',
        'Centre Attacking Midfielder': 'M',
        'Centre Defensive Midfielder': 'DM',
        'Left Wing Back': 'LW',
        'Right Wing Back': 'RW',
        'Right Forward': 'RW',
        'Left Forward': 'LW',
        'Centre Forward': 'CF',
        'Left Centre Midfielder': 'M',
        'Left Centre Forward': 'CF',
        'Right Centre Forward': 'CF',
        'Right Centre Midfielder': 'M',
        'Left Attacking Midfielder': 'M',
        'Right Attacking Midfielder': 'M'
    }

    # Map the primary_position to position
    df['position'] = df['primary_position'].map(position_map)

    # Handle any NaN values (if any remain)
    df.dropna(subset=['position'], inplace=True)

    # Filter out players based on min_90s_played
    df_filtered = df[df['player_season_90s_played'] >= min_90s_played].copy()

    # Calculate the sum of player_season_np_xg_90 and player_season_obv_90 for LW and RW
    df_filtered['sum_x_obv'] = df_filtered['player_season_np_xg_90'] + df_filtered['player_season_obv_90']

    # Create the position_rank column based on player_season_obv_90 within each position, except for CF, LW, and RW
    df_filtered['position_rank'] = df_filtered.groupby('position').apply(
        lambda x: x['sum_x_obv'].rank(ascending=False, method='min') if x.name in ['LW', 'RW']
        else x['sum_x_obv'].rank(ascending=False, method='min') if x.name == 'CF'
         else x['player_season_obv_gk_90'].rank(ascending=False, method='min') if x.name == 'GK'
        else x['player_season_obv_90'].rank(ascending=False, method='min')
    ).reset_index(level=0, drop=True)

    # Calculate FIFA_score (1-100 scale) within each position
    df_filtered['FIFA_score'] = df_filtered.groupby('position')['position_rank'].transform(
        lambda x: 100 * (1 - (x - 1) / (x.max() - 1))
    )

    # Optional: shorten player names if needed (assuming shorten_name function is defined)
    if 'shorten_name' in globals():
        df_filtered['player_name'] = df_filtered['player_name'].apply(shorten_name)

    return df_filtered



data = map_filter_rank(data, selected_games)
dat = data
team_data = data[data['team_name'] == selected_team] 
# Top 3 players for each position
top_6_gk = team_data[team_data['position'] == 'GK'].nlargest(6, 'player_season_obv_gk_90')
gks = dat[dat['position'] == 'GK']['position_rank'].values

top_3_players = team_data.groupby('position').apply(lambda x: x.nlargest(3, 'player_season_obv_90'))

# Top 6 players for CB
top_6_cb = team_data[team_data['position'] == 'CB'].nlargest(6, 'player_season_obv_90')
cbs = dat[dat['position'] == 'CB']['position_rank'].values

top_6_lb = team_data[team_data['position'] == 'LB'].nlargest(6, 'player_season_obv_90')
lbs = dat[dat['position'] == 'LB']['position_rank'].values

top_6_rb = team_data[team_data['position'] == 'RB'].nlargest(6, 'player_season_obv_90')
rbs = dat[dat['position'] == 'RB']['position_rank'].values

# Top 9 players for M
top_9_m = team_data[team_data['position'] == 'M'].nlargest(9, 'player_season_obv_90')
ms = dat[dat['position'] == 'M']['position_rank'].values

# Top 9 players for M
top_9_dm = team_data[team_data['position'] == 'DM'].nlargest(9, 'player_season_obv_90')
dms = dat[dat['position'] == 'DM']['position_rank'].values

top_6_lw = team_data[team_data['position'] == 'LW'].nlargest(6, 'player_season_obv_90')
lws = dat[dat['position'] == 'LW']['position_rank'].values


top_6_rw = team_data[team_data['position'] == 'RW'].nlargest(6, 'player_season_obv_90')
rws = dat[dat['position'] == 'RW']['position_rank'].values


top_6_cf = team_data[team_data['position'] == 'CF'].nlargest(6, 'player_season_np_xg_90')
cfs = dat[dat['position'] == 'CF']['position_rank'].values

logo_paths = {
    'AmaZulu FC': 'data/clublogos/amazulu_fc_badge.png',
    'Cape Town City FC': 'data/clublogos/cape_town_city_fc_badge.png',
    'Cape Town Spurs': 'data/clublogos/cape_town_spurs_badge.png',
    'Chippa United FC': 'data/clublogos/chippa_united_fc_badge.png',
    'Kaizer Chiefs FC': 'data/clublogos/kaizer_chiefs_fc_badge.png',
    'Lamontville Golden Arrows FC': 'data/clublogos/lamontville_golden_arrows_fc_badge.png',
    'Mamelodi Sundowns FC': 'data/clublogos/mamelodi_sundowns_fc_badge.png',
    'Orlando Pirates FC': 'data/clublogos/orlando_pirates_fc_badge.png',
    'Polokwane City': 'data/clublogos/polokwane_city_badge.png',
    'Richards Bay FC': 'data/clublogos/richards_bay_fc_badge.png',
    'Royal AM FC': 'data/clublogos/royal_am_fc_badge.png',
    'Sekhukhune United': 'data/clublogos/sekhukhune_united_badge.png',
    'Stellenbosch FC': 'data/clublogos/stellenbosch_fc_badge.png',
    'SuperSport United FC': 'data/clublogos/supersport_united_fc_badge.png',
    'Swallows FC': 'data/clublogos/swallows_fc_badge.png',
    'TS Galaxy FC': 'data/clublogos/ts_galaxy_fc_badge.png'
}

import matplotlib.pyplot as plt
import mplsoccer
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# Create the pitch
pitch = mplsoccer.Pitch(pitch_color='#303030', line_color='gray')

# Set the figure size
fig, ax = pitch.draw(figsize=(40, 18))

# Set the background color of the whole plot
fig.patch.set_facecolor('#303030')

# Create a reversed colormap
colors = [(0, 1, 0), (1, 0, 0)]  # Green to red (reversed)
n_bins = 100  # Discretize into 100 bins
cmap_name = 'green_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
cmap = plt.get_cmap(cm)

colors = [(0, 1, 0), (0.6, 0.4, 0), (0.8, 0.4, 0), (1, 0, 0)]  # Green to yellow-brown to red
positions = [0, 0.3, 0.7, 1]  # Positions for the colors in the colormap

# Create a skewed colormap
cmap_name = 'green_yellowbrown_red'
cm = LinearSegmentedColormap.from_list(cmap_name, list(zip(positions, colors)))

cmap = plt.get_cmap(cm)

offset = 35

####### CENTERBACKS ################################################################################################################################################################
x_cb = [15] * len(top_6_cb)  # Same x coordinates for all points
y_cb = [40] * len(top_6_cb)  # Same y coordinates for all points
labels_cb = top_6_cb['player_name']
values_cb = top_6_cb['FIFA_score'].values
pos_rank_cb = top_6_cb['position_rank'].values
games_value_cb = top_6_cb['player_season_90s_played'].values

# Sort the indices by the FIFA score (highest first)
sorted_indices_cb = sorted(range(len(values_cb)), key=lambda k: values_cb[k], reverse=True)

# Sort values, labels, position ranks, and games played according to sorted indices
sorted_values_cb = values_cb[sorted_indices_cb]
sorted_labels_cb = labels_cb.iloc[sorted_indices_cb].values
sorted_pos_rank_cb = pos_rank_cb[sorted_indices_cb]
sorted_games_cb = games_value_cb[sorted_indices_cb]
norm_cb = plt.Normalize(min(cbs), max(cbs))
colors_cb = [cmap(norm_cb(value)) for value in sorted_pos_rank_cb]
offset = offset+10
# Adjust the annoations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_cb)):
    # Full annotation text
    full_text = f"{sorted_labels_cb[i]}"

    # Base annotation for the full text
    ax.annotate(
        full_text,
        (x_cb[i], y_cb[i]),
        color=colors_cb[i],
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )

    # Calculate the position and dimensions of the sorted_values_cb part
    label_text = f"{sorted_labels_cb[i]} "
    value_text = f"{sorted_values_cb[i]:.0f}"
    additional_text = f" {sorted_games_cb[i]:.0f}"

    # Temporarily annotate the label part to measure its width
    text_before = ax.annotate(
        label_text,
        (x_cb[i], y_cb[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_before = text_before.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_before.remove()
    x_offset_games = offset + 60 

    # Temporarily annotate the value part to measure its width
    text_value = ax.annotate(
        value_text,
        (x_cb[i], y_cb[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(bbox_before.width / 2, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_value = text_value.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_value.remove()

    # Calculate the x position offset for the value part
    x_offset_value = (bbox_before.width -2) / 2

    # Annotate the value part with a filled circle around it
    ax.annotate(
        value_text,
        (x_cb[i], y_cb[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(x_offset_value, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="circle,pad=0.3", edgecolor=colors_cb[i], facecolor=colors_cb[i], alpha=0.9)
    )
    x_offset_value = (bbox_before.width + 30) / 2
    ax.annotate(
        additional_text,
        (x_cb[i], y_cb[i]),
        fontsize=20,
        ha='center',
        va='bottom',
        xytext=(x_offset_value + bbox_value.width +0.8 / 2, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="round4,pad=0.02", edgecolor='white', facecolor='white', alpha=0.8)
    )




####### LEFT BACKS ################################################################################################################################################################
x_lb = [34] * len(top_6_lb)  # Same x coordinates for all points
y_lb = [10] * len(top_6_lb)
labels_lb = top_6_lb['player_name']
values_lb = top_6_lb['FIFA_score'].values
pos_rank_lb = top_6_lb['position_rank'].values
games_value_lb = top_6_lb['player_season_90s_played'].values

# Sort the indices by the FIFA score (highest first)
sorted_indices_lb = sorted(range(len(values_lb)), key=lambda k: values_lb[k], reverse=True)

# Sort values, labels, position ranks, and games played according to sorted indices
sorted_values_lb = values_lb[sorted_indices_lb]
sorted_labels_lb = labels_lb.iloc[sorted_indices_lb].values
sorted_pos_rank_lb = pos_rank_lb[sorted_indices_lb]
sorted_games_lb = games_value_lb[sorted_indices_lb]

norm_lb = plt.Normalize(min(lbs), max(lbs))
colors_lb = [cmap(norm_lb(value)) for value in sorted_pos_rank_lb]


# Adjust the annoations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_lb)):
    # Full annotation text
    full_text = f"{sorted_labels_lb[i]}"

    # Base annotation for the full text
    ax.annotate(
        full_text,
        (x_lb[i], y_lb[i]),
        color=colors_lb[i],
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )

    # Calculate the position and dimensions of the sorted_values_lb part
    label_text = f"{sorted_labels_lb[i]} "
    value_text = f"{sorted_values_lb[i]:.0f}"
    additional_text = f" {sorted_games_lb[i]:.0f}"

    # Temporarily annotate the label part to measure its width
    text_before = ax.annotate(
        label_text,
        (x_lb[i], y_lb[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_before = text_before.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_before.remove()
    x_offset_games = offset + 60 

    # Temporarily annotate the value part to measure its width
    text_value = ax.annotate(
        value_text,
        (x_lb[i], y_lb[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(bbox_before.width / 2, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_value = text_value.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_value.remove()

    # Calculate the x position offset for the value part
    x_offset_value = (bbox_before.width -2) / 2

    # Annotate the value part with a filled circle around it
    ax.annotate(
        value_text,
        (x_lb[i], y_lb[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(x_offset_value, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="circle,pad=0.3", edgecolor=colors_lb[i], facecolor=colors_lb[i], alpha=0.9)
    )
    x_offset_value = (bbox_before.width + 30) / 2
    ax.annotate(
        additional_text,
        (x_lb[i], y_lb[i]),
        fontsize=20,
        ha='center',
        va='bottom',
        xytext=(x_offset_value + bbox_value.width +0.8 / 2, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="round4,pad=0.02", edgecolor='white', facecolor='white', alpha=0.8)
    )



####### RIGHT BACKS ################################################################################################################################################################
x_rb = [34] * len(top_6_rb)  # Same x coordinates for all points
y_rb = [70] * len(top_6_rb)  # Same y coordinates for all points
labels_rb = top_6_rb['player_name']
values_rb = top_6_rb['FIFA_score'].values
pos_rank_rb = top_6_rb['position_rank'].values
games_value_rb = top_6_rb['player_season_90s_played'].values

# Sort the indices by the FIFA score (highest first)
sorted_indices_rb = sorted(range(len(values_rb)), key=lambda k: values_rb[k], reverse=True)

# Sort values, labels, position ranks, and games played according to sorted indices
sorted_values_rb = values_rb[sorted_indices_rb]
sorted_labels_rb = labels_rb.iloc[sorted_indices_rb].values
sorted_pos_rank_rb = pos_rank_rb[sorted_indices_rb]
sorted_games_rb = games_value_rb[sorted_indices_rb]


norm_rb = plt.Normalize(min(rbs), max(rbs))
colors_rb = [cmap(norm_rb(value)) for value in sorted_pos_rank_rb]


# Adjust the annoations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_rb)):
    # Full annotation text
    full_text = f"{sorted_labels_rb[i]}"

    # Base annotation for the full text
    ax.annotate(
        full_text,
        (x_rb[i], y_rb[i]),
        color=colors_rb[i],
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )

    # Calculate the position and dimensions of the sorted_values_rb part
    label_text = f"{sorted_labels_rb[i]} "
    value_text = f"{sorted_values_rb[i]:.0f}"
    additional_text = f" {games_value_rb[i]:.0f}"

    # Temporarily annotate the label part to measure its width
    text_before = ax.annotate(
        label_text,
        (x_rb[i], y_rb[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_before = text_before.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_before.remove()
    x_offset_games = offset + 60 

    # Temporarily annotate the value part to measure its width
    text_value = ax.annotate(
        value_text,
        (x_rb[i], y_rb[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(bbox_before.width / 2, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_value = text_value.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_value.remove()

    # Calculate the x position offset for the value part
    x_offset_value = (bbox_before.width -2) / 2

    # Annotate the value part with a filled circle around it
    ax.annotate(
        value_text,
        (x_rb[i], y_rb[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(x_offset_value, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="circle,pad=0.3", edgecolor=colors_rb[i], facecolor=colors_rb[i], alpha=0.9)
    )
    x_offset_value = (bbox_before.width + 30) / 2
    ax.annotate(
        additional_text,
        (x_rb[i], y_rb[i]),
        fontsize=20,
        ha='center',
        va='bottom',
        xytext=(x_offset_value + bbox_value.width +0.8 / 2, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="round4,pad=0.02", edgecolor='white', facecolor='white', alpha=0.8)
    )

####### DEFENSIVE MIDFIELDERS ################################################################################################################################################################
# Example data for plotting
x_dm = [40] * len(top_9_dm)  # Same x coordinates for all points
y_dm = [40] * len(top_9_dm)  # Same y coordinates for all points
labels_dm = top_9_dm['player_name']
values_dm = top_9_dm['FIFA_score'].values
pos_rank_dm = top_9_dm['position_rank'].values
games_value_dm = top_9_dm['player_season_90s_played'].values

# Sort the indices by the FIFA score (highest first)
sorted_indices_dm = sorted(range(len(values_dm)), key=lambda k: values_dm[k], reverse=True)

# Sort values, labels, position ranks, and games played according to sorted indices
sorted_values_dm = values_dm[sorted_indices_dm]
sorted_labels_dm = labels_dm.iloc[sorted_indices_dm].values
sorted_pos_rank_dm = pos_rank_dm[sorted_indices_dm]
sorted_games_dm = games_value_dm[sorted_indices_dm]


norm_dm = plt.Normalize(min(dms), max(dms))
colors_dm = [cmap(norm_dm(value)) for value in sorted_pos_rank_dm]

# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_dm)):
    # Full annotation text
    full_text = f"{sorted_labels_dm[i]}"

    # Base annotation for the full text
    ax.annotate(
        full_text,
        (x_dm[i], y_dm[i]),
        color=colors_dm[i],
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )

    # Calculate the position and dimensions of the sorted_values_m part
    label_text = f"{sorted_labels_dm[i]} "
    value_text = f"{sorted_values_dm[i]:.0f}"
    additional_text = f" {sorted_games_dm[i]:.0f}"

    # Temporarily annotate the label part to measure its width
    text_before = ax.annotate(
        label_text,
        (x_dm[i], y_dm[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_before = text_before.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_before.remove()
    x_offset_games = offset + 60 

    # Temporarily annotate the value part to measure its width
    text_value = ax.annotate(
        value_text,
        (x_dm[i], y_dm[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(bbox_before.width / 2, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_value = text_value.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_value.remove()

    # Calculate the x position offset for the value part
    x_offset_value = (bbox_before.width -2) / 2

    # Annotate the value part with a filled circle around it
    ax.annotate(
        value_text,
        (x_dm[i], y_dm[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(x_offset_value, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="circle,pad=0.3", edgecolor=colors_dm[i], facecolor=colors_dm[i], alpha=0.9)
    )
    x_offset_value = (bbox_before.width + 30) / 2
    ax.annotate(
        additional_text,
        (x_dm[i], y_dm[i]),
        fontsize=20,
        ha='center',
        va='bottom',
        xytext=(x_offset_value + bbox_value.width +0.8 / 2, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="round4,pad=0.02", edgecolor='white', facecolor='white', alpha=0.8)
    )



####### MIDFIELDERS ################################################################################################################################################################
# Example data for plotting
x_m = [70] * len(top_9_m)  # Same x coordinates for all points
y_m = [40] * len(top_9_m)  # Same y coordinates for all points
labels_m = top_9_m['player_name']
values_m = top_9_m['FIFA_score'].values
pos_rank_m = top_9_m['position_rank'].values
games_value_m = top_9_m['player_season_90s_played'].values

# Sort the indices by the FIFA score (highest first)
sorted_indices_m = sorted(range(len(values_m)), key=lambda k: values_m[k], reverse=True)

# Sort values, labels, position ranks, and games played according to sorted indices
sorted_values_m = values_m[sorted_indices_m]
sorted_labels_m = labels_m.iloc[sorted_indices_m].values
sorted_pos_rank_m = pos_rank_m[sorted_indices_m]
sorted_games_m = games_value_m[sorted_indices_m]


norm_m = plt.Normalize(min(ms), max(ms))
colors_m = [cmap(norm_m(value)) for value in sorted_pos_rank_m]

# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_m)):
    # Full annotation text
    full_text = f"{sorted_labels_m[i]}"

    # Base annotation for the full text
    ax.annotate(
        full_text,
        (x_m[i], y_m[i]),
        color=colors_m[i],
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )

    # Calculate the position and dimensions of the sorted_values_m part
    label_text = f"{sorted_labels_m[i]} "
    value_text = f"{sorted_values_m[i]:.0f}"
    additional_text = f" {sorted_games_m[i]:.0f}"

    # Temporarily annotate the label part to measure its width
    text_before = ax.annotate(
        label_text,
        (x_m[i], y_m[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_before = text_before.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_before.remove()
    x_offset_games = offset + 60 

    # Temporarily annotate the value part to measure its width
    text_value = ax.annotate(
        value_text,
        (x_m[i], y_m[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(bbox_before.width / 2, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_value = text_value.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_value.remove()

    # Calculate the x position offset for the value part
    x_offset_value = (bbox_before.width -2) / 2

    # Annotate the value part with a filled circle around it
    ax.annotate(
        value_text,
        (x_m[i], y_m[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(x_offset_value, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="circle,pad=0.3", edgecolor=colors_m[i], facecolor=colors_m[i], alpha=0.9)
    )
    x_offset_value = (bbox_before.width + 30) / 2
    ax.annotate(
        additional_text,
        (x_m[i], y_m[i]),
        fontsize=20,
        ha='center',
        va='bottom',
        xytext=(x_offset_value + bbox_value.width +0.8 / 2, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="round4,pad=0.02", edgecolor='white', facecolor='white', alpha=0.8)
    )


####### LEFT WING ################################################################################################################################################################
# Example data for plotting
x_lw = [100] * len(top_6_lw)  # Same x coordinates for all points
y_lw = [10] * len(top_6_lw)  # Same y coordinates for all points
labels_lw = top_6_lw['player_name']
values_lw = top_6_lw['FIFA_score'].values
pos_rank_lw = top_6_lw['position_rank'].values
games_value_lw = top_6_lw['player_season_90s_played'].values

# Sort the indices by the FIFA score (highest first)
sorted_indices_lw = sorted(range(len(values_lw)), key=lambda k: values_lw[k], reverse=True)

# Sort values, labels, position ranks, and games played according to sorted indices
sorted_values_lw = values_lw[sorted_indices_lw]
sorted_labels_lw = labels_lw.iloc[sorted_indices_lw].values
sorted_pos_rank_lw = pos_rank_lw[sorted_indices_lw]
sorted_games_lw = games_value_lw[sorted_indices_lw]


norm_lw = plt.Normalize(min(lws), max(lws))
colors_lw = [cmap(norm_lw(value)) for value in sorted_pos_rank_lw]

# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_lw)):
    # Full annotation text
    full_text = f"{sorted_labels_lw[i]}"

    # Base annotation for the full text
    ax.annotate(
        full_text,
        (x_lw[i], y_lw[i]),
        color=colors_lw[i],
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )

    # Calculate the position and dimensions of the sorted_values_lw part
    label_text = f"{sorted_labels_lw[i]} "
    value_text = f"{sorted_values_lw[i]:.0f}"
    additional_text = f" {sorted_games_lw[i]:.0f}"

    # Temporarily annotate the label part to measure its width
    text_before = ax.annotate(
        label_text,
        (x_lw[i], y_lw[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_before = text_before.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_before.remove()
    x_offset_games = offset + 60 

    # Temporarily annotate the value part to measure its width
    text_value = ax.annotate(
        value_text,
        (x_lw[i], y_lw[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(bbox_before.width / 2, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_value = text_value.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_value.remove()

    # Calculate the x position offset for the value part
    x_offset_value = (bbox_before.width -2) / 2

    # Annotate the value part with a filled circle around it
    ax.annotate(
        value_text,
        (x_lw[i], y_lw[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(x_offset_value, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="circle,pad=0.3", edgecolor=colors_lw[i], facecolor=colors_lw[i], alpha=0.9)
    )
    x_offset_value = (bbox_before.width + 30) / 2
    ax.annotate(
        additional_text,
        (x_lw[i], y_lw[i]),
        fontsize=20,
        ha='center',
        va='bottom',
        xytext=(x_offset_value + bbox_value.width +0.8 / 2, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="round4,pad=0.02", edgecolor='white', facecolor='white', alpha=0.8)
    )

####### RIGHT WING ################################################################################################################################################################
# Example data for plotting
x_rw = [100] * len(top_6_rw)  # Same x coordinates for all points
y_rw = [70] * len(top_6_rw)  # Same y coordinates for all points
labels_rw = top_6_rw['player_name']
values_rw = top_6_rw['FIFA_score'].values
pos_rank_rw = top_6_rw['position_rank'].values
games_value_rw = top_6_rw['player_season_90s_played'].values

# Sort the indices by the FIFA score (highest first)
sorted_indices_rw = sorted(range(len(values_rw)), key=lambda k: values_rw[k], reverse=True)

# Sort values, labels, position ranks, and games played according to sorted indices
sorted_values_rw = values_rw[sorted_indices_rw]
sorted_labels_rw = labels_rw.iloc[sorted_indices_rw].values
sorted_pos_rank_rw = pos_rank_rw[sorted_indices_rw]
sorted_games_rw = games_value_rw[sorted_indices_rw]

norm_rw = plt.Normalize(min(rws), max(rws))
colors_rw = [cmap(norm_rw(value)) for value in sorted_pos_rank_rw]


# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_rw)):
    # Full annotation text
    full_text = f"{sorted_labels_rw[i]}"

    # Base annotation for the full text
    ax.annotate(
        full_text,
        (x_rw[i], y_rw[i]),
        color=colors_rw[i],
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )

    # Calculate the position and dimensions of the sorted_values_rw part
    label_text = f"{sorted_labels_rw[i]} "
    value_text = f"{sorted_values_rw[i]:.0f}"
    additional_text = f" {sorted_games_rw[i]:.0f}"

    # Temporarily annotate the label part to measure its width
    text_before = ax.annotate(
        label_text,
        (x_rw[i], y_rw[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_before = text_before.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_before.remove()
    x_offset_games = offset + 60 

    # Temporarily annotate the value part to measure its width
    text_value = ax.annotate(
        value_text,
        (x_rw[i], y_rw[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(bbox_before.width / 2, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_value = text_value.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_value.remove()

    # Calculate the x position offset for the value part
    x_offset_value = (bbox_before.width -1) / 2

    # Annotate the value part with a filled circle around it
    ax.annotate(
        value_text,
        (x_rw[i], y_rw[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(x_offset_value, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="circle,pad=0.3", edgecolor=colors_rw[i], facecolor=colors_rw[i], alpha=0.9)
    )
    x_offset_value = (bbox_before.width + 30) / 2
    ax.annotate(
        additional_text,
        (x_rw[i], y_rw[i]),
        fontsize=20,
        ha='center',
        va='bottom',
        xytext=(x_offset_value + bbox_value.width +0.8 / 2, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="round4,pad=0.02", edgecolor='white', facecolor='white', alpha=0.8)
    )



####### CENTER FORWARD ################################################################################################################################################################
# Example data for plotting
x_cf = [100] * len(top_6_cf)  # Same x coordinates for all points
y_cf = [40] * len(top_6_cf)  # Same y coordinates for all points
labels_cf = top_6_cf['player_name']
values_cf = top_6_cf['FIFA_score'].values
pos_rank_cf = top_6_cf['position_rank'].values
games_value_cf = top_6_cf['player_season_90s_played'].values

# Sort the indices by the FIFA score (highest first)
sorted_indices_cf = sorted(range(len(values_cf)), key=lambda k: values_cf[k], reverse=True)

# Sort values, labels, position ranks, and games played according to sorted indices
sorted_values_cf = values_cf[sorted_indices_cf]
sorted_labels_cf = labels_cf.iloc[sorted_indices_cf].values
sorted_pos_rank_cf = pos_rank_cf[sorted_indices_cf]
sorted_games_cf = games_value_cf[sorted_indices_cf]


norm_cf = plt.Normalize(min(cfs), max(cfs))
colors_cf = [cmap(norm_cf(value)) for value in sorted_pos_rank_cf]

# Adjust the annotations to be slightly lower for each subsequent label
# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_cf)):
    # Full annotation text
    full_text = f"{sorted_labels_cf[i]}"

    # Base annotation for the full text
    ax.annotate(
        full_text,
        (x_cf[i], y_cf[i]),
        color=colors_cf[i],
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )

    # Calculate the position and dimensions of the sorted_values_cf part
    label_text = f"{sorted_labels_cf[i]} "
    value_text = f"{sorted_values_cf[i]:.0f}"
    additional_text = f" {sorted_games_cf[i]:.0f}"

    # Temporarily annotate the label part to measure its width
    text_before = ax.annotate(
        label_text,
        (x_cf[i], y_cf[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(0, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_before = text_before.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_before.remove()
    x_offset_games = offset + 60 

    # Temporarily annotate the value part to measure its width
    text_value = ax.annotate(
        value_text,
        (x_cf[i], y_cf[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(bbox_before.width / 2, -offset * i),
        textcoords='offset points',
        font='futura'
    )
    plt.draw()
    bbox_value = text_value.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_value.remove()

    # Calculate the x position offset for the value part
    x_offset_value = (bbox_before.width -1) / 2

    # Annotate the value part with a filled circle around it
    ax.annotate(
        value_text,
        (x_cf[i], y_cf[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(x_offset_value, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="circle,pad=0.3", edgecolor=colors_cf[i], facecolor=colors_cf[i], alpha=0.9)
    )
    x_offset_value = (bbox_before.width + 30) / 2
    ax.annotate(
        additional_text,
        (x_cf[i], y_cf[i]),
        fontsize=20,
        ha='center',
        va='bottom',
        xytext=(x_offset_value + bbox_value.width +0.8 / 2, -offset * i),
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="round4,pad=0.02", edgecolor='white', facecolor='white', alpha=0.8)
    )


####### GOALKEEPER ################################################################################################################################################################
# Example data for plotting
x_gk = [2] * len(top_6_gk)  # Same x coordinates for all points
y_gk = [50] * len(top_6_gk)  # Same y coordinates for all points
labels_gk = top_6_gk['player_name']
values_gk = top_6_gk['FIFA_score'].values
pos_rank_gk = top_6_gk['position_rank'].values
games_value_gk = top_6_gk['player_season_90s_played'].values

# Sort the indices by the FIFA score (highest first)
sorted_indices_gk = sorted(range(len(values_gk)), key=lambda k: values_gk[k], reverse=True)

# Sort values, labels, position ranks, and games played according to sorted indices
sorted_values_gk = values_gk[sorted_indices_gk]
sorted_labels_gk = labels_gk.iloc[sorted_indices_gk].values
sorted_pos_rank_gk = pos_rank_gk[sorted_indices_gk]
sorted_games_gk = games_value_gk[sorted_indices_gk]
norm_gk = plt.Normalize(min(gks), max(gks))
colors_gk = [cmap(norm_gk(value)) for value in sorted_pos_rank_gk]
for i in range(len(sorted_indices_gk)):
    offset = 30

    # Full annotation text
    full_text = f"{sorted_labels_gk[i]} {sorted_values_gk[i]:.3f} {sorted_games_gk[i]:.3f}"
    full_text2 = f"{sorted_labels_gk[i]}"

    # Base annotation for the full text
    ax.annotate(
        full_text2,
        (x_gk[i], y_gk[i]),
        color=colors_gk[i],
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(offset * i, 0),  # Apply the offset horizontally
        textcoords='offset points',
        font='futura',
        rotation=-90  # Rotate 90 degrees counterclockwise to mirror
    )

    # Calculate the position and dimensions of the sorted_values_gk part
    label_text = f"{sorted_labels_gk[i]} "
    value_text = f"{sorted_values_gk[i]:.0f}"
    additional_text = f" {sorted_games_gk[i]:.0f}"

    # Temporarily annotate the label part to measure its width
    text_before = ax.annotate(
        label_text,
        (x_gk[i], y_gk[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(offset * i, 0),  # Apply the offset horizontally
        textcoords='offset points',
        font='futura',
        rotation=-90  # Rotate 90 degrees counterclockwise to mirror
    )
    plt.draw()
    bbox_before = text_before.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_before.remove()
    
    # Calculate the x position offset for the value part
    x_offset_value = bbox_before.width / 2

    # Temporarily annotate the value part to measure its width
    text_value = ax.annotate(
        value_text,
        (x_gk[i], y_gk[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(offset * i + x_offset_value, 0),  # Apply the offset horizontally
        textcoords='offset points',
        font='futura',
        rotation=-90  # Rotate 90 degrees counterclockwise to mirror
    )
    plt.draw()
    bbox_value = text_value.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    text_value.remove()

    # Calculate the x position offset for the value part
    x_offset_value = bbox_before.width / 2

    # Annotate the value part with a filled circle around it
    offset = offset + 15
    ax.annotate(
        value_text,
        (x_gk[i], y_gk[i]),
        fontsize=25,
        ha='center',
        va='bottom',
        xytext=(offset * i + x_offset_value - 20, -60),  # Apply the offset horizontally
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="circle,pad=0.3", edgecolor=colors_gk[i], facecolor=colors_gk[i], alpha=0.9),
        rotation=-90  # Rotate 90 degrees counterclockwise to mirror
    )

    # Adjust the offset for the additional text
    x_offset_additional = bbox_before.width + bbox_value.width 

    ax.annotate(
        additional_text,
        (x_gk[i], y_gk[i]),
        fontsize=20,
        ha='center',
        va='bottom',
        xytext=(offset * i + x_offset_additional-80, -110),  # Apply the offset horizontally
        textcoords='offset points',
        font='futura',
        bbox=dict(boxstyle="round4,pad=0.02", edgecolor='white', facecolor='white', alpha=0.8),
        rotation=-90  # Rotate 90 degrees counterclockwise to mirror
    )




################################################################################################################################################################


legend_text = "Column Order: Player Name, Game Quality, Games Played"
ax.text(0.5, 0.98, legend_text, fontsize=25, color='lightgrey', ha='center', va='center', transform=ax.transAxes, font='futura', fontweight='bold')


# Reverse the colormap
reversed_cmap = cmap.reversed()

# Create a color bar legend for the gradient labeled "Per-Position Quality in League" in white
sm = plt.cm.ScalarMappable(cmap=reversed_cmap, norm=norm_gk)
sm.set_array([])

# Add the color bar to the plot at the bottom, horizontally
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.00000000000001, fraction=0.01352, aspect=100)
cbar.set_label('Per-Position Quality in League', color='white')

# Set the color of the tick labels to white and hide the tick values
cbar.ax.xaxis.set_tick_params(color='white')

lab_dict = {'fontsize': 15}
#cbar.ax.set_xticklabels([0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100], color = 'white',fontdict =lab_dict)

# Set the label color to white
cbar.ax.set_xlabel('Per Position Quality in League', color='lightgrey', font='futura', fontsize=35)
logo_path = logo_paths[selected_team]

# Read the logo image
logo_img = mpimg.imread(logo_path)

# Create an OffsetImage and AnnotationBbox for the logo
imagebox = OffsetImage(logo_img, zoom=0.7)  # Adjust zoom as necessary
ab = AnnotationBbox(imagebox, (60, 5), frameon=False) 
# Annotate outside the plot, just above its edges
ax.annotate('PSL 23/24',
            xy=(-1, -2),  # Original position in the plot  # Adjusting the text position above the plot
            ha='center', va='bottom',
            fontsize=40, color='white',
            fontname='futura',
            arrowprops=dict(facecolor='black', shrink=0.05))
# Add the logo to the plot
ax.add_artist(ab)

#### Add positional Labels
ax.annotate("LB", (33, 7), font="futura", fontsize=30, color='white')
ax.annotate("RB", (33, 67), font="futura", fontsize=30, color='white')
ax.annotate("CB", (20, 36), font="futura", fontsize=30, color='white')
ax.annotate("DM", (42, 36), font="futura", fontsize=30, color='white')
ax.annotate("M", (70, 36), font="futura", fontsize=30, color='white')
ax.annotate("LW", (98, 7), font="futura", fontsize=30, color='white')
ax.annotate("RW", (98, 67), font="futura", fontsize=30, color='white')
ax.annotate("CF", (98, 36), font="futura", fontsize=30, color='white')
ax.annotate("Gk", (6, 47), font="futura", fontsize=30, color='white')
st.pyplot(fig)