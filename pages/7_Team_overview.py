import streamlit as st
import pandas as pd
import numpy as np
import mplsoccer
import matplotlib.pyplot as plt

st.title("PSL 23/24 Team overview")
data = pd.read_csv("data/leagues/fullseason (1).csv")
team_options = data["team_name"].unique()
selected_team = st.selectbox('Select a team to filter by:', team_options)


max_games = 20
selected_games = st.selectbox('Select the number of games to filter by:', list(range(1, 21)))

import pandas as pd
# Define the combined function
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
        'Left Wing Back': 'LB',
        'Right Wing Back': 'RB',
        'Right Forward': 'RW',
        'Left Forward': 'LW',
        'Centre Forward': 'CF',
        'Left Centre Midfielder': 'M',
        'Left Centre Forward': 'CF',
        'Right Centre Forward': 'CF',
        'Right Midfielder': 'RW',
        'Right Centre Midfielder': 'M',
        'Left Attacking Midfielder': 'M',
        'Right Attacking Midfielder': 'M'
    }

    # Map the primary_position to position
    df['position'] = df['primary_position'].map(position_map)

    # Handle any NaN values (if any remain)

    # Filter out players based on min_90s_played
    df_filtered = df[df['player_season_90s_played'] >= min_90s_played].copy()

    # Create the position_rank column based on player_season_obv_90 within each position, except for CF
    df_filtered['position_rank'] = df_filtered.groupby('position').apply(
        lambda x: x['player_season_np_xg_90'].rank(ascending=False, method='min') if x.name == 'CF'
        else x['player_season_obv_90'].rank(ascending=False, method='min')
    ).reset_index(level=0, drop=True)

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
offset = 35

####### CENTERBACKS ################################################################################################################################################################
x_cb = [25] * len(top_6_cb)  # Same x coordinates for all points
y_cb = [40] * len(top_6_cb)  # Same y coordinates for all points
labels_cb = top_6_cb['player_name']
values_cb = top_6_cb['player_season_obv_90'].values
pos_rank_cb = top_6_cb['position_rank'].values
games_value_cb = top_6_cb['player_season_90s_played'].values

# Sort the indices by the games played (highest first)
sorted_indices_cb = sorted(range(len(games_value_cb)), key=lambda k: games_value_cb[k], reverse=True)
sorted_additional_value_cb = games_value_cb[sorted_indices_cb]

# Sort labels and position ranks according to sorted indices
sorted_labels_cb = labels_cb.iloc[sorted_indices_cb].values
sorted_pos_rank_cb = pos_rank_cb[sorted_indices_cb]
sorted_values_cb = values_cb[sorted_indices_cb]

norm_cb = plt.Normalize(min(cbs), max(cbs))
colors_cb = [cmap(norm_cb(value)) for value in sorted_pos_rank_cb]

# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_cb)):
    ax.annotate(f"{sorted_labels_cb[i]} {sorted_values_cb[i]:.3f} {sorted_additional_value_cb[i]:.3f}", (x_cb[i], y_cb[i]), color=colors_cb[i], fontsize=25, ha='center', va='bottom', 
                xytext=(0, -offset * i), textcoords='offset points', font='futura')


####### LEFT BACKS ################################################################################################################################################################
x_lb = [34] * len(top_6_lb)  # Same x coordinates for all points
y_lb = [10] * len(top_6_lb)
labels_lb = top_6_lb['player_name']
values_lb = top_6_lb['player_season_obv_90'].values
pos_rank_lb = top_6_lb['position_rank'].values
games_value_lb = top_6_lb['player_season_90s_played'].values

# Sort the indices by the games played (highest first)
sorted_indices_lb = sorted(range(len(games_value_lb)), key=lambda k: games_value_lb[k], reverse=True)
sorted_additional_value_lb = games_value_lb[sorted_indices_lb]

# Sort labels and position ranks according to sorted indices
sorted_labels_lb = labels_lb.iloc[sorted_indices_lb].values
sorted_pos_rank_lb = pos_rank_lb[sorted_indices_lb]
sorted_values_lb = values_lb[sorted_indices_lb]

norm_lb = plt.Normalize(min(lbs), max(lbs))
colors_lb = [cmap(norm_lb(value)) for value in sorted_pos_rank_lb]

# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_lb)):
    ax.annotate(f"{sorted_labels_lb[i]} {sorted_values_lb[i]:.3f} {sorted_additional_value_lb[i]:.3f}", (x_lb[i], y_lb[i]), color=colors_lb[i], fontsize=25, ha='center', va='bottom', 
                xytext=(0, -offset * i), textcoords='offset points', font='futura')


####### RIGHT BACKS ################################################################################################################################################################
x_rb = [34] * len(top_6_rb)  # Same x coordinates for all points
y_rb = [70] * len(top_6_rb)  # Same y coordinates for all points
labels_rb = top_6_rb['player_name']
values_rb = top_6_rb['player_season_obv_90'].values
pos_rank_rb = top_6_rb['position_rank'].values
games_value_rb = top_6_rb['player_season_90s_played'].values

# Sort the indices by the games played (highest first)
sorted_indices_rb = sorted(range(len(games_value_rb)), key=lambda k: games_value_rb[k], reverse=True)
sorted_additional_value_rb = games_value_rb[sorted_indices_rb]

# Sort labels and position ranks according to sorted indices
sorted_labels_rb = labels_rb.iloc[sorted_indices_rb].values
sorted_pos_rank_rb = pos_rank_rb[sorted_indices_rb]
sorted_values_rb = values_rb[sorted_indices_rb]

norm_rb = plt.Normalize(min(rbs), max(rbs))
colors_rb = [cmap(norm_rb(value)) for value in sorted_pos_rank_rb]

# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_rb)):
    ax.annotate(f"{sorted_labels_rb[i]} {sorted_values_rb[i]:.3f} {sorted_additional_value_rb[i]:.3f}", (x_rb[i], y_rb[i]), color=colors_rb[i], fontsize=25, ha='center', va='bottom', 
                xytext=(0, -offset * i), textcoords='offset points', font='futura')


####### MIDFIELDERS ################################################################################################################################################################
# Example data for plotting
x_m = [60] * len(top_9_m)  # Same x coordinates for all points
y_m = [40] * len(top_9_m)  # Same y coordinates for all points
labels_m = top_9_m['player_name']
values_m = top_9_m['player_season_obv_90'].values
pos_rank_m = top_9_m['position_rank'].values
games_value_m = top_9_m['player_season_90s_played'].values

# Sort the indices by the games played (highest first)
sorted_indices_m = sorted(range(len(games_value_m)), key=lambda k: games_value_m[k], reverse=True)
sorted_additional_value_m = games_value_m[sorted_indices_m]

# Sort labels and position ranks according to sorted indices
sorted_labels_m = labels_m.iloc[sorted_indices_m].values
sorted_pos_rank_m = pos_rank_m[sorted_indices_m]
sorted_values_m = values_m[sorted_indices_m]

norm_m = plt.Normalize(min(ms), max(ms))
colors_m = [cmap(norm_m(value)) for value in sorted_pos_rank_m]

# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_m)):
    ax.annotate(f"{sorted_labels_m[i]} {sorted_values_m[i]:.3f} {sorted_additional_value_m[i]:.3f}", (x_m[i], y_m[i]), color=colors_m[i], fontsize=25, ha='center', va='bottom', 
                xytext=(0, -offset * i), textcoords='offset points', font='futura')

####### LEFT WING ################################################################################################################################################################
# Example data for plotting
x_lw = [100] * len(top_6_lw)  # Same x coordinates for all points
y_lw = [10] * len(top_6_lw)  # Same y coordinates for all points
labels_lw = top_6_lw['player_name']
values_lw = top_6_lw['player_season_obv_90'].values
pos_rank_lw = top_6_lw['position_rank'].values
games_value_lw = top_6_lw['player_season_90s_played'].values

# Sort the indices by the games played (highest first)
sorted_indices_lw = sorted(range(len(games_value_lw)), key=lambda k: games_value_lw[k], reverse=True)
sorted_additional_value_lw = games_value_lw[sorted_indices_lw]

# Sort labels and position ranks according to sorted indices
sorted_labels_lw = labels_lw.iloc[sorted_indices_lw].values
sorted_pos_rank_lw = pos_rank_lw[sorted_indices_lw]
sorted_values_lw = values_lw[sorted_indices_lw]

norm_lw = plt.Normalize(min(lws), max(lws))
colors_lw = [cmap(norm_lw(value)) for value in sorted_pos_rank_lw]

# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_lw)):
    ax.annotate(f"{sorted_labels_lw[i]} {sorted_values_lw[i]:.3f} {sorted_additional_value_lw[i]:.3f}", (x_lw[i], y_lw[i]), color=colors_lw[i], fontsize=25, ha='center', va='bottom', 
                xytext=(0, -offset * i), textcoords='offset points', font='futura')

####### RIGHT WING ################################################################################################################################################################
# Example data for plotting
x_rw = [100] * len(top_6_rw)  # Same x coordinates for all points
y_rw = [70] * len(top_6_rw)  # Same y coordinates for all points
labels_rw = top_6_rw['player_name']
values_rw = top_6_rw['player_season_obv_90'].values
pos_rank_rw = top_6_rw['position_rank'].values
games_value_rw = top_6_rw['player_season_90s_played'].values

# Sort the indices by the games played (highest first)
sorted_indices_rw = sorted(range(len(games_value_rw)), key=lambda k: games_value_rw[k], reverse=True)
sorted_additional_value_rw = games_value_rw[sorted_indices_rw]

# Sort labels and position ranks according to sorted indices
sorted_labels_rw = labels_rw.iloc[sorted_indices_rw].values
sorted_pos_rank_rw = pos_rank_rw[sorted_indices_rw]
sorted_values_rw = values_rw[sorted_indices_rw]

norm_rw = plt.Normalize(min(rws), max(rws))
colors_rw = [cmap(norm_rw(value)) for value in sorted_pos_rank_rw]

# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_rw)):
    ax.annotate(f"{sorted_labels_rw[i]} {sorted_values_rw[i]:.3f} {sorted_additional_value_rw[i]:.3f}", (x_rw[i], y_rw[i]), color=colors_rw[i], fontsize=25, ha='center', va='bottom', 
                xytext=(0, -offset * i), textcoords='offset points', font='futura')


####### CENTER FORWARD ################################################################################################################################################################
# Example data for plotting
x_cf = [100] * len(top_6_cf)  # Same x coordinates for all points
y_cf = [40] * len(top_6_cf)  # Same y coordinates for all points
labels_cf = top_6_cf['player_name']
values_cf = top_6_cf['player_season_np_xg_90'].values
pos_rank_cf = top_6_cf['position_rank'].values
games_value_cf = top_6_cf['player_season_90s_played'].values

# Sort the indices by the games played (highest first)
sorted_indices_cf = sorted(range(len(games_value_cf)), key=lambda k: games_value_cf[k], reverse=True)
sorted_additional_value_cf = games_value_cf[sorted_indices_cf]

# Sort labels and position ranks according to sorted indices
sorted_labels_cf = labels_cf.iloc[sorted_indices_cf].values
sorted_pos_rank_cf = pos_rank_cf[sorted_indices_cf]
sorted_values_cf = values_cf[sorted_indices_cf]

norm_cf = plt.Normalize(min(cfs), max(cfs))
colors_cf = [cmap(norm_cf(value)) for value in sorted_pos_rank_cf]

# Adjust the annotations to be slightly lower for each subsequent label
for i in range(len(sorted_indices_cf)):
    ax.annotate(f"{sorted_labels_cf[i]} {sorted_values_cf[i]:.3f} {sorted_additional_value_cf[i]:.3f}", (x_cf[i], y_cf[i]), color=colors_cf[i], fontsize=25, ha='center', va='bottom', 
                xytext=(0, -offset * i), textcoords='offset points', font='futura')

####### GOALKEEPER ################################################################################################################################################################
# Example data for plotting
x_gk = [2] * len(top_6_gk)  # Same x coordinates for all points
y_gk = [50] * len(top_6_gk)  # Same y coordinates for all points
labels_gk = top_6_gk['player_name']
values_gk = top_6_gk['player_season_obv_90'].values
pos_rank_gk = top_6_gk['position_rank'].values
games_value_gk = top_6_gk['player_season_90s_played'].values

# Sort the indices by the games played (highest first)
sorted_indices_gk = sorted(range(len(games_value_gk)), key=lambda k: games_value_gk[k], reverse=True)
sorted_additional_value_gk = games_value_gk[sorted_indices_gk]

# Sort labels and position ranks according to sorted indices
sorted_labels_gk = labels_gk.iloc[sorted_indices_gk].values
sorted_pos_rank_gk = pos_rank_gk[sorted_indices_gk]
sorted_values_gk = values_gk[sorted_indices_gk]
norm_gk = plt.Normalize(min(gks), max(gks))
colors_gk = [cmap(norm_gk(value)) for value in sorted_pos_rank_gk]
for i in range(len(sorted_indices_gk)):
    offset = 30
    ax.annotate(
        f"{sorted_labels_gk[i]} {sorted_values_gk[i]:.3f} {sorted_additional_value_gk[i]:.3f}", 
        (x_gk[i], y_gk[i]), 
        color=colors_gk[i], 
        fontsize=25, 
        ha='center', 
        va='bottom', 
        xytext=(offset * i, 0),  # Apply the offset vertically in the mirrored state
        textcoords='offset points', 
        font='futura', 
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
cbar.ax.set_xticklabels([])

# Set the label color to white
cbar.ax.set_xlabel('Per Position Quality in League', color='lightgrey', font='futura', fontsize=25)
logo_path = logo_paths[selected_team]

# Read the logo image
logo_img = mpimg.imread(logo_path)

# Create an OffsetImage and AnnotationBbox for the logo
imagebox = OffsetImage(logo_img, zoom=0.7)  # Adjust zoom as necessary
ab = AnnotationBbox(imagebox, (60, 5), frameon=False) 
# Annotate outside the plot, just above its edges
ax.annotate('PSL 23/34',
            xy=(-1, -2),  # Original position in the plot  # Adjusting the text position above the plot
            ha='center', va='bottom',
            fontsize=40, color='white',
            fontname='Futura',
            arrowprops=dict(facecolor='black', shrink=0.05))
# Add the logo to the plot
ax.add_artist(ab)

#### Add positional Labels
ax.annotate("LB", (33, 8), font="futura", fontsize=25, color='lightgrey')
ax.annotate("RB", (33, 68), font="futura", fontsize=25, color='lightgrey')
ax.annotate("CB", (25, 37), font="futura", fontsize=25, color='lightgrey')
ax.annotate("M", (59, 38), font="futura", fontsize=25, color='lightgrey')
ax.annotate("LW", (98, 8), font="futura", fontsize=25, color='lightgrey')
ax.annotate("RW", (98, 68), font="futura", fontsize=25, color='lightgrey')
ax.annotate("CF", (98, 38), font="futura", fontsize=25, color='lightgrey')
ax.annotate("Gk", (5, 38), font="futura", fontsize=25, color='lightgrey')
st.pyplot(fig)