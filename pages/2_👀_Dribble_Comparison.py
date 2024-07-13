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

# Calculate heatmap function
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

def plot_dribbles_on_pitch(data, min_dribble_length):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    # Filter out dribbles and successful dribbles
    dribbles = data[(data['type'] == 'Carry')]
     # Calculate dribble lengths
    dribbles['dribble_length'] = np.sqrt((dribbles['carry_end_x'] - dribbles['start_x'])**2 + (dribbles['carry_end_y'] - dribbles['start_y'])**2)

    # Filter dribbles based on minimum dribble length
    dribbles = dribbles[dribbles['dribble_length'] >= min_dribble_length]

    # Define the soccer pitch dimensions (105x68 meters)
    pitch_length = 104
    pitch_width = 68

    # Increase the grid size
    grid_size = 10
    x_bins = int(pitch_length / grid_size)
    y_bins = int(pitch_width / grid_size)

    # Create a heatmap for the end points of successful dribbles
    heatmap, xedges, yedges = np.histogram2d(dribbles['carry_end_x'], dribbles['carry_end_y'], bins=[x_bins, y_bins])

    # Find the indices of the top 3 cells with the highest number of dribbles
    top_3_indices = np.dstack(np.unravel_index(np.argsort(heatmap.ravel())[-3:], heatmap.shape))[0]

    # Plotting these dribbles on the soccer pitch
    fig, ax = plt.subplots(figsize=(10, 6.8))
    fig.patch.set_facecolor('black')
    
    ax.set_facecolor('#383434')
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Highlight the top 3 zones and plot dribbles
    colors = ['red', 'green', 'blue']
    for idx, (x_idx, y_idx) in enumerate(top_3_indices):
        zone_x = xedges[x_idx]
        zone_y = yedges[y_idx]

        # Ensure the box does not go over the edge of the pitch
        if zone_x + grid_size > pitch_length:
            zone_x = pitch_length - grid_size
        if zone_y + grid_size > pitch_width:
            zone_y = pitch_width - grid_size

        # Filter dribbles that ended in the current top zone
        zone_dribbles = dribbles[(dribbles['carry_end_x'] >= zone_x) & (dribbles['carry_end_x'] < zone_x + grid_size) &
                                 (dribbles['carry_end_y'] >= zone_y) & (dribbles['carry_end_y'] < zone_y + grid_size)]

        # Plot the dribbles in the current top zone
        for _, row in zone_dribbles.iterrows():
            #plt.plot([row['start_x'], row['carry_end_x']], [row['start_y'], row['carry_end_y']], color=colors[idx])
            plt.arrow(row['start_x'], row['start_y'], row['carry_end_x'] - row['start_x'], row['carry_end_y'] - row['start_y'], color=colors[idx], head_width=1, head_length=1, length_includes_head=True)
            plt.scatter(row['carry_end_x'], row['carry_end_y'], color=colors[idx], alpha=0.4)

        # Highlight the current top zone
        rect = Rectangle((zone_x, zone_y), grid_size, grid_size, linewidth=1, edgecolor=colors[idx], facecolor='none')
        ax.add_patch(rect)

    plt.title('Dribbles Ending in the Top 3 Most Active Zones')
    plt.xlabel('Length of the pitch (meters)')
    plt.ylabel('Width of the pitch (meters)')
    return fig


def topXdribblers(country, league, season, x):
    '''takes in country name, league name, and season(string) name, 
    and X to output the top X d&c players in the league'''    
    filtered_comps = sb.competitions(creds=creds)
    league = filtered_comps[(filtered_comps['country_name']==country) & (filtered_comps['competition_name']==league) &
                (filtered_comps['season_name']==season)]
    comp_id = league['competition_id'].iloc[0]
    season_id = league['season_id'].iloc[0]
    player_season = sb.player_season_stats(competition_id=comp_id, season_id=season_id,creds=creds)[["player_name",'player_season_obv_dribble_carry_90','player_season_minutes','primary_position']]
    player_season['primary_position'] = player_season['primary_position'].map(position_map).fillna('none')
    player_season['primary_position'] = player_season['primary_position'].map(second_position_map).fillna('none')    
    player_season = player_season[player_season['primary_position']==player_position]
    names_obv = player_season[player_season['player_season_minutes']>600].sort_values('player_season_obv_dribble_carry_90', ascending=False)[["player_name",'player_season_obv_dribble_carry_90']].reset_index(drop=True)[['player_name','player_season_obv_dribble_carry_90']]
    
    return names_obv.iloc[0:x,:]

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

second_position_map = {
    'RB': 'FB',
    'LB': 'FB',
    'LW': 'W',
    'RW': 'W',
    'DM': 'DM',
    'CB': 'CB',
    'M': 'M',
    'CF': 'CF',
    'GK': 'GK'
}


    # Map the primary_position to position


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


def compareDribbleHeats(country, league, season, x, event_data, comp_start, comp_end):
   
    '''x is the number of top d&c players to compare to the target player;
    event_data is the event_data of the players;
    comp_start and comp_end accept ⁠⁠ output, of Doku in this case'''

    ns = topXdribblers(country, league, season, x)
    data = {
    'country': [country] * len(ns),
    'league': [league] * len(ns),
    'season': [season] * len(ns),
    'player_names': ns['player_name'],
    'D&C': ns['player_season_obv_dribble_carry_90']
    }
    df = pd.DataFrame(data)
    # filter out non-dribblers
    event_data = event_data[event_data['player'].isin(df['player_names'].values)]
    event_data = event_data[event_data['type']=='Carry']
    subset_event_d = event_data[["player","start_x","start_y","carry_end_x","carry_end_y"]]
    names = subset_event_d["player"].unique()

    # create empty dataframe to store the dribbling intensities of each player
    start_intensity_df = pd.DataFrame()
    end_intensity_df = pd.DataFrame()

    for current_name in names:
        temp = subset_event_d[subset_event_d['player'] == current_name]
        heatmap_start = calculate_heatmap_frequencies(temp, 16, 11)
        flat_start = heatmap_start.flatten()
        
        # Adjust data for end points
        data_end = temp.assign(start_x=temp['carry_end_x'], start_y=temp['carry_end_y'])
        heatmap_end = calculate_heatmap_frequencies(data_end, 16, 11)
        flat_end = heatmap_end.flatten()

        # Append intensity scores to DataFrames
        start_intensity_df[current_name] = flat_start
        end_intensity_df[current_name ] = flat_end
    
    # flatten target player's intensities
    comp_start = comp_start.flatten()
    comp_end = comp_end.flatten()

    # calculate euc distance between potential players and PLAYER
    distances = {}
    for column in start_intensity_df.columns:
        intensity_vector = start_intensity_df[column].values  # Extract intensity values 
        distance = np.linalg.norm(intensity_vector - comp_start)
        distances[column] = distance

    # distances dictionary contains column names from start_intensity_df as keys and their Euclidean distances as values
    sorted_distances_start = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    
    distances = {}
    for column in end_intensity_df.columns:
        intensity_vector = end_intensity_df[column].values  # Extract intensity values 
        distance = np.linalg.norm(intensity_vector - comp_end)
        distances[column] = distance

    # distances dictionary contains column names from start_intensity_df as keys and their Euclidean distances as values
    sorted_distances_end = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    
    for index, row in df.iterrows():
        player_name = row['player_names']
        if player_name in sorted_distances_start:
            sorted_distances_start[player_name] = [sorted_distances_start[player_name], row['D&C']]
    
    for index, row in df.iterrows():
        player_name = row['player_names']
        if player_name in sorted_distances_end:
            sorted_distances_end[player_name] = [sorted_distances_end[player_name], row['D&C']]

    #out = [sorted_distances_start, sorted_distances_end]
    
    df_start = pd.DataFrame.from_dict(sorted_distances_start)
    df_start_new = pd.DataFrame()
    df_end = pd.DataFrame.from_dict(sorted_distances_end)
    df_end_new = pd.DataFrame()

    df_start_new['PlayerName'] = df_start.columns.values
    df_start_new['StartSimilarity'] = df_start.iloc[0,:].values
    df_start_new['D&C'] = df_start.iloc[1,:].values

    df_end_new['PlayerName'] = df_end.columns.values
    df_end_new['EndSimilarity'] = df_end.iloc[0,:].values
    df_end_new['D&C'] = df_end.iloc[1,:].values

    out = df_start_new.merge(df_end_new, on='PlayerName', suffixes=('', '_end')).drop(['D&C_end'],axis=1)
    out = out.sort_values(by='D&C', ascending=False)

    return out


def extract_file_info(filename):
    # Remove the .csv extension and split the filename by underscores
    parts = filename.replace('.csv', '').split('_')
    
    # Extract the country and league name
    country = parts[0]
    leaguename = ' '.join(parts[1:-1])
    
    # Extract and format the season
    season = parts[-1]
    if len(season) == 2:
        season = f"20{season}"
    elif len(season) == 4:
        season = f"20{season[:2]}/20{season[2:]}"
    
    return country, leaguename, season



def getPosition(country, league, season, selected_player):
    filtered_comps = sb.competitions(creds=creds)
    league = filtered_comps[(filtered_comps['country_name']==country) & (filtered_comps['competition_name']==league) & (filtered_comps['season_name']==season)].copy()
    comp_id = league['competition_id'].iloc[0]
    season_id = league['season_id'].iloc[0]
    player_season = sb.player_season_stats(competition_id=comp_id, season_id=season_id,creds=creds)[['player_id','team_name','player_name','primary_position','player_season_90s_played']]
    # for players who moved, we take the club they played the most games for
    player_season = player_season.sort_values('player_season_90s_played', ascending=False)
    player_season = player_season.groupby('player_id').agg({
    'team_name': 'first', # choose team he played the most games for
    'player_name': 'first',
    'primary_position': 'first',
    'player_season_90s_played': 'sum' # but some over both teams
    }).reset_index()

    # Sorting by 'player_season_90s_played' and dropping duplicates (if any, this step might be redundant here)
    player_season = player_season.drop_duplicates('player_id')
    

    player_season = player_season[player_season['player_name']==selected_player]
    player_season['primary_position'] = player_season['primary_position'].map(position_map).fillna('none')
    player_season['primary_position'] = player_season['primary_position'].map(second_position_map).fillna('none')
    return player_season['primary_position'].values[0]




st.title('Player Comparison')

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

country, league, season = extract_file_info(selected_file)
player_position = getPosition(country, league, season, selected_player)

# Filter the original DataFrame based on both league and player
# selected_player = 'Jeremy Doku'
data_doku = df[(df['player'] == selected_player) & (df['type']=='Carry')]

grid_size_x = 16
grid_size_y = 11
# Calculate heatmap frequencies
heatmap_doku_start = calculate_heatmap_frequencies(data_doku, grid_size_x, grid_size_y)

# Adjust data for end points
data_doku_end = data_doku.assign(start_x=data_doku['carry_end_x'], start_y=data_doku['carry_end_y'])

heatmap_doku_end = calculate_heatmap_frequencies(data_doku_end, grid_size_x, grid_size_y)
plt1 = plot_heatmap_with_pitch(heatmap_doku_start, '', grid_size_x, grid_size_y)
plt2 = plot_heatmap_with_pitch(heatmap_doku_end, '', grid_size_x, grid_size_y)

col1, col2 = st.columns(2)

# Display each plot in a separate column
with col1:
    st.header("Starting Positions of Dribbles")
    st.pyplot(plt1)

with col2:
    st.header("Ending Positions of Dribbles")
    st.pyplot(plt2)




data = data_doku
min_length1 = st.slider('Minimum Dribble Length', min_value=0, max_value=100, key=1)
#plot dribbles on the pitch
plt3 = plot_dribbles_on_pitch(data, min_dribble_length=min_length1)
st.pyplot(plt3)



def display_top_dribblers():
    if 'tops' in st.session_state and st.session_state['data_loaded']:
        st.write(pd.DataFrame(st.session_state['tops']))


# Seccond Player
creds = {"user":"daylesolomon@gmail.com", "passwd": "qIRf28g8"}
all_sb_leagues = sb.competitions(creds=creds)
# Apply the function across each row of the DataFrame
mask = all_sb_leagues.apply(lambda row: match_to_file(row, file_names), axis=1)
# Filter the DataFrame based on the mask
all_sb_leagues = all_sb_leagues[mask]
leagues = all_sb_leagues['competition_name'].unique()
countries = all_sb_leagues['country_name'].unique()
seasons = all_sb_leagues['season_name'].unique()



st.subheader('Choose League to Compare')


selected_l = st.selectbox("Select a League", leagues)

all_sb_leagues2 = all_sb_leagues[all_sb_leagues["competition_name"]== selected_l]
countries = all_sb_leagues2['country_name'].unique()
country = st.selectbox("Select a Country", countries)


all_sb_leagues3 = all_sb_leagues2[all_sb_leagues2["country_name"]== country]
seasons = all_sb_leagues3['season_name'].unique()
season = st.selectbox("Select a Season", seasons)


    # Adjust season_name for year range format
if '/' in season:
    season2 = ''.join(season.split('/')[0][-2:] + season.split('/')[1][-2:])
else:  # Handle single year format
    season2= season[-2:]

    # Create a pattern to match
matching_event = f"{country}_{selected_l}_{season2}.csv"

data_folder = 'data/leagues'

# List all files in the data folder
files = os.listdir(data_folder)
df_event = pd.read_csv(os.path.join(data_folder, matching_event), low_memory=False)



if st.button('Show Top Dribblers'):
    with st.spinner('Fetching data...'):
        ns_array = compareDribbleHeats(country, selected_l, season, 40, df_event, heatmap_doku_start, heatmap_doku_end)
        st.session_state['tops'] = ns_array
        st.session_state['data_loaded'] = True
display_top_dribblers()

if st.session_state['data_loaded']:
    st.subheader('Choose Player to Compare')
    Player2 = st.selectbox("Select a player", st.session_state['tops'])
    data_2 = df_event[ (df_event['player'] == Player2) & (df_event['type'] == 'Carry')]
    #data_2 = player_df_2[player_df_2.type_name =="dribble"]
    grid_size_x = 16
    grid_size_y = 11
    heatmap_start = calculate_heatmap_frequencies(data_2, grid_size_x, grid_size_y)
    data_end = data_2.assign(start_x=data_2['carry_end_x'], start_y=data_2['carry_end_y'])
    heatmap_end = calculate_heatmap_frequencies(data_end, grid_size_x, grid_size_y)
    plt1_1 = plot_heatmap_with_pitch(heatmap_start, '', grid_size_x, grid_size_y)
    plt1_2 = plot_heatmap_with_pitch(heatmap_end, '', grid_size_x, grid_size_y)
    colx, coly = st.columns(2)
    with colx:
        st.header("Starting Positions of Dribbles")
        st.pyplot(plt1_1)
    with coly:
        st.header("Ending Positions of Dribbles")
        st.pyplot(plt1_2)
    dat = data_2
    min_length = st.slider('Minimum Dribble Length', min_value=0, max_value=100, key=2)
    plt1_3 = plot_dribbles_on_pitch(dat, min_dribble_length=min_length)
    st.pyplot(plt1_3)


