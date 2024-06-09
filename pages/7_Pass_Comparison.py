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
    # heatmap = heatmap_doku_start
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
    
    # Plot the heatmap
    ax.imshow(np.flipud(heatmap_rescaled), extent=(0, pitch_length, 0, pitch_width), interpolation='nearest', cmap='magma', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('Pitch Length')
    ax.set_ylabel('Pitch Width')
    return fig 

def plot_passes_on_pitch(data):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Arc, Rectangle

    # Filter out dribbles and successful dribbles
    dribbles = data[(data['type'] == 'Pass')]

    # Define the soccer pitch dimensions (105x68 meters)
    pitch_length = 104
    pitch_width = 68

    # Create a grid for heatmap
    grid_size = 5
    x_bins = int(pitch_length / grid_size)
    y_bins = int(pitch_width / grid_size)

    # Create a heatmap for the end points of successful dribbles
    heatmap, xedges, yedges = np.histogram2d(dribbles['pass_end_x'], dribbles['pass_end_y'], bins=[x_bins, y_bins])

    # Find the cell with the highest number of dribbles
    max_dribbles_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    max_dribbles_zone = (xedges[max_dribbles_index[0]], yedges[max_dribbles_index[1]])

    # Filter dribbles that ended in the zone with the highest number of dribbles
    zone_dribbles = dribbles[(dribbles['pass_end_x'] >= max_dribbles_zone[0]) & (dribbles['pass_end_x'] < max_dribbles_zone[0] + grid_size) &
                             (dribbles['pass_end_y'] >= max_dribbles_zone[1]) & (dribbles['pass_end_y'] < max_dribbles_zone[1] + grid_size)]

    # Plotting these dribbles on the soccer pitch
    fig, ax = plt.subplots(figsize=(10, 6.8))

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

    # Plot the dribbles
    for index, row in zone_dribbles.iterrows():
        plt.plot([row['start_x'], row['pass_end_x']], [row['start_y'], row['pass_end_y']], color='blue')
        plt.scatter(row['pass_end_x'], row['pass_end_y'], color='red')

    # Highlight the zone with the most dribbles
    rect = Rectangle((max_dribbles_zone[0], max_dribbles_zone[1]), grid_size, grid_size, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.title('Passes Ending in the Most Active Zone')
    plt.xlabel('Length of the pitch (meters)')
    plt.ylabel('Width of the pitch (meters)')
    return fig

def topXpasses(country, league, season, x):
    '''takes in country name, league name, and season(string) name, 
    and X to output the top X pass players in the league'''

    filtered_comps = sb.competitions(creds=creds)
    league = filtered_comps[(filtered_comps['country_name']==country) & (filtered_comps['competition_name']==league) &
                (filtered_comps['season_name']==season)]
    comp_id = league['competition_id'].iloc[0]
    season_id = league['season_id'].iloc[0]
    player_season = sb.player_season_stats(competition_id=comp_id, season_id=season_id,creds=creds)

    names_obv = player_season[player_season['player_season_minutes']>600].sort_values('player_season_obv_pass_90', ascending=False)[["player_name",'player_season_obv_pass_90']].reset_index(drop=True)[['player_name','player_season_obv_pass_90']]
    return names_obv.iloc[0:x,:]

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


def comparePassHeats(country, league, season, x, event_data, comp_start, comp_end):
   
    '''x is the number of top Pass players to compare to the target player;
    event_data is the event_data of the players;
    comp_start and comp_end accept output, of Doku in this case'''

    ns = topXpasses(country, league, season, x)
    data = {
    'country': [country] * len(ns),
    'league': [league] * len(ns),
    'season': [season] * len(ns),
    'player_names': ns['player_name'],
    'Pass': ns['player_season_obv_pass_90']
    }
    df = pd.DataFrame(data)
    # filter out non-passes
    event_data = event_data[event_data['player'].isin(df['player_names'].values)]
    event_data = event_data[event_data['type']=='Pass']
    subset_event_d = event_data[["player","start_x","start_y","pass_end_x","pass_end_y"]]
    names = subset_event_d["player"].unique()

    # create empty dataframe to store the dribbling intensities of each player
    start_intensity_df = pd.DataFrame()
    end_intensity_df = pd.DataFrame()

    for current_name in names:
        temp = subset_event_d[subset_event_d['player'] == current_name]
        heatmap_start = calculate_heatmap_frequencies(temp, 16, 11)
        flat_start = heatmap_start.flatten()
        
        # Adjust data for end points
        data_end = temp.assign(start_x=temp['pass_end_x'], start_y=temp['pass_end_y'])
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
            sorted_distances_start[player_name] = [sorted_distances_start[player_name], row['Pass']]
    
    for index, row in df.iterrows():
        player_name = row['player_names']
        if player_name in sorted_distances_end:
            sorted_distances_end[player_name] = [sorted_distances_end[player_name], row['Pass']]

    #out = [sorted_distances_start, sorted_distances_end]
    
    df_start = pd.DataFrame.from_dict(sorted_distances_start)
    df_start_new = pd.DataFrame()
    df_end = pd.DataFrame.from_dict(sorted_distances_end)
    df_end_new = pd.DataFrame()

    df_start_new['PlayerName'] = df_start.columns.values
    df_start_new['StartSimilarity'] = df_start.iloc[0,:].values
    df_start_new['Pass'] = df_start.iloc[1,:].values

    df_end_new['PlayerName'] = df_end.columns.values
    df_end_new['EndSimilarity'] = df_end.iloc[0,:].values
    df_end_new['Pass'] = df_end.iloc[1,:].values

    out = df_start_new.merge(df_end_new, on='PlayerName', suffixes=('', '_end')).drop(['Pass_end'],axis=1)
    out = out.sort_values(by='Pass', ascending=False)

    return out



st.title('Player Comparison')

data_folder = 'data/leagues'

# List all files in the data folder
files = os.listdir(data_folder)

# Streamlit selectbox
st.subheader('Choose Base Player')
selected_file = st.selectbox("Select a League", files)
df = pd.read_csv(os.path.join(data_folder, selected_file), low_memory=False)
# df = pd.read_csv('data/England_Premier League_2324.csv', low_memory=False)
# Selecting the players
player_choices = df['player'].unique()
selected_player = st.selectbox("Select a Player", player_choices)

# Filter the original DataFrame based on both league and player
# selected_player = 'Jeremy Doku'
data_doku = df[(df['player'] == selected_player) & (df['type']=='Pass')]

grid_size_x = 16
grid_size_y = 11
# Calculate heatmap frequencies
heatmap_doku_start = calculate_heatmap_frequencies(data_doku, grid_size_x, grid_size_y)

# Adjust data for end points
data_doku_end = data_doku.assign(start_x=data_doku['pass_end_x'], start_y=data_doku['pass_end_y'])

heatmap_doku_end = calculate_heatmap_frequencies(data_doku_end, grid_size_x, grid_size_y)
plt1 = plot_heatmap_with_pitch(heatmap_doku_start, '', grid_size_x, grid_size_y)
plt2 = plot_heatmap_with_pitch(heatmap_doku_end, '', grid_size_x, grid_size_y)

col1, col2 = st.columns(2)

# Display each plot in a separate column
with col1:
    st.header("Starting Positions of Passes")
    st.pyplot(plt1)

with col2:
    st.header("Ending Positions of Passes")
    st.pyplot(plt2)

data = data_doku

#plot dribbles on the pitch
plt3 = plot_passes_on_pitch(data)
st.pyplot(plt3)


def display_top_passes():
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



if st.button('Show Top Passes'):
    with st.spinner('Fetching data...'):
        ns_array = comparePassHeats(country, selected_l, season, 20, df_event, heatmap_doku_start, heatmap_doku_end)
        st.session_state['tops'] = ns_array
        st.session_state['data_loaded'] = True
display_top_passes()

if st.session_state['data_loaded']:
    st.subheader('Choose Player to Compare')
    Player2 = st.selectbox("Select a player", st.session_state['tops'])
    data_2 = df_event[ (df_event['player'] == Player2) & (df_event['type'] == 'Pass')]
    #data_2 = player_df_2[player_df_2.type_name =="dribble"]
    grid_size_x = 16
    grid_size_y = 11
    heatmap_start = calculate_heatmap_frequencies(data_2, grid_size_x, grid_size_y)
    data_end = data_2.assign(start_x=data_2['pass_end_x'], start_y=data_2['pass_end_y'])
    heatmap_end = calculate_heatmap_frequencies(data_end, grid_size_x, grid_size_y)
    plt1_1 = plot_heatmap_with_pitch(heatmap_start, '', grid_size_x, grid_size_y)
    plt1_2 = plot_heatmap_with_pitch(heatmap_end, '', grid_size_x, grid_size_y)
    colx, coly = st.columns(2)
    with colx:
        st.pyplot(plt1_1)
    with coly:
        st.pyplot(plt1_2)
    dat = data_2
    plt1_3 = plot_passes_on_pitch(dat)
    st.pyplot(plt1_3)


