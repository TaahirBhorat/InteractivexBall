import streamlit as st
import plotly.express as px
import pandas as pd
logo_url = 'pics/output-onlinepngtools (1).png'
st.sidebar.image(logo_url)


# Display the image and text above the sidebar
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-top: -10px;'>
        <h2 style='margin-bottom: 0;'>xBall'/ h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Load your data
def main():
    st.title('PSL Strikers Analysis')

    df = pd.read_csv("data/PSlPlayers_STR (1).csv")

    # Filter for players with more than 400 minutes
    df_filtered = df[df['player_season_minutes'] > 400]

    # Calculate the difference between actual goals and expected goals
    df_filtered['Goals_minus_xG'] = df_filtered['player_season_goals_90'] - df_filtered['player_season_np_xg_90']

    # Create the Plotly scatter plot
    fig = px.scatter(df_filtered, x='player_season_np_xg_90', y='player_season_goals_90',
                     text='player_name', hover_data=['Goals_minus_xG', 'team_name'],
                     labels={'player_season_np_xg_90': 'NP xG/90', 'player_season_goals_90': 'Goals/90'},
                     title='NP xG/90 vs Goals/90')

    # Adding mean lines
    mean_x = df_filtered['player_season_np_xg_90'].mean()
    mean_y = df_filtered['player_season_goals_90'].mean()
    fig.add_hline(y=mean_y, line_dash="dash", line_color="green")
    fig.add_vline(x=mean_x, line_dash="dash", line_color="red")

    # Adjusting layout for readability
    fig.update_traces(textposition='top center')
    fig.update_layout(showlegend=False)

    # Display the plot
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()