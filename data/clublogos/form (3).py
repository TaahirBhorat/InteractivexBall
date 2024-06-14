###ASSUMING PLAYERS WHO COME ON STAY ON THE WHOLE GAME
import numpy as np
import pandas as pd
import json
from datetime import datetime

data = pd.read_csv('data/South Africa_PSL_2324.csv', low_memory=False)
for col in data['type'].unique():
    print(col)
data = data.sort_values(by=['match_id', 'index'])

def get_match_obv(player, match_data):
    '''input player name and match event data for specific game'''
    xg = match_data[(match_data['player']==player) & (match_data['type']=='Shot') & (match_data['shot_type']=='Open Play')]['shot_statsbomb_xg'].sum()
    shot_obv = match_data[(match_data['player']==player) & (match_data['type']=='Shot')]['obv_total_net'].sum()
    dc_obv = match_data[(match_data['player']==player) & (match_data['type'].isin(['Carry','Dribble']))]['obv_total_net'].sum()
    pass_obv = match_data[(match_data['player']==player) & (match_data['type'] == 'Pass')]['obv_total_net'].sum()
    da_obv = match_data[(match_data['player']==player) & (match_data['type'].isin(['Block', 'Clearance','Duel','Foul Committed','Interception']))]['obv_total_net'].sum()
    total = shot_obv + dc_obv + pass_obv + da_obv
    return [xg, shot_obv, dc_obv, da_obv, pass_obv, total]

#event_data = data.copy()

def get_form(player, event_data):
    # team plays for
    team = event_data[event_data['player']==player]['team'].unique()
    # matches
    matches = event_data[event_data['team'].isin(team)]['match_id'].unique()
    data_dict = {}

    for j in range(1, len(matches)+1):
        data_dict[f"gw{j}opponent"] = 0
        data_dict[f"xg{j}"] = 0
        data_dict[f"shot{j}"] = 0
        data_dict[f"dc{j}"] = 0
        data_dict[f"da{j}"] = 0
        data_dict[f"pass{j}"] = 0
        data_dict[f"total{j}"] = 0
        data_dict[f"g{j}on"] = 0
        data_dict[f"g{j}off"] = 0
        data_dict[f'res{j}'] = 0

    gw = 0
    
    for match in matches:
        gw = gw + 1
        # current match event data
        match_data = event_data[event_data['match_id']==match]
        
        # who won the game
        opponent = [match_data['team'].unique()[i] for i in range(2) if match_data['team'].unique()[i] != team[0]][0]
        gf = match_data[(match_data['shot_outcome']=='Goal') & (match_data['team'].isin(team))].shape[0]
        gc = match_data[((match_data['shot_outcome']=='Goal') | (match_data['type'] == 'Own Goal For')) & (match_data['team'] == opponent)].shape[0]
        if gf>gc:
            data_dict[f'res{gw}'] = 'W'
        elif gc>gf:
            data_dict[f'res{gw}'] = 'L'
        else:
            data_dict[f'res{gw}'] = 'D'
        
        # just for start11
        temp = match_data[(match_data['type']=='Starting XI') & (match_data['team'].isin(team))]['tactics'].values[0]
        temp_dict = eval(temp.replace("'", '"')) 
        lineup_info = temp_dict['lineup']
        starting11 = [player['player']['name'] for player in lineup_info[:11]]
        # if player starts the game
        if player in starting11:
            # if player finishes the game
            if player not in match_data[match_data['type']=='Substitution']['player'].values:
                data_dict[f'gw{gw}opponent'] = [match_data['team'].unique()[i] for i in range(2) if match_data['team'].unique()[i] != team[0]][0]
                data_dict[f"g{gw}on"] = 0
                data_dict[f"g{gw}off"] = 90
                data_dict[f"xg{gw}"],data_dict[f"shot{gw}"],data_dict[f"dc{gw}"], data_dict[f"da{gw}"],data_dict[f"pass{gw}"],data_dict[f"total{gw}"] = get_match_obv(player, match_data)
            
            # if player comes off
            else:
                data_dict[f'gw{gw}opponent'] = [match_data['team'].unique()[i] for i in range(2) if match_data['team'].unique()[i] != team[0]][0]
                data_dict[f"g{gw}on"] = 0
                
                # if comes off in first half; take the time as is
                if match_data[(match_data['type']=='Substitution') & ((match_data['player']==player))]['period'].values[0] == 1:
                    time_string = match_data[(match_data['type']=='Substitution') & ((match_data['player']==player))]['timestamp'].values[0]
                    time_object = datetime.strptime(time_string, '%H:%M:%S.%f')
                    total_minutes = time_object.hour * 60 + time_object.minute + time_object.second / 60
                    data_dict[f"g{gw}off"] = total_minutes
                    data_dict[f"xg{gw}"],data_dict[f"shot{gw}"],data_dict[f"dc{gw}"], data_dict[f"da{gw}"],data_dict[f"pass{gw}"],data_dict[f"total{gw}"] = get_match_obv(player, match_data)
                
                # else if he comes off in second half add 45 mins to the time
                else:
                    time_string = match_data[(match_data['type']=='Substitution') & ((match_data['player']==player))]['timestamp'].values[0]
                    time_object = datetime.strptime(time_string, '%H:%M:%S.%f')
                    total_minutes = (time_object.hour * 60 + time_object.minute + time_object.second / 60) + 45
                    data_dict[f"g{gw}off"] = total_minutes
                    data_dict[f"xg{gw}"],data_dict[f"shot{gw}"],data_dict[f"dc{gw}"], data_dict[f"da{gw}"],data_dict[f"pass{gw}"],data_dict[f"total{gw}"] = get_match_obv(player, match_data)

        # if player doesn't start the game but comes on
        elif player in match_data['substitution_replacement'].unique():
            # half he comes on
            half_on = match_data[(match_data['substitution_replacement']==player)]['period'].values[0]
            if half_on == 2:
                time_string = match_data[(match_data['substitution_replacement']==player)]['timestamp'].values[0]
                time_object = datetime.strptime(time_string, '%H:%M:%S.%f')
                total_minutes = time_object.hour * 60 + time_object.minute + time_object.second / 60 + 45
                data_dict[f"g{gw}on"] = total_minutes
                data_dict[f"g{gw}off"] = 90
                data_dict[f'gw{gw}opponent'] = [match_data['team'].unique()[i] for i in range(2) if match_data['team'].unique()[i] != team[0]][0]
                data_dict[f"xg{gw}"],data_dict[f"shot{gw}"],data_dict[f"dc{gw}"], data_dict[f"da{gw}"],data_dict[f"pass{gw}"],data_dict[f"total{gw}"] = get_match_obv(player, match_data)
            elif half_on==1:
                time_string = match_data[(match_data['substitution_replacement']==player)]['timestamp'].values[0]
                time_object = datetime.strptime(time_string, '%H:%M:%S.%f')
                total_minutes = time_object.hour * 60 + time_object.minute + time_object.second / 60
                data_dict[f"g{gw}on"] = total_minutes
                data_dict[f"g{gw}off"] = 90
                data_dict[f'gw{gw}opponent'] = [match_data['team'].unique()[i] for i in range(2) if match_data['team'].unique()[i] != team[0]][0]
                data_dict[f"xg{gw}"],data_dict[f"shot{gw}"],data_dict[f"dc{gw}"], data_dict[f"da{gw}"],data_dict[f"pass{gw}"],data_dict[f"total{gw}"] = get_match_obv(player, match_data)
        # else player doesnt play at all
        else:
            data_dict[f'gw{gw}opponent'] = [match_data['team'].unique()[i] for i in range(2) if match_data['team'].unique()[i] != team[0]][0]

    xg = [data_dict[f'xg{i}'] for i in range(1,gw+1)]
    shot_obv = [data_dict[f'shot{i}'] for i in range(1,gw+1)]
    da_obv = [data_dict[f'da{i}'] for i in range(1,gw+1)]
    dc_obv = [data_dict[f'dc{i}'] for i in range(1,gw+1)]
    pass_obv = [data_dict[f'pass{i}'] for i in range(1,gw+1)]
    total_obv = [data_dict[f'total{i}'] for i in range(1,gw+1)]
    opponents = [data_dict[f'gw{i}opponent'] for i in range(1,gw+1)]
    on = [data_dict[f'g{i}on'] for i in range(1,gw+1)]
    off = [data_dict[f'g{i}off'] for i in range(1,gw+1)]
    res = [data_dict[f'res{i}'] for i in range(1,gw+1)]

    df = pd.DataFrame({
        'xg':xg,
        'shot_obv': shot_obv,
        'da_obv': da_obv,
        'dc_obv': dc_obv,
        'pass_obv': pass_obv,
        'total_obv': total_obv,
        'opponent': opponents,
        'on': on,
        'off': off,
        'res':res
    })

    return df


dick = get_form('Iqraam Rayners', data)
