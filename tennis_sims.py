import os
os.chdir(r"C:\Users\agad4\OneDrive\Documents\python_work\tennis")
import random
import pulp
import joblib
import itertools
import re

import pandas as pd
import numpy as np

from scipy.stats import poisson, rankdata

def name_cleanup(projections):
    name_dict = {'Cori Gauff':'Coco Gauff','Alex De Minaur':'Alex de Minaur',
                 'Caty Mcnally':'Catherine McNally','Jan Lennard Struff':'Jan-Lennard Struff',
                 'Luca Van Assche':'Luca van Assche','Marc Andrea Huesler':'Marc-Andrea Huesler',
                 'Anna Lena Friedsam':'Anna-Lena Friedsam','Jodie Burrage':'Jodie Anna Burrage',
                 'Mackenzie Mcdonald':'Mackenzie McDonald',
                 'Botic Van De Zandschulp':'Botic Van de Zandschulp',
                 'Albert Ramos':'Albert Ramos-Vinolas','J J Wolf':'Jeffrey John Wolf',
                 'Leylah Fernandez':'Leylah Annie Fernandez',
                 'Irina Camelia Begu':'Irina-Camelia Begu',
                 'Anna Karolina Schmiedlova':'Anna-Karolina Schmiedlova',
                 'Alison Riske Amritraj':'Alison Riske-Amritraj',
                 'Jaqueline Cristian':'Jaqueline Adina Cristian',
                 'Tomas Barrios Vera':'Marcelo Tomas Barrios Vera',
                 'Christopher Oconnell':"Christopher O'Connell",
                 'Abedallah Shelbayh':'Abdullah Shelbayh',
                 #'Fanny Stollar':'Fanni Stollar',
                 'Sasi Kumar Mukund':'Sasikumar Mukund',
                 'Felix Auger Aliassime':'Felix Auger-Aliassime',
                 'Xin Yu Wang':'Xinyu Wang',
                 'Thai Son Kwiatkowski':'Thai-Son Kwiatkowski','Storm Sanders':'Storm Hunter',
                 'Viktoria Kuzmova':'Viktoria Hruncakova',
                 'Kathinka Von Deichmann':'Kathinka von Deichmann',
                 'En Shuo Liang':'En-Shuo Liang','Na Lae Han':'Na-Lae Han',
                 'Miriam Bulgaru':'Miriam Bianca Bulgaru','Giovanni Perricard':'Giovanni Mpetshi Perricard'}
    for i in name_dict:
        if 'name' in projections.columns:
            projections['name'].where(projections['name'] != i, name_dict[i], inplace=True)
        else:
            projections['name'].where(projections['name'] != i, name_dict[i], inplace=True)            
        
    return projections

def missing_players(proj_csv, dk_csv, own_csv):
    projections = name_cleanup(pd.read_csv(proj_csv))
    player_pool = pd.read_csv(dk_csv, encoding = "ISO-8859-1")
    
    [print(i) for i in player_pool['Name'].tolist() if i not in projections['Name'].tolist()]
    
    if own_csv:
        ownership = name_cleanup(pd.read_csv(own_csv))
        [print(i) for i in player_pool['Name'].tolist() if i not in ownership['Name'].tolist()]
        
def name_check(odds, aces, players, surface):
    for i in players:
        if i not in odds:
            print(f"{i} not in Pinnacle.")
        if (i, surface) not in aces:
            print(f"{i} not in Aces.")
        
def ace_fault_models(tour):
    tennis_logs = []
    for i in range(2013, 2024):
        logs = pd.read_csv(f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{tour}_matches_{i}.csv")
        tennis_logs.append(logs)
        
    tennis_logs = pd.concat(tennis_logs)

    winning_aces = tennis_logs[['winner_id','winner_name','loser_id','tourney_date','match_num',
                              'surface','w_ace','w_svpt','w_SvGms','l_ace','l_svpt','l_SvGms',
                              'w_df','l_bpSaved','l_bpFaced','w_bpSaved','w_bpFaced']]  \
        .rename(columns={'winner_id':'id', 'winner_name':'name', 'w_ace':'ace',
                         'w_svpt':'svpt', 'w_SvGms':'svgms','loser_id':'opp_id',
                         'l_ace':'ace_allowed','l_svpt':'svpt_allowed','l_SvGms':'svgms_allowed',
                         'w_df':'df','l_bpSaved':'bp_failed','l_bpFaced':'bp_attempt',
                         'w_bpSaved':'bp_prevent','w_bpFaced':'bp_attempt_against'})
    losing_aces = tennis_logs[['loser_id','loser_name','winner_id','tourney_date','match_num',
                              'surface','l_ace','l_svpt','l_SvGms','w_ace','w_svpt','w_SvGms',
                              'l_df','l_bpSaved','l_bpFaced','w_bpSaved','w_bpFaced']]  \
        .rename(columns={'loser_id':'id', 'loser_name':'name', 'l_ace':'ace',
                         'l_svpt':'svpt', 'l_SvGms':'svgms', 'winner_id':'opp_id',
                         'w_ace':'ace_allowed','w_svpt':'svpt_allowed','w_SvGms':'svgms_allowed',
                         'l_df':'df','l_bpSaved':'bp_prevent','l_bpFaced':'bp_attempt_against',
                         'w_bpSaved':'bp_failed','w_bpFaced':'bp_attempt'})
    aces = pd.concat([winning_aces, losing_aces])
    aces.query("surface != 'Carpet'", inplace=True)
    aces['bp_converted'] = aces['bp_attempt'] - aces['bp_failed']
    aces['bp_allowed'] = aces['bp_attempt_against'] - aces['bp_prevent']

    aces.dropna(inplace=True)
    aces.sort_values(['id','tourney_date','match_num'], inplace=True)
    aces.reset_index(drop=True, inplace=True)

    aces['match_all'] = aces.groupby(['id'])['tourney_date'].rank(method='first', ascending=True)
    aces['df_roll'] = aces.groupby(['id'], sort=False)['df'].ewm(halflife=4).mean().reset_index(drop=True) * aces['match_all']
    aces['svgms_all_roll'] = aces.groupby(['id'], sort=False)['svgms'].ewm(halflife=4).mean().reset_index(drop=True) * aces['match_all']

    a_df = aces['df_roll'].mean()
    b_df = aces['svgms_all_roll'].mean()
    
    aces['df_pred'] = (a_df + aces['df_roll']) / (a_df + b_df + aces['svgms_all_roll'])

    aces.dropna(inplace=True)
    aces.sort_values(['id','surface','tourney_date','match_num'], inplace=True)
    aces.reset_index(drop=True, inplace=True)

    aces['match_surface'] = aces.groupby(['id','surface'])['tourney_date'].rank(method='first', ascending=True)
    aces['ace_roll'] = aces.groupby(['id','surface'], sort=False)['ace'].ewm(halflife=4).mean().reset_index(drop=True) * aces['match_surface']
    aces['ace_allowed_roll'] = aces.groupby(['id','surface'], sort=False)['ace_allowed'].ewm(halflife=4).mean().reset_index(drop=True) * aces['match_surface']
    aces['break_roll'] = aces.groupby(['id','surface'], sort=False)['bp_converted'].ewm(halflife=4).mean().reset_index(drop=True) * aces['match_surface']
    aces['break_allowed_roll'] = aces.groupby(['id','surface'], sort=False)['bp_allowed'].ewm(halflife=4).mean().reset_index(drop=True) * aces['match_surface']
    aces['svgms_roll'] = aces.groupby(['id','surface'], sort=False)['svgms'].ewm(halflife=4).mean().reset_index(drop=True) * aces['match_surface']
    aces['svgms_allowed_roll'] = aces.groupby(['id','surface'], sort=False)['svgms_allowed'].ewm(halflife=4).mean().reset_index(drop=True) * aces['match_surface']

    aces.dropna(inplace=True)
    aces.reset_index(drop=True, inplace=True)

    a_grass = aces.query("surface == 'Grass'")['ace_roll'].mean()
    b_grass = aces.query("surface == 'Grass'")['svgms_roll'].mean()
    a_clay = aces.query("surface == 'Clay'")['ace_roll'].mean()
    b_clay = aces.query("surface == 'Clay'")['svgms_roll'].mean()
    a_hard = aces.query("surface == 'Hard'")['ace_roll'].mean()
    b_hard = aces.query("surface == 'Hard'")['svgms_roll'].mean()
    
    aces['surface_ace_mean'] = np.where(aces['surface'] == 'Grass', a_grass/b_grass,
                                    np.where(aces['surface'] == 'Clay', a_clay/b_clay,
                                             a_hard/b_hard))
    
    aces['ace_rate_pred'] = np.where(aces['surface']=='Grass', ((a_grass + aces['ace_roll']) / (a_grass + b_grass + aces['svgms_roll'])),
                                     np.where(aces['surface']=='Clay', ((a_clay + aces['ace_roll']) / (a_clay + b_clay + aces['svgms_roll'])),
                                              ((a_hard + aces['ace_roll']) / (a_hard + b_hard + aces['svgms_roll']))))
    
    aces['ace_rate_allowed_pred'] = np.where(aces['surface']=='Grass', ((a_grass + aces['ace_allowed_roll']) / (a_grass + b_grass + aces['svgms_allowed_roll'])),
                                     np.where(aces['surface']=='Clay', ((a_clay + aces['ace_allowed_roll']) / (a_clay + b_clay + aces['svgms_allowed_roll'])),
                                              ((a_hard + aces['ace_allowed_roll']) / (a_hard + b_hard + aces['svgms_allowed_roll']))))
    
    aces['aroa+'] = (aces['ace_rate_pred']/(aces['surface_ace_mean']))-1
    aces['araoa+'] = (aces['ace_rate_allowed_pred']/(aces['surface_ace_mean']))-1
        
    abp_grass = aces.query("surface == 'Grass'")['break_roll'].mean()
    abp_clay = aces.query("surface == 'Clay'")['break_roll'].mean()
    abp_hard = aces.query("surface == 'Hard'")['break_roll'].mean()
    
    aces['surface_bp_mean'] = np.where(aces['surface'] == 'Grass', abp_grass/b_grass,
                                    np.where(aces['surface'] == 'Clay', abp_clay/b_clay,
                                             abp_hard/b_hard))
    
    aces['bp_rate_pred'] = np.where(aces['surface']=='Grass', ((abp_grass + aces['break_roll']) / (abp_grass + b_grass + aces['svgms_roll'])),
                                     np.where(aces['surface']=='Clay', ((abp_clay + aces['break_roll']) / (abp_clay + b_clay + aces['svgms_roll'])),
                                              ((abp_hard + aces['break_roll']) / (abp_hard + b_hard + aces['svgms_roll']))))
    
    aces['bp_rate_allowed_pred'] = np.where(aces['surface']=='Grass', ((abp_grass + aces['break_allowed_roll']) / (abp_grass + b_grass + aces['svgms_allowed_roll'])),
                                     np.where(aces['surface']=='Clay', ((abp_clay + aces['break_allowed_roll']) / (abp_clay + b_clay + aces['svgms_allowed_roll'])),
                                              ((abp_hard + aces['break_allowed_roll']) / (abp_hard + b_hard + aces['svgms_allowed_roll']))))
    
    aces['bproa+'] = (aces['bp_rate_pred']/(aces['surface_bp_mean']))-1
    aces['bpraoa+'] = (aces['bp_rate_allowed_pred']/(aces['surface_bp_mean']))-1

    aces_current = name_cleanup(aces.drop_duplicates(['id','surface'], keep='last')) \
        [['name','surface','aroa+','araoa+','surface_ace_mean','df_pred',
          'surface_bp_mean','bproa+','bpraoa+']] \
            .set_index(['name','surface']) \
                .to_dict(orient='index')
                
    aces_current[(f'baseline_{tour}','Grass')] = {'aroa+':0, 'araoa+':0, 'surface_ace_mean':a_grass/b_grass,
                                          'df_pred':a_df/b_df, 'surface_bp_mean':abp_grass/b_grass,
                                          'bproa+':0, 'bpraoa+':0}
    aces_current[(f'baseline_{tour}','Clay')] = {'aroa+':0, 'araoa+':0, 'surface_ace_mean':a_clay/b_clay,
                                          'df_pred':a_df/b_df, 'surface_bp_mean':abp_clay/b_clay,
                                          'bproa+':0, 'bpraoa+':0}
    aces_current[(f'baseline_{tour}','Hard')] = {'aroa+':0, 'araoa+':0, 'surface_ace_mean':a_hard/b_hard,
                                          'df_pred':a_df/b_df, 'surface_bp_mean':abp_hard/b_hard,
                                          'bproa+':0, 'bpraoa+':0}
    
    return aces_current
        
def load_models(participants, tour, odds, surface, aces):
    if tour == 'atp':
        gms_model = joblib.load("sim_models/gms_model_men.pkl")
        scaler_gms = joblib.load('sim_models/scaler_gms_men.pkl')
    else:
        gms_model = joblib.load("sim_models/gms_model_women.pkl")
        scaler_gms = joblib.load('sim_models/scaler_gms_women.pkl')
            
    stat_categories = ['game_win_rate', 'ace_per_game', 'df_per_game', 'break_odds']
    proj_dict = {i:{j:0 for j in stat_categories} for i in participants}
    
    if (participants[0], surface) in aces:
        ace0 = aces[(participants[0], surface)]['aroa+']
        ace0_against = aces[(participants[0], surface)]['araoa+']
        bp0 = aces[(participants[0], surface)]['bproa+']
        bp0_against = aces[(participants[0], surface)]['bpraoa+']
        surface_ace_mean = aces[(participants[0], surface)]['surface_ace_mean']
        surface_bp_mean = aces[(participants[0], surface)]['surface_bp_mean']
    else:
        print(f"{participants[0]} not in Aces")
        ace0 = aces[(f'baseline_{tour}', surface)]['aroa+']
        ace0_against = aces[(f'baseline_{tour}', surface)]['araoa+']
        bp0 = aces[(f'baseline_{tour}', surface)]['surface_bp_mean']
        bp0_against = aces[(f'baseline_{tour}', surface)]['surface_bp_mean']
        surface_ace_mean = aces[(f'baseline_{tour}', surface)]['surface_ace_mean']
        surface_bp_mean = aces[(f'baseline_{tour}', surface)]['surface_bp_mean']
        
    if (participants[1], surface) in aces:
        ace1 = aces[(participants[1], surface)]['aroa+']
        ace1_against = aces[(participants[1], surface)]['araoa+']
        bp1 = aces[(participants[1], surface)]['bproa+']
        bp1_against = aces[(participants[1], surface)]['bpraoa+']
    else:
        print(f"{participants[1]} not in Aces")
        ace1 = aces[(f'baseline_{tour}', surface)]['aroa+']
        ace1_against = aces[(f'baseline_{tour}', surface)]['araoa+']
        bp1 = aces[(f'baseline_{tour}', surface)]['surface_bp_mean']
        bp1_against = aces[(f'baseline_{tour}', surface)]['surface_bp_mean']

    proj_dict[participants[0]]['ace_per_game'] = ((ace0 + ace1_against) * surface_ace_mean) + surface_ace_mean
    proj_dict[participants[0]]['break_odds'] = ((bp0 + bp1_against) * surface_bp_mean) + surface_bp_mean
    
    proj_dict[participants[1]]['ace_per_game'] = ((ace1 + ace0_against) * surface_ace_mean) + surface_ace_mean
    proj_dict[participants[1]]['break_odds'] = ((bp1 + bp0_against) * surface_bp_mean) + surface_bp_mean

    odds_array = gms_model.predict(scaler_gms.transform(np.array([odds[participants[0]][participants[1]]['moneyline']['match'],
                                                                  odds[participants[1]][participants[0]]['moneyline']['match']]).reshape(-1, 1)))
    proj_dict[participants[0]]['game_win_rate'] = odds_array[0]
    proj_dict[participants[1]]['game_win_rate'] = odds_array[1]
    
    for i in participants:
        if (i, surface) in aces:
            proj_dict[i]['df_per_game'] = aces[(i, surface)]['df_pred']
        else:
            proj_dict[i]['df_per_game'] = aces[(f'baseline_{tour}', surface)]['df_pred']            

    return proj_dict

def run_sims(participants, proj_dict_full, sets=3, sim_count=10000):
    proj_dict = {i:proj_dict_full[i] for i in proj_dict_full if i in participants}
    
    winning_sets = 2 if sets == 3 else 3
    tiebreak_sets = 2 if sets == 3 else 4
    
    roo_dict = {i:{'aces':[poisson.pmf(j, proj_dict[i]['ace_per_game']) for j in range(5)], 
                   'df':[poisson.pmf(j, proj_dict[i]['df_per_game']) for j in range(5)]} for i in proj_dict}

    stat_categories = ['games_won','games_lost','sets_won','sets_lost','match_won',
                       'aces','double_faults','breaks','clean_set','straight_sets',
                       'no_df','ace_bonus','dkp']
    results = {i:{j:[0 for k in range(sim_count)] for j in stat_categories} for i in participants}
    
    sim = 0
    while sim < sim_count:
        set_count = 0
        while (results[participants[0]]['sets_won'][sim] < winning_sets) & (results[participants[1]]['sets_won'][sim] < winning_sets):
            game_wins = {i:0 for i in participants}
            service = participants[0]
            receiver = participants[1]
            while (((game_wins[participants[0]] < 6) & (game_wins[participants[1]] < 6)) |
                   (abs(game_wins[participants[0]] - game_wins[participants[1]]) < 2)):
                game_winner = random.choices([i for i in proj_dict], 
                                             [proj_dict[i]['game_win_rate'] for i in proj_dict],
                                             k=1)[0]
                results[game_winner]['games_won'][sim] += 1
                game_wins[game_winner] += 1
                
                game_loser = [i for i in participants if i != game_winner][0]
                results[game_loser]['games_lost'][sim] += 1
                
                aces = random.choices([i for i in range(5)], 
                                      [roo_dict[service]['aces'][i] for i in range(5)],
                                      k=1)[0]
                df = random.choices([i for i in range(5)], 
                                    [roo_dict[service]['df'][i] for i in range(5)],
                                    k=1)[0]
                breaks = random.choices([0,1], [1-proj_dict[receiver]['break_odds'],proj_dict[receiver]['break_odds']],
                                        k=1)[0]
                
                results[service]['aces'][sim] += aces
                results[service]['double_faults'][sim] += df
                results[receiver]['breaks'][sim] += breaks
                
                if service == participants[0]:
                    service = participants[1]
                    receiver = participants[0]
                else:
                    service = participants[0]
                    receiver = participants[1]
                if (game_wins[game_winner] == 7) & (set_count < tiebreak_sets):
                    #print(game_wins[game_winner], game_wins[game_loser], set_count)
                    break

            results[game_winner]['sets_won'][sim] += 1
            results[game_loser]['sets_lost'][sim] += 1
            if game_wins[game_loser] == 0:
                results[game_winner]['clean_set'][sim] += 1
            set_count += 1
            
        results[game_winner]['match_won'][sim] += 1
        if results[game_loser]['sets_won'][sim] == 0:
            results[game_winner]['straight_sets'][sim] += 1
            
        for player in participants:
            results[player]['no_df'][sim] = 1 if results[player]['double_faults'][sim] == 0 else 0
            if sets == 3:
                results[player]['ace_bonus'][sim] = 1 if results[player]['aces'][sim] >= 10 else 0
                
                results[player]['dkp'][sim] = ((results[player]['games_won'][sim]*2.5) + 
                                               (results[player]['games_lost'][sim]*-2) + 
                                               (results[player]['sets_won'][sim]*6) + 
                                               (results[player]['sets_lost'][sim]*-3) + 
                                               (results[player]['match_won'][sim]*6) + 
                                               (results[player]['aces'][sim]*0.4) + 
                                               (results[player]['double_faults'][sim]*-1) + 
                                               (results[player]['breaks'][sim]*0.75) + 
                                               (results[player]['clean_set'][sim]*4) + 
                                               (results[player]['straight_sets'][sim]*6) + 
                                               (results[player]['no_df'][sim]*2.5) + 
                                               (results[player]['ace_bonus'][sim]*2) + 30)
            else:
                results[player]['ace_bonus'][sim] = 1 if results[player]['aces'][sim] >= 15 else 0
                
                results[player]['dkp'][sim] = ((results[player]['games_won'][sim]*2) + 
                                               (results[player]['games_lost'][sim]*-1.6) + 
                                               (results[player]['sets_won'][sim]*5) + 
                                               (results[player]['sets_lost'][sim]*-2.5) + 
                                               (results[player]['match_won'][sim]*5) + 
                                               (results[player]['aces'][sim]*0.25) + 
                                               (results[player]['double_faults'][sim]*-1) + 
                                               (results[player]['breaks'][sim]*0.5) + 
                                               (results[player]['clean_set'][sim]*2.5) + 
                                               (results[player]['straight_sets'][sim]*5) + 
                                               (results[player]['no_df'][sim]*5) + 
                                               (results[player]['ace_bonus'][sim]*2) + 30)
                
        
        sim += 1
            
    return results

def stokastic_sims(proj_csv, participants, sim_count=10000):
    projections = pd.read_csv(proj_csv).query("Name in @participants").reset_index(drop=True)
    projections['game_win_rate'] = projections['Game Won'] / (projections['Game Won'] + projections['Game Lost'])
    projections['ace_per_game'] = projections['Ace'] / ((projections['Game Won'] + projections['Game Lost']) / 2)
    projections['df_per_game'] = projections['DF'] / ((projections['Game Won'] + projections['Game Lost']) / 2)
    projections['break_odds'] = projections['Break'] / ((projections['Game Won'] + projections['Game Lost']) / 2)
    
    sets = projections.loc[0,'Set Won'] + projections.loc[0,'Set Lost']
    winning_sets = 2 if sets < 3 else 3
    tiebreak_sets = 2 if sets < 3 else 4
    
    proj_dict = projections.set_index(['Name']).to_dict('index')
    roo_dict = {i:{'aces':[poisson.pmf(j, proj_dict[i]['ace_per_game']) for j in range(5)], 
                   'df':[poisson.pmf(j, proj_dict[i]['df_per_game']) for j in range(5)]} for i in proj_dict}
    
    stat_categories = ['games_won','games_lost','sets_won','sets_lost','match_won',
                       'aces','double_faults','breaks','clean_set','straight_sets',
                       'no_df','ace_bonus','dkp']
    results = {i:{j:[0 for k in range(sim_count)] for j in stat_categories} for i in proj_dict}
    
    sim = 0
    while sim < sim_count:
        set_count = 0
        while (results[participants[0]]['sets_won'][sim] < winning_sets) & (results[participants[1]]['sets_won'][sim] < winning_sets):
            game_wins = {i:0 for i in participants}
            service = participants[0]
            receiver = participants[1]
            while (((game_wins[participants[0]] < 6) & (game_wins[participants[1]] < 6)) |
                   (abs(game_wins[participants[0]] - game_wins[participants[1]]) < 2)):
                game_winner = random.choices([i for i in proj_dict], 
                                             [proj_dict[i]['game_win_rate'] for i in proj_dict],
                                             k=1)[0]
                results[game_winner]['games_won'][sim] += 1
                game_wins[game_winner] += 1
                
                game_loser = [i for i in participants if i != game_winner][0]
                results[game_loser]['games_lost'][sim] += 1
                
                aces = random.choices([i for i in range(5)], 
                                      [roo_dict[service]['aces'][i] for i in range(5)],
                                      k=1)[0]
                df = random.choices([i for i in range(5)], 
                                    [roo_dict[service]['df'][i] for i in range(5)],
                                    k=1)[0]
                breaks = random.choices([0,1], [1-proj_dict[receiver]['break_odds'],proj_dict[receiver]['break_odds']],
                                        k=1)[0]
                
                results[service]['aces'][sim] += aces
                results[service]['double_faults'][sim] += df
                results[receiver]['breaks'][sim] += breaks
                
                if service == participants[0]:
                    service = participants[1]
                    receiver = participants[0]
                else:
                    service = participants[0]
                    receiver = participants[1]
                if (game_wins[game_winner] == 7) & (set_count < tiebreak_sets):
                    #print(game_wins[game_winner], game_wins[game_loser], set_count)
                    break

            results[game_winner]['sets_won'][sim] += 1
            results[game_loser]['sets_lost'][sim] += 1
            if game_wins[game_loser] == 0:
                results[game_winner]['clean_set'][sim] += 1
            set_count += 1
            
        results[game_winner]['match_won'][sim] += 1
        if results[game_loser]['sets_won'][sim] == 0:
            results[game_winner]['straight_sets'][sim] += 1
            
        for player in participants:
            results[player]['no_df'][sim] = 1 if results[player]['double_faults'][sim] == 0 else 0
            if sets < 3:
                results[player]['ace_bonus'][sim] = 1 if results[player]['aces'][sim] >= 10 else 0
                
                results[player]['dkp'][sim] = ((results[player]['games_won'][sim]*2.5) + 
                                               (results[player]['games_lost'][sim]*-2) + 
                                               (results[player]['sets_won'][sim]*6) + 
                                               (results[player]['sets_lost'][sim]*-3) + 
                                               (results[player]['match_won'][sim]*6) + 
                                               (results[player]['aces'][sim]*0.4) + 
                                               (results[player]['double_faults'][sim]*-1) + 
                                               (results[player]['breaks'][sim]*0.75) + 
                                               (results[player]['clean_set'][sim]*4) + 
                                               (results[player]['straight_sets'][sim]*6) + 
                                               (results[player]['no_df'][sim]*2.5) + 
                                               (results[player]['ace_bonus'][sim]*2) + 30)
            else:
                results[player]['ace_bonus'][sim] = 1 if results[player]['aces'][sim] >= 15 else 0
                
                results[player]['dkp'][sim] = ((results[player]['games_won'][sim]*2) + 
                                               (results[player]['games_lost'][sim]*-1.6) + 
                                               (results[player]['sets_won'][sim]*5) + 
                                               (results[player]['sets_lost'][sim]*-2.5) + 
                                               (results[player]['match_won'][sim]*5) + 
                                               (results[player]['aces'][sim]*0.25) + 
                                               (results[player]['double_faults'][sim]*-1) + 
                                               (results[player]['breaks'][sim]*0.5) + 
                                               (results[player]['clean_set'][sim]*2.5) + 
                                               (results[player]['straight_sets'][sim]*5) + 
                                               (results[player]['no_df'][sim]*5) + 
                                               (results[player]['ace_bonus'][sim]*2) + 30)
                
        
        sim += 1
            
    return results

def opto_setup(dk_csv, slate):
    player_pool = pd.read_csv(dk_csv, encoding = "ISO-8859-1")
    player_pool_info = player_pool.set_index(['Name','Roster Position'])
    draft_pool = player_pool_info.index
    name_set = player_pool_info.index.unique(0)
    costs = player_pool_info['Salary'].to_dict()
                
    draftables = pulp.LpVariable.dicts('selected', draft_pool, cat='Binary')
    prob = pulp.LpProblem('tennis', pulp.LpMaximize)
        
    # salary cap
    prob += pulp.lpSum([draftables[n]*costs[n] for n in draft_pool]) <= 50000
    #prob += pulp.lpSum([draftables[n, p, t]*costs[n,p,t] for (n, p, t) in draft_pool]) >= 46000
    
    # use each player at most only once
    for name in name_set:
        prob += pulp.lpSum([draftables[n, p] for (n, p) in draft_pool if n == name]) <= 1
        
    if slate == 'short':
        prob += pulp.lpSum([draftables[n, p] for (n, p) in draft_pool if p == 'CPT']) == 1
        
        prob += pulp.lpSum([draftables[n, p] for (n, p) in draft_pool if p == 'A-CPT']) == 1
        
        prob += pulp.lpSum([draftables[n, p] for (n, p) in draft_pool if p == 'P']) == 1
            
    else:
        prob += pulp.lpSum([draftables[n, p] for (n, p) in draft_pool]) == 6
    
    return player_pool, draft_pool, draftables, prob

def run_opto(projections, player_pool, draft_pool, draftables, prob, slate, sim):    
    lineup = []
    
    one_opto = {i:projections[i]['dkp'][sim] for i in projections}
    player_pool['dkp'] = player_pool['Name'].map(one_opto).fillna(0)
    if slate == 'short':
        player_pool['dkp'].where(player_pool['Roster Position']!='CPT', player_pool['dkp']*1.5, inplace=True)
        player_pool['dkp'].where(player_pool['Roster Position']!='A-CPT', player_pool['dkp']*1.25, inplace=True)
    values = player_pool.set_index(['Name','Roster Position'])['dkp'].to_dict()
                
    # obj
    prob += pulp.lpSum([draftables[n, p]*values[n, p] for (n, p) in draft_pool])
        
    prob.solve()
    #print(pulp.LpStatus[prob.status])
    
    for i in draftables:
        if draftables[i].varValue:
            #opto_dict[i[0]] += 1
            lineup.append(i)
    
    return lineup

def lineup_odds(data, player_dkp, count, sim, just_scores=False):
    scores = {i:0 for i in data}
    for lineup in scores:
        for player in lineup:
            if player[1] == 'CPT':
                scores[lineup] += player_dkp[player[0]][sim] * 1.5
            elif player[1] == 'A-CPT':
                scores[lineup] += player_dkp[player[0]][sim] * 1.25
            else:
                scores[lineup] += player_dkp[player[0]][sim]
    
    if just_scores:
        return scores
    
    else:
        top_lineups = sorted(scores, key=scores.get, reverse=True)[:count]
        
        return top_lineups

def lineup_data(lineups, salaries, ownership, opto_rates, slate, n):
    salaries = pd.read_csv(salaries, encoding = "ISO-8859-1").set_index(['Name','Roster Position'])[['Salary']].transpose().to_dict(orient='records')[0]

    data = {}
    for i in lineups:
        if tuple(i) not in data:
            data[tuple(i)] = {'occurences':1}
        else:
            data[tuple(i)]['occurences'] += 1
                            
        data[tuple(i)]['total_salary'] = 0
        data[tuple(i)]['ownership'] = 1
        data[tuple(i)]['theo_rate'] = 1
        data[tuple(i)]['top1p'] = 0
        data[tuple(i)]['top10p'] = 0
        for j in i:
            data[tuple(i)]['total_salary'] += salaries[j]
            own_name = j[0] if slate != 'short' else tuple(j)
            data[tuple(i)]['ownership'] *= ownership[own_name]
            data[tuple(i)]['theo_rate'] *= opto_rates[j]
    
    for i in data:
        data[i]['score'] = data[i]['occurences'] * (data[i]['theo_rate'])

    return data

def project_ownership(dk_csv, opto_rates, projections, n, slate):
    ownership_data = {}
    if slate != 'short':
        salaries = pd.read_csv(dk_csv, encoding = "ISO-8859-1").query('`Roster Position` == "P"').set_index(['Name'])[['Salary']].transpose().to_dict(orient='records')[0]
        for i in projections:
            ownership_data[i] = {'opto':opto_rates[(i,'P')], 'match_won':sum([x for x in projections[i]['match_won']])/n,
                                 'straight_sets':sum([x for x in projections[i]['straight_sets']])/n,
                                 'dkp':sum([x for x in projections[i]['dkp']])/n,
                                 'Salary':salaries[i], 'opto_salary':(opto_rates[(i,'P')]/salaries[i])*1000}
        ownership_data = pd.DataFrame(ownership_data).transpose()
        
        ownership_model = joblib.load("projections/ownership_model.pkl")
        ownership_scaler = joblib.load("projections/ownership_scaler.pkl")
        
        ownership_data['proj_own'] = ownership_model.predict(ownership_scaler.transform(ownership_data))
        ownership_data['proj_own'] = ownership_data['proj_own']/(ownership_data['proj_own'].sum()/6)
        ownership = ownership_data[['proj_own']].transpose().to_dict(orient='records')[0]
    else:
        salaries = pd.read_csv(dk_csv, encoding = "ISO-8859-1").set_index(['Name','Roster Position'])[['Salary']].transpose().to_dict(orient='records')[0]
        
        for i in opto_rates:
            ownership_data[i] = {'opto':opto_rates[i], 'match_won':sum([x for x in projections[i[0]]['match_won']])/n,
                                 'straight_sets':sum([x for x in projections[i[0]]['straight_sets']])/n,
                                 'dkp':sum([x for x in projections[i[0]]['dkp']])/n,
                                 'salary':salaries[i], 'opto_salary':(opto_rates[i]/salaries[i])*1000,
                                 'games_won':sum([x for x in projections[i[0]]['games_won']])/n,
                                 'total_opto':sum([opto_rates[(i[0],j)] for j in ['CPT','A-CPT','P']]),
                                 'opto_squared':opto_rates[i]**2}
            
        cpt_data = pd.DataFrame({i:ownership_data[i] for i in ownership_data if 'CPT' in i}).transpose() \
            .rename(columns={'opto':'CPT_opto','opto_salary':'cpt_opto_salary',
                             'salary':'CPT_salary','opto_squared':'cpt_opto_squared'}) \
                .drop(['games_won'], axis=1)
        acpt_data = pd.DataFrame({i:ownership_data[i] for i in ownership_data if 'A-CPT' in i}).transpose() \
            .rename(columns={'opto':'ACPT_opto','opto_salary':'acpt_opto_salary',
                             'salary':'ACPT_salary'}) \
                .drop(['opto_squared'], axis=1)
        p_data = pd.DataFrame({i:ownership_data[i] for i in ownership_data if 'P' in i}).transpose() \
            .rename(columns={'opto':'P_opto','opto_salary':'p_opto_salary',
                             'opto_squared':'p_opto_squared'}) \
                .drop(['salary','games_won','dkp','total_opto'], axis=1)
        
        cpt_model = joblib.load("projections/cpt_model.pkl")
        cpt_scaler = joblib.load("projections/cpt_scaler.pkl")
        acpt_model = joblib.load("projections/acpt_model.pkl")
        acpt_scaler = joblib.load("projections/acpt_scaler.pkl")
        p_model = joblib.load("projections/p_model.pkl")
        p_scaler = joblib.load("projections/p_scaler.pkl")
                
        cpt_data['proj_own'] = cpt_model.predict(cpt_scaler.transform(cpt_data))
        cpt_data['proj_own'].where(cpt_data['CPT_salary']!=50000, 0, inplace=True)
        cpt_data['proj_own'] = cpt_data['proj_own'] / cpt_data['proj_own'].sum()
        acpt_data['proj_own'] = acpt_model.predict(acpt_scaler.transform(acpt_data))
        acpt_data['proj_own'].where(acpt_data['ACPT_salary']!=50000, 0, inplace=True)
        acpt_data['proj_own'] = acpt_data['proj_own'] / acpt_data['proj_own'].sum()
        p_data['proj_own'] = p_model.predict(p_scaler.transform(p_data))
        p_data['proj_own'].where(p_data['P_opto']!=0, 0, inplace=True)
        p_data['proj_own'] = p_data['proj_own'] / p_data['proj_own'].sum()
        
        ownership = pd.concat([cpt_data, acpt_data, p_data])[['proj_own']].transpose().to_dict(orient='records')[0]
        
    return ownership

def lineup_filter(data, my_entries, limit_allocations, allo_range, limit_doops=0.2):
    selection_pool = list(data.keys())
    values = {i:data[i]['score'] for i in data}
    costs = {i:data[i]['ownership'] for i in data}
    draftables = pulp.LpVariable.dicts('selected', selection_pool, cat='Binary')
    ownership_cap = my_entries*(limit_doops**6)
    prob = pulp.LpProblem('tennis', pulp.LpMaximize)
    
    # maximize lineup strength
    prob += pulp.lpSum([draftables[x]*values[x] for x in selection_pool])
        
    # use each lineup at most only once
    for lu in selection_pool:
        prob += pulp.lpSum([draftables[x] for x in selection_pool if x == lu]) <= 1
        
    prob += pulp.lpSum([draftables[x] for x in selection_pool]) == my_entries

    prob += pulp.lpSum([draftables[x]*costs[x] for x in selection_pool]) <= ownership_cap
    
    if limit_allocations:
        for allo in limit_allocations:
            if not allo_range:
                allo_own = int(round(my_entries*limit_allocations[allo],0))
                prob += pulp.lpSum([draftables[x] for x in selection_pool if allo in x]) == allo_own
            else:
                allo_max = int(round(my_entries*(limit_allocations[allo]+allo_range),0))
                allo_min = int(round(my_entries*(limit_allocations[allo]-allo_range),0))
                prob += pulp.lpSum([draftables[x] for x in selection_pool if allo in x]) >= allo_min
                prob += pulp.lpSum([draftables[x] for x in selection_pool if allo in x]) <= allo_max
        
    prob.solve()
    #print(pulp.LpStatus[prob.status])
    
    opto_entries = []
    for i in draftables:
        if draftables[i].varValue:
            #opto_dict[i[0]] += 1
            opto_entries.append(i)
            
    return opto_entries

def contest_sims(dk_csv, agad, lineup_odds, double_matchup, ownership, player_dkp, 
                 field_size, payouts, entry_fee, sim, slate):
    if slate == 'short':        
        field = random.choices(list(lineup_odds.keys()), list(lineup_odds.values()), k=field_size-len(agad))
        tourney_dict = {i:{'points':0, 'lineup':j, 'strikeout':1, 'user':i} for i, j in enumerate(field)}
        tourney_dict = {**tourney_dict, **{i+(field_size-len(agad)):{'points':0, 'lineup':j, 'strikeout':1, 'user':'agad'} for i, j in enumerate(agad)}}
        for lu in tourney_dict:
            for player in tourney_dict[lu]['lineup']:
                if player[1] == 'CPT':
                    tourney_dict[lu]['points'] += player_dkp[player[0]][sim] * 1.5
                elif player[1] == 'A-CPT':
                    tourney_dict[lu]['points'] += player_dkp[player[0]][sim] * 1.25
                else:
                    tourney_dict[lu]['points'] += player_dkp[player[0]][sim]
                    
        points = {i:tourney_dict[i]['points'] for i in tourney_dict}            
        rank = dict(zip(points.keys(), rankdata([-i for i in points.values()], method='min')))
        entry_no = 0
        entry_list = [i for i in tourney_dict]
        while entry_no < field_size:
            entry = entry_list[entry_no]
            tourney_dict[entry]['rank'] = rank[entry]
            entry_no += 1
            
        winnings = {}
        unique_ranks = set(rank.values())
        for x in unique_ranks:
            winnings[x] = 0
            tot_winners = sum(y == x for y in rank.values())
            y = x
            while y < x+tot_winners:
                if y in payouts:
                    winnings[x] += payouts[y]
                y += 1
                        
            entry_no = 0
            while entry_no < field_size:
                entry = entry_list[entry_no]
                if tourney_dict[entry]['rank'] == x:
                    tourney_dict[entry]['winnings'] = winnings[x]/tot_winners
                    if winnings[x] > 0:
                        tourney_dict[entry]['strikeout'] = 0
                    
                entry_no += 1
            

    return tourney_dict

def dfs_optimizer(dk_csv, own_csv, matchups, values, entries, rando, limit_doops):
    player_pool = pd.read_csv(dk_csv, encoding = "ISO-8859-1")
    ownership = pd.read_csv(own_csv).set_index(['Name'])['Own'].to_dict()
    ownership = {i:ownership[i]/100 for i in ownership}
    
    matchup_dict = {}
    for i in matchups:
        mcode = f'{i[0].split()[-1]}_{i[1].split()[-1]}'
        for j in i:
            matchup_dict[j] = mcode
            
    player_pool['matchup'] = player_pool['Name'].map(matchup_dict)
    player_pool_info = player_pool.set_index(['Name','matchup'])
    draft_pool = player_pool_info.index
    name_set = player_pool_info.index.unique(0)
    matchup_set = player_pool_info.index.unique(1)
    costs = player_pool_info['Salary'].to_dict()

    lineup_values = {}
    
    lineup_count = 0
    while lineup_count < entries*100:
        rando_values = random.choices(population=np.arange(-1*rando, rando+0.01, 0.01), k=len(values))
        current_values = {(i,matchup_dict[i]):values[i]*j for i, j in zip(values, rando_values)}

        draftables = pulp.LpVariable.dicts('selected', draft_pool, cat='Binary')
        prob = pulp.LpProblem('tennis', pulp.LpMaximize)
        
        # obj
        prob += pulp.lpSum([draftables[n,m]*current_values[n,m] for (n,m) in draft_pool])
        
        # salary cap
        prob += pulp.lpSum([draftables[n,m]*costs[n,m] for (n,m) in draft_pool]) <= 50000
        #prob += pulp.lpSum([draftables[n, p, t]*costs[n,p,t] for (n, p, t) in draft_pool]) >= 46000
        
        # use each player at most only once
        for name in name_set:
            prob += pulp.lpSum([draftables[n,m] for (n,m) in draft_pool if n == name]) <= 1
                
        for mu in matchup_set:
            prob += pulp.lpSum([draftables[n,m] for (n,m) in draft_pool if m == mu]) <= 1

        prob += pulp.lpSum([draftables[n,m] for (n,m) in draft_pool]) == 6
        
        prob.solve()
        #print(pulp.LpStatus[prob.status])
        
        lineup = []
        odds, own = 1, 1
        for i in draftables:
            if draftables[i].varValue:
                #opto_dict[i[0]] += 1
                odds *= values[i[0]]
                own *= ownership[i[0]]
                lineup.append(i[0])
        
        lineup_values[tuple(lineup)] = {'score':odds, 'ownership':own}
        lineup_count += 1
        
    lineup_set = lineup_filter(lineup_values, entries, values, 0.1, limit_doops=limit_doops)
    
    return lineup_set

def past_lineup_dist(lineups, contest):
        players = pd.read_csv(contest) \
            .drop(['Rank', 'EntryId', 'EntryName', 'TimeRemaining', 'Points', 
                   'Lineup', 'Unnamed: 6'], axis=1) \
                .dropna()
        players['Player'] = players['Player'].str.strip()
        players['FPTS'].where(players['Roster Position'] != 'CPT', players['FPTS'] / 1.5, inplace=True)
        #players['Player'] = players['Player'].astype(str).apply(unidecode.unidecode)        
        players = players.set_index(['Player'])[['FPTS']].transpose().to_dict(orient='records')[0]
        
        agad_lineups = joblib.load(lineups)
        agad_points = {}
        for i in agad_lineups:
            points = 0
            for x in i:
                if x[0] in players:
                    points += players[x[0]]
            agad_points[tuple(i)] = points
            
        return agad_points, players
        
def find_past_opto(salary_csv, contest_csv):
    player_pool = pd.read_csv(salary_csv, encoding = "ISO-8859-1")
    player_pool['Name'] = player_pool['Name'].str.strip()
    point_results = pd.read_csv(contest_csv) \
       .drop(['Rank', 'EntryId', 'EntryName', 'TimeRemaining', 'Points', 
              'Lineup', 'Unnamed: 6'], axis=1) \
           .dropna()
    one_opto = point_results.set_index(['Player'])[['FPTS']].transpose().to_dict(orient='records')[0]

    player_pool_info = player_pool.set_index(['Name'])
    draft_pool = player_pool_info.index
    name_set = player_pool_info.index.unique(0)
    costs = player_pool_info['Salary'].to_dict()

    lineup = []

    player_pool['dkp'] = player_pool['Name'].map(one_opto).fillna(0)

    values = player_pool.set_index(['Name'])['dkp'].to_dict()
            
    draftables = pulp.LpVariable.dicts('selected', draft_pool, cat='Binary')
    prob = pulp.LpProblem('tennis', pulp.LpMaximize)
    
    # obj
    prob += pulp.lpSum([draftables[n]*values[n] for n in draft_pool])
    
    # salary cap
    prob += pulp.lpSum([draftables[n]*costs[n] for n in draft_pool]) <= 50000
    #prob += pulp.lpSum([draftables[n, p, t]*costs[n,p,t] for (n, p, t) in draft_pool]) >= 46000
    
    # use each player at most only once
    for name in name_set:
        prob += pulp.lpSum([draftables[n] for n in draft_pool if n == name]) <= 1
            
    prob += pulp.lpSum([draftables[n] for n in draft_pool]) == 6
    
    prob.solve()
    #print(pulp.LpStatus[prob.status])

    for i in draftables:
        if draftables[i].varValue:
            #opto_dict[i[0]] += 1
            lineup.append(i)

    return lineup

def contest_analysis(contest_csv, entry_limit, basic=False):
    standings = pd.read_csv(contest_csv, low_memory=False) \
        .drop(['Unnamed: 6', 'Player', 'Roster Position', '%Drafted', 'FPTS'], axis=1) \
            .fillna(value={'Lineup':''})
            
    if basic:
        return standings
        
    standings['playerEntry'] = standings['EntryName'].str.extract('\s\((\d{1,3})/\d{1,3}\)').fillna(1).astype(int)
    standings['playerEntryCount'] = standings['EntryName'].str.extract("/(\d{1,3})").fillna(1).astype(int)
    standings['EntryName'] = standings['EntryName'].str.replace("\s\(\d{1,3}/\d{1,3}\)", "", regex=True)
    
    entry_counts = {}
    for i in standings['EntryName'].unique():
        entry_counts[i] = standings.query("EntryName == @i").reset_index().loc[0,'playerEntryCount']
    
    if entry_limit:
        max_entries = [i for i in entry_counts if entry_counts[i] == entry_limit]
        print(f"{len(max_entries)} users took advantage of all {entry_limit} entries ({round(len(max_entries)*100/len(entry_counts),2)}%).")
    
    single_entries = [i for i in entry_counts if entry_counts[i] == 1]
    print(f"{len(single_entries)} users only entered 1 entry ({round(len(single_entries)*100/len(entry_counts),2)}%).")

    dooped_entries = standings.duplicated(keep=False, subset=['Lineup'])
    standings['dooped'] = dooped_entries

    entry_rates = standings.groupby(['Lineup']) \
        .agg({'EntryId':'count'}).rename(columns={'EntryId':'doops'}).reset_index()
    standings = standings.merge(entry_rates)
    
    if 'CPT' in standings.loc[0, 'Lineup']:
        standings['CPT'] = standings['Lineup'].str.extract(" CPT (.+?(?=P\s)|.+)", expand=False).str.strip()

        standings['ACPT'] = standings['Lineup'].str.extract("A-CPT (.+?(?=CPT)|.+)", expand=False).str.strip()

        standings['P'] = pd.DataFrame(standings['Lineup'].str.split('P ').tolist()) \
            .drop([0], axis=1) \
                .reset_index(drop=True)
    else:
        flexes = pd.DataFrame(standings['Lineup'].str.split('P ').tolist()) \
            .drop([0], axis=1) \
                .reset_index(drop=True)
        standings = standings.merge(flexes, left_index=True, right_index=True)
        for i in range(1, 7):
            standings[i] = standings[i].str.strip()
    
    return standings

def chs_setup(contest, proj_path, slate):
    standings = pd.read_csv(contest, encoding = "ISO-8859-1") \
        .drop(['Unnamed: 6', 'Player', 'Roster Position', '%Drafted', 'FPTS'], axis=1) \
            .fillna(value={'Lineup':''})
    standings['playerEntry'] = standings['EntryName'].str.extract('\s\((\d{1,3})/\d{1,3}\)').fillna(1).astype(int)
    standings['EntryName'] = standings['EntryName'].str.replace("\s\(\d{1,3}/\d{1,3}\)", "", regex=True)
    
    if slate == 'short':
        standings['CPT'] = standings['Lineup'].str.extract(" CPT (.+?(?=P\s)|.+)", expand=False).str.strip()
    
        standings['ACPT'] = standings['Lineup'].str.extract("A-CPT (.+?(?=CPT)|.+)", expand=False).str.strip()
    
        standings['P'] = pd.DataFrame(standings['Lineup'].str.split('P ').tolist()) \
            .drop([0], axis=1) \
                .reset_index(drop=True)
        
        contest_lineups = standings.set_index(['EntryName','playerEntry'])[['CPT','ACPT','P']].to_dict(orient='index')
    else:
        utils = pd.DataFrame(standings['Lineup'].str.split('P ').tolist()) \
            .drop([0], axis=1) \
                .reset_index(drop=True) \
                    .fillna('loser')
        standings = standings.merge(utils, left_index=True, right_index=True)
            
        contest_lineups = standings.set_index(['EntryName','playerEntry'])[[1,2,3,4,5,6]].to_dict(orient='index')
        contest_lineups = {i:{j:contest_lineups[i][j].strip() for j in contest_lineups[i]} for i in contest_lineups}

    projections = joblib.load(proj_path)
    player_dkp = {i:projections[i]['dkp'] for i in projections}
        
    return contest_lineups, player_dkp

def contest_history_sims(payout, fee, contest_lineups, player_dkp):
    sim = int(25000 * random.random())

    alt_standings = {tuple(i): {'lineup':contest_lineups[tuple(i)]} for i in contest_lineups}
    points_dict = {tuple(i):0 for i in contest_lineups}
    for i in contest_lineups:
        for pos in contest_lineups[i]:
            player = contest_lineups[i][pos]
            if contest_lineups[i][pos] in player_dkp:
                if pos == 'CPT':
                    points_dict[tuple(i)] += player_dkp[player][sim] *1.5
                elif pos == 'ACPT':
                    points_dict[tuple(i)] += player_dkp[player][sim] *1.25
                else:
                    points_dict[tuple(i)] += player_dkp[player][sim]
            else:
                #print(contest_lineups[i][pos])
                if contest_lineups[i][pos] == 'loser':
                    points_dict[tuple(i)] += 0
                elif pos == 'CPT':
                    points_dict[tuple(i)] += 60
                elif pos == 'ACPT':
                    points_dict[tuple(i)] += 45
                else:
                    points_dict[tuple(i)] += 30
                
    total_entries = len(alt_standings)
    contest_entries = list(alt_standings.keys())
    i = 0
    entries = 0
    while i < total_entries:
        if contest_entries[i][0] == 'agad495':
            entries += 1
        i += 1
        
    rank = dict(zip(points_dict.keys(), rankdata([-i for i in points_dict.values()], method='min')))
    
    i = 0
    while i < total_entries:
        entry = contest_entries[i]
        alt_standings[entry]['rank'] = rank[entry]
        i += 1
    
    winnings = {}
    unique_ranks = set(rank.values())
    for x in unique_ranks:
        winnings[x] = 0
        tot_winners = sum(y == x for y in rank.values())
        y = x
        while y < x+tot_winners:
            if y in payout:
                winnings[x] += payout[y]
            y += 1
                    
        i = 0
        while i < total_entries:
            entry = contest_entries[i]
            if alt_standings[entry]['rank'] == x:
                alt_standings[entry]['winnings'] = winnings[x]/tot_winners
            i += 1
            
    i = 0
    while i < total_entries:
        entry = contest_entries[i]
        if alt_standings[entry]['winnings'] == payout[1]:
            alt_standings[entry]['solo_win'] = 1
        else:
            alt_standings[entry]['solo_win'] = 0                    
        i += 1
    
    return alt_standings

def short_slate_doops(data_df):
    scaler = joblib.load("projections/doop_scaler.pkl")
    model = joblib.load("projections/doop_model.pkl")
    
    X = data_df.reset_index(drop=True)[['total_salary','ownership','theo_rate','sweep']] \
        .rename(columns={'total_salary':'salary','ownership':'prod_own','theo_rate':'prod_opto'})
        
    X = scaler.transform(X)
    
    preds = model.predict(X)
    
    return preds