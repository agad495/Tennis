import requests

def name_cleanup(player):
    name_dict = {'Alex De Minaur':'Alex de Minaur', 'Cori Gauff':'Coco Gauff',
                 'Caty Mcnally':'Catherine McNally','Bianca Vanessa Andreescu':'Bianca Andreescu',
                 'Luca Van Assche':'Luca van Assche','Yibing WU':'Yibing Wu',
                 'Mackenzie Mcdonald':'Mackenzie McDonald',
                 'Botic Van De Zandschulp':'Botic Van de Zandschulp',
                 'Albert Ramos':'Albert Ramos-Vinolas','JJ Wolf':'Jeffrey John Wolf',
                 'Maria Camila Osorio Serrano':'Camila Osorio','Dan Evans':'Daniel Evans',
                 'Dominic Stephan Stricker':'Dominic Stricker',
                 'Anna Schmiedlova':'Anna-Karolina Schmiedlova',
                 'Jaqueline Cristian':'Jaqueline Adina Cristian',
                 'Greetje Minnen':'Greet Minnen',
                 'Marcelo Tomas Barrios-Vera':'Marcelo Tomas Barrios Vera',
                 'Nuria Parrizas-Diaz':'Nuria Parrizas Diaz',
                 "Christopher O'connell":"Christopher O'Connell",
                 'Abedallah Shelbayh':'Abdullah Shelbayh',
                 #'Fanny Stollar':'Fanni Stollar', # DK apparently changes this name sometimes
                 'Mukund Sasikumar':'Sasikumar Mukund',
                 'Amarissa Toth':'Amarissa Kiara Toth','Aliona Bolsova Zadoinov':'Aliona Bolsova',
                 'Ewald, Weronika':'Weronika Ewald','Christian Garin':'Cristian Garin',
                 'Felix Auger Aliassime':'Felix Auger-Aliassime',
                 'Ludmilla Samsonova':'Liudmila Samsonova',
                 'Tomas Martn Etcheverry':'Tomas Martin Etcheverry',
                 'Ann LI':'Ann Li','Zhe LI':'Zhe Li','Ye Cong Mo':'Yecong Mo',
                 'Jodie Burrage':'Jodie Anna Burrage','Fransisco Cerundolo':'Francisco Cerundolo',
                 'Kathinka Von Deichmann':'Kathinka von Deichmann','Yeon Ku':'Yeon Woo Ku',
                 'En Shuo Liang':'En-Shuo Liang','Eudice Chong':'Eudice Wong Chong',
                 'Miriam Bulgaru':'Miriam Bianca Bulgaru','Giovanni Perricard':'Giovanni Mpetshi Perricard'}
    
    if player in name_dict:
        return name_dict[player]
    else:
        return player

def am_to_dec(am_odds):
    if am_odds > 0:
        return 1 + (am_odds/100)
    else:
        return 1 - (100/am_odds)
    
def pinny_api():
    headers = {
        'accept': 'application/json',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'application/json',
        'origin': 'https://www.pinnacle.com',
        'referer': 'https://www.pinnacle.com/',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
        'x-api-key': 'CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R',
        'x-device-uuid': '61a5b593-f3221855-5c08b5d4-18293b83',
        }
    
    matchups = requests.get("https://guest.api.arcadia.pinnacle.com/0.1/sports/33/matchups?withSpecials=false&brandId=0", headers=headers).json()
    
    straight = requests.get("https://guest.api.arcadia.pinnacle.com/0.1/sports/33/markets/straight?primaryOnly=false&withSpecials=false", headers=headers).json()
                
    periods = ['match', '1st set']
    
    games = {}
    for i in matchups:
         if ('special' not in i) & (~i['isLive']):
            for j in straight:
                if j['matchupId'] == i['id']:
                    for tm in i['participants']:
                        if tm['alignment'] == 'home':
                            home_team = name_cleanup(tm['name'])
                        else:
                            away_team = name_cleanup(tm['name'])
                                                                        
                    if (not home_team) | (not away_team) | ('(Games)' in home_team) | ('(Games)' in away_team):
                        continue
                    if home_team not in games:
                        games[home_team] = {away_team: {'moneyline': {'match': 0},
                                                        'spread': {i: {} for i in periods},
                                                        'total': {i: {'over': {}, 'under': {}} for i in periods},
                                                        'tour': i['league']['name']}}
                    if away_team not in games[home_team]:
                        games[home_team][away_team] = {'moneyline': {'match': 0},
                                                        'spread': {i: {} for i in periods},
                                                        'total': {i: {'over': {}, 'under': {}} for i in periods},
                                                        'tour': i['league']['name']}
                    if away_team not in games:
                        games[away_team] = {home_team: {'moneyline': {'match': 0},
                                                        'spread': {i: {} for i in periods},
                                                        'total': {i: {'over': {}, 'under': {}} for i in periods},
                                                        'tour': i['league']['name']}}
                    if home_team not in games[away_team]:
                        games[away_team][home_team] = {'moneyline': {'match': 0},
                                                        'spread': {i: {} for i in periods},
                                                        'total': {i: {'over': {}, 'under': {}} for i in periods},
                                                        'tour': i['league']['name']}
                        
                    if j['period'] == 0:
                        period = 'match'
                    elif j['period'] == 1:
                        period = '1st set'
                    elif j['period'] == 3:
                        pass
                    else:
                        continue
                                            
                    if j['type'] == 'moneyline':
                        for price in j['prices']:
                            if price['designation'] == 'home':
                                games[home_team][away_team]['moneyline'][period] = am_to_dec(float(price['price']))
                            elif price['designation'] == 'away':
                                games[away_team][home_team]['moneyline'][period] = am_to_dec(float(price['price']))
                            elif price['designation'] == 'draw':
                                games[home_team][away_team]['moneyline'][f'{period} Draw'] = am_to_dec(float(price['price']))
                                games[away_team][home_team]['moneyline'][f'{period} Draw'] = am_to_dec(float(price['price']))

                    if j['type'] == 'spread':
                        for price in j['prices']:
                            if price['designation'] == 'home':
                                games[home_team][away_team]['spread'][period][float(price['points'])] = am_to_dec(float(price['price']))
                            elif price['designation'] == 'away':
                                games[away_team][home_team]['spread'][period][float(price['points'])] = am_to_dec(float(price['price']))
                        
                    if j['type'] == 'total':
                        for price in j['prices']:
                            if price['designation'] == 'over':
                                games[home_team][away_team]['total'][period]['over'][float(price['points'])] = am_to_dec(float(price['price']))
                                games[away_team][home_team]['total'][period]['over'][float(price['points'])] = am_to_dec(float(price['price']))
                            elif price['designation'] == 'under':
                                games[home_team][away_team]['total'][period]['under'][float(price['points'])] = am_to_dec(float(price['price']))
                                games[away_team][home_team]['total'][period]['under'][float(price['points'])] = am_to_dec(float(price['price']))
                                        
    return games
