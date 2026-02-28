import streamlit as st
from datetime import datetime, timedelta, UTC, date
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.lines as lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch, BoxStyle, Ellipse
from PIL import Image
import urllib
import pickle
import requests
import random
import io
import time
import tqdm
import xgboost as xgb
from xgboost import XGBClassifier
from io import BytesIO
from pyfonts import set_default_font, load_google_font
from typing import List

from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

base_font = 'DM Sans'
font = load_google_font(base_font, weight='bold')
italic = load_google_font(base_font, weight='bold', italic=True)
fm.fontManager.addfont(str(font.get_file()))

## Set Styling
# Plot Style
pl_white = '#FFFFFF'
pl_background = '#292C42'
pl_text = '#72CBFD'
pl_line_color = '#8D96B3'
pl_highlight = '#F1C647'
pl_highlight_gradient = ['#F1C647','#F5A05E']
pl_highlight_cmap = sns.color_palette(f'blend:{pl_highlight_gradient[0]},{pl_highlight_gradient[1]}', as_cmap=True)

sns.set_theme(
    style={
        'axes.edgecolor': pl_line_color,
        'axes.facecolor': pl_background,
        'axes.labelcolor': pl_white,
        'xtick.color': pl_line_color,
        'ytick.color': pl_line_color,
        'figure.facecolor':pl_background,
        'grid.color': pl_background,
        'grid.linestyle': '-',
        'legend.facecolor':pl_background,
        'text.color': pl_white
     },
    font=base_font
    )
mpl.rcParams.update({"font.weight": 600})
chart_red = sns.color_palette('vlag',n_colors=10000)[-1]
chart_blue = sns.color_palette('vlag',n_colors=10000)[0]

@st.cache_data(ttl=3600)
def letter_logo():
    logo_loc = 'https://github.com/Blandalytics/baseball_snippets/blob/main/teal_letter_logo.png?raw=true'
    logo = Image.open(urllib.request.urlopen(logo_loc))
    return logo

letter_logo = letter_logo()

st.set_page_config(page_title='PLV Pitcher Game Card', page_icon=letter_logo,layout='wide')
# st.title("NFBC Draft Data, over Time")
new_title = '<p style="color:#72CBFD; font-weight: bold; font-size: 42px;">PLV Pitcher Game Card</p>'
st.markdown(new_title, unsafe_allow_html=True)

pitchtype_map = {
    'FF':'FF',
    'FA':'FF',
    'SI':'SI',
    'FT':'SI',
    'FC':'FC',
    'SL':'SL',
    'ST':'ST',
    'CH':'CH',
    'CU':'CU',
    'KC':'CU',
    'CS':'CU',
    'SV':'CU',
    'FS':'FS',
    'FO':'FS',
    'SC':'CH',
    'KN':'KN',
    'UN':'UN',
    'EP':'UN'
}

pitch_names = {
    'FF':'Four-Seam',
    'FA':'Fastball',
    'SI':'Sinker',
    'FT':'Two-Seam',
    'FS':'Splitter',
    'FO':'Forkball',
    'FC':'Cutter',
    'SL':'Slider',
    'ST':'Sweeper',
    'CU':'Curveball',
    'KC':'Knuckle Curve',
    'CS':'Slow Curve',
    'SV':'Slurve',
    'CH':'Changeup',
    'KN':'Knuckleball',
    'SC':'Screwball',
    'UN':'Unknown',
    'EP':'Eephus'
}

marker_colors = {
    'FF':'#FF6683',
    'SI':'#F2B24B',
    'FS':'#83D6FF',
    'FC':'#C59C9C',
    'SL':'#CE66FF',
    'ST':'#FFAAF7',
    'CU':'#339cff',
    'CS':'#2A98FF',
    'SV':'#2A98FF',
    'CH':'#6DE95D',
    'SC':'#6DE95D',
    'KN':'#999999',
    'SC':'#999999',
    'UN':'#999999',
}

color_min = '#4BBFDF'
color_max = '#ff5757'
diverge_palette = sns.blend_palette([color_min,'w',color_max],n_colors=5)

grade_palette = sns.blend_palette([color_min,'w',color_max],n_colors=9)
grade_colors = {
    'F':grade_palette[0],
    'D-':grade_palette[2],
    'D':grade_palette[2],
    'D+':grade_palette[2],
    'C-':grade_palette[4],
    'C':grade_palette[4],
    'C+':grade_palette[4],
    'B-':grade_palette[6],
    'B':grade_palette[6],
    'B+':grade_palette[6],
    'A-':grade_palette[8],
    'A':grade_palette[8],
    'A+':pl_highlight,
}

whiff_codes = ['S','W','T','M','O']
called_strike_codes = ['C']
foul_strike_codes = ['F','L']
batted_ball_codes = ['D','E','X']
ball_codes = ['B','*B','P']
pickoff_codes = ['1','2','3']

desc_map = {x:'swinging_strike' for x in whiff_codes}
desc_map.update({x:'called_strike' for x in called_strike_codes})
desc_map.update({x:'foul_strike' for x in foul_strike_codes})
desc_map.update({x:'in_play' for x in batted_ball_codes})
desc_map.update({x:'ball' for x in ball_codes})
desc_map.update({x:'pickoff' for x in pickoff_codes})
desc_map.update({'H':'hit_by_pitch',
                 'PSO':'step_off'})

bip_events = {'single': 'single',
 'double': 'double',
 'triple': 'triple',
 'home_run': 'home_run',
 'field_out': 'out',
 'force_out': 'out',
 'grounded_into_double_play': 'out',
 'sac_fly': 'out',
 'field_error': 'out',
 'fielders_choice': 'out',
 'double_play': 'out',
 'fielders_choice_out': 'out',
 'sac_bunt': 'out',
 'strikeout_double_play': 'out',
 'sac_fly_double_play': 'out',
 'triple_play': 'out'
}

category_feats = ['pitcherHand','hitterHand',
                  'balls_before_pitch','strikes_before_pitch']

def bio_text(pitcher_id: str):
    # Construct the URL to fetch player data
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={pitcher_id}&hydrate=currentTeam"

    # Send a GET request to the URL and parse the JSON response
    player_data = requests.get(url).json()['people'][0]

    # Extract player information from the JSON data
    player_name = player_data['fullName']
    player_number = player_data['primaryNumber']
    current_team = player_data['currentTeam']['name']
    p_hand = player_data['pitchHand']['code']
    dob = player_data['birthDate']
    dob = datetime.strptime(dob, '%Y-%m-%d')#.strftime("%m/%d/%Y")
    age = player_data['currentAge']
    height = player_data['height'].replace(' ','')
    weight = player_data['weight']
    home_city = player_data['birthCity']
    home_state = player_data['birthStateProvince'] if 'birthStateProvince' in player_data.keys() else player_data['birthCountry']
    return player_name, (f'{p_hand}',f'Age: {age}'), p_hand

def get_data(game_id: str):
    r = requests.get(f'https://statsapi.mlb.com/api/v1/game/{game_id}/playByPlay')
    x = r.json()
    x.update({'game_id':game_id})
    return x

def get_game_date(game_id: str):
    # r = requests.get(f'https://statsapi.mlb.com/api/v1/game/{game_id}/playByPlay')
    # return datetime.strptime(r.json()['allPlays'][0]['about']['startTime'][:10], '%Y-%m-%d')
    r = requests.get(f'https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live')
    return datetime.strptime(r.json()['gameData']['datetime']['originalDate'], '%Y-%m-%d').replace(tzinfo=UTC)

def game_text(pitcher_id: str, game_id: str):
    r = requests.get(f'https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore')
    home_team = r.json()['teams']['home']['team']['name']
    away_team = r.json()['teams']['away']['team']['name']
    venue = [x['value'][:-1] for x in r.json()['info'] if x['label']=='Venue'][0]
    if venue==r.json()['teams']['home']['team']['venue']['name']:
        venue = ''
    else:
        venue = f' ({venue})'
    home_away = f'vs {away_team}' if f'ID{pitcher_id}' in r.json()['teams']['home']['players'].keys() else f'@ {home_team}'
    date_text = get_game_date(game_id).strftime("%#m/%#d/%Y")
    return f'{date_text} {home_away}{venue}'

def player_height(mlbamid):
    mlbamid = int(mlbamid)
    url = f'https://statsapi.mlb.com/api/v1/people?personIds={mlbamid}&fields=people,id,height,weight'
    response = requests.get(url)
    # raise an HTTPError if the request was unsuccessful
    response.raise_for_status()
    return sum(list(map(lambda x, y: int(x)*y, response.json()['people'][0]['height'].replace("\'","").replace('"',"").split(' '),[1,1/12])))

def to_nested_dict(df, orient='index'):
    if df.index.nlevels == 1:
        return df.to_dict(orient=orient)

    result = {}

    for key, sub_df in df.groupby(level=0, sort=False):
        result[key] = to_nested_dict(sub_df.droplevel(0))
    return result

def arm_angle_day(game_date,season=2025):
    today_date = datetime.today().replace(tzinfo=UTC).astimezone(tz=None)
    if (today_date - game_date).days >= 3:
        url = f"https://baseballsavant.mlb.com/leaderboard/pitcher-arm-angles?batSide=&dateStart={game_date.strftime('%Y-%m-%d')}&dateEnd={game_date.strftime('%Y-%m-%d')}&gameType=R&groupBy=api_pitch_type_group03&min=1&minGroupPitches=1&perspective=back&pitchHand=&pitchType=&season=&size=small&sort=ascending&team=&csv=true"
    else:
        url = f"https://baseballsavant.mlb.com/leaderboard/pitcher-arm-angles?batSide=&dateStart=&dateEnd={game_date.strftime('%Y-%m-%d')}&gameType=R&groupBy=api_pitch_type_group03&min=1&minGroupPitches=1&perspective=back&pitchHand=&pitchType=&season=&size=small&sort=ascending&team=&csv=true"
    res = requests.get(url, timeout=None).content
    res_df = pd.read_csv(io.StringIO(res.decode('utf-8')))
    return to_nested_dict(res_df.groupby(['pitcher','api_pitch_type_group03'])[['ball_angle']].mean())

def name_chunk(pitcher_id,game_id,ax):
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
    x = r.json()
    team_abbr = x['scoreboard']['teams']['home' if str(pitcher_id) in x['home_pitchers'].keys() else 'away']['abbreviation']
    opp_abbr = x['scoreboard']['teams']['away' if str(pitcher_id) in x['home_pitchers'].keys() else 'home']['abbreviation']
    home_away = 'vs' if str(pitcher_id) in x['home_pitchers'].keys() else '@'

    bio_info = bio_text(pitcher_id)
    name = bio_info[0]
    name_size = len(name)
    font_adj = 10/name_size

    pl_name_gradient = ['#b4e2ff','#3da8ff']
    pl_name_cmap = sns.color_palette(f'blend:{pl_name_gradient[0]},{pl_name_gradient[1]}', as_cmap=True)

    sub_text = ax.text(0.1,0,
        f'{bio_info[1][0]}HP | {team_abbr} | {bio_info[1][1]}',
        ha='left',va='center',font=italic,color=pl_line_color,fontsize=20)

    # define text before gradient to get extent
    text = mpl.textpath.TextPath((0, 0.4), name, size=font_adj)
    # use text to define imshow extent
    extent = text.get_extents().extents[[0, 2, 1, 3]]
    im = gradient_image(ax, direction=0.7, extent=extent,
                        cmap=pl_name_cmap, cmap_range=(0, 1), alpha=1)

    # use transData instead of transAxes
    im.set_clip_path(text, transform=ax.transData)

    ax.set(ylim=(0,1.3))
    ax.axis('off')
    sns.despine()
    return f'Pitcher Performance: {x['gameDate']} {home_away} {opp_abbr}'

def load_logo():
    logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
    logo = Image.open(urllib.request.urlopen(logo_loc))
    return logo

def szn_games(player_id,game_id,sport_id=1,current_year=True):
    season = get_game_date(game_id).year
    if current_year:
        start_date = f'{season}-01-01'
        end_date = (get_game_date(game_id) - timedelta(days=1)).strftime('%Y-%m-%d') # stats from before the game
    else:
        season = season-1
        start_date = f'{season}-01-01'
        end_date = f'{season}-11-30'
    game_type_str = 'R'
    # Pull games the pitcher was in
    response = requests.get(url=f'http://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=stats(group=pitching,type=gameLog,season={season},startDate={start_date},endDate={end_date},sportId={sport_id},gameType=[{game_type_str}]),hydrations').json()
    game_ids = [x['game']['gamePk'] for x in response['people'][0]['stats'][0]['splits']]
    return game_ids

def adjusted_vaa(dataframe):
    ## Physical characteristics of pitch
    # Pitch velocity (to plate) at plate
    dataframe['vYf'] = -1 * (dataframe['vY0']**2 - (2 * dataframe['aY']*(50-17/12)))**0.5
    # Pitch time in air (50ft to home plate)
    dataframe['pitch_time_50ft'] = (dataframe['vYf'] - dataframe['vY0'])/dataframe['aY']
    # Pitch velocity (vertical) at plate
    dataframe['vZf'] = dataframe['vZ0'] + dataframe['aZ'] * dataframe['pitch_time_50ft']

    ## raw and height-adjusted VAA
    # Raw VAA
    dataframe['raw_vaa'] = -1 * np.arctan(dataframe['vZf']/dataframe['vYf']) * (180/np.pi)
    # VAA of all pitches at that height
    # dataframe['vaa_z_adj'] = dataframe['raw_vaa'].groupby([dataframe['p_z'],dataframe['year_played']]).transform('mean')
    dataframe['vaa_z_adj'] = np.where(dataframe['pZ']<3.5,
                                      dataframe['pZ'].mul(1.5635).add(-10.092),
                                      dataframe['pZ'].pow(2).mul(-0.1996).add(dataframe['pZ'].mul(2.704)).add(-11.69))
    # Adjusted VAA, based on height
    dataframe['adj_vaa'] = dataframe['raw_vaa'].sub(dataframe['vaa_z_adj'])

    return dataframe[['raw_vaa','adj_vaa']]

def pull_play(play,outside_data,base_outs,single_pitcher=True):
    game_pitcher_id = play['matchup']['pitcher']['id']
    if (int(game_pitcher_id) != int(outside_data['pitcher_id'])) & single_pitcher:
        return
    pitcher_name = play['matchup']['pitcher']['fullName']
    pitcher_hand = play['matchup']['pitchHand']['code']

    hitter_id = play['matchup']['batter']['id']
    hitter_name = play['matchup']['batter']['fullName']
    hitter_hand = play['matchup']['batSide']['code']

    pa_list = []
    base_balls = 0
    base_strikes = 0

    for pitch in play['playEvents']:
        if 'eventType' in pitch['details'].keys():
            continue
        if pitch['type']=='no_pitch':
            continue
        if 'isPitch' not in pitch.keys():
            continue
        if not pitch['isPitch']:
            continue
        pitch_count = 1
        pitch_type = pitch['details'].get('type')
        if pitch_type:
            pitch_type = pitch_type.get('code')
        pitch_description = pitch['details'].get('description')
        balls = base_balls
        base_balls = pitch['count'].get('balls')
        strikes = base_strikes
        base_strikes = pitch['count'].get('strikes')
        outs = base_outs
        event = play['result'].get('eventType')
        pitch_code = pitch['details'].get('code')
        zone = pitch['pitchData'].get('zone')
        if zone:
            zone = 0 if zone > 10 else 1
        velo = pitch['pitchData'].get('startSpeed')
        ext = pitch['pitchData'].get('extension')
        plate_time = pitch['pitchData'].get('plateTime')
        HB = pitch['pitchData']['breaks'].get('breakHorizontal')
        if HB:
            HB = float(HB) if pitcher_hand=='R' else float(HB) * -1
        IVB = pitch['pitchData']['breaks'].get('breakVerticalInduced')
        spin_rate = pitch['pitchData']['breaks'].get('spinRate')
        spin_dir = pitch['pitchData']['breaks'].get('spinDirection')
        play_id = pitch.get('playId')
        sz_top = pitch['pitchData'].get('strikeZoneTop')
        sz_bot = pitch['pitchData'].get('strikeZoneBottom')

        arm_angles = outside_data['arm_angles']
        if int(game_pitcher_id) in arm_angles.keys():
            if pitch_type in arm_angles[int(game_pitcher_id)].keys():
                arm_angle = arm_angles[int(game_pitcher_id)][pitch_type].get('ball_angle')
            else:
                arm_angle = None
        else:
            arm_angle = None

        if 'coordinates' in pitch['pitchData'].keys():
            pX = pitch['pitchData']['coordinates'].get('pX')
            pZ = pitch['pitchData']['coordinates'].get('pZ')
            x0 = pitch['pitchData']['coordinates'].get('x0')
            z0 = pitch['pitchData']['coordinates'].get('z0')
            vY0 = pitch['pitchData']['coordinates'].get('vY0')
            vZ0 = pitch['pitchData']['coordinates'].get('vZ0')
            aY = pitch['pitchData']['coordinates'].get('aY')
            aZ = pitch['pitchData']['coordinates'].get('aZ')
        else:
            pX = None
            pZ = None
            x0 = None
            z0 = None
            vY0 = None
            vZ0 = None
            aY = None
            aZ = None

        if 'hitData' in pitch.keys():
            launch_angle = pitch['hitData'].get('launchAngle')
            exit_velo = pitch['hitData'].get('launchSpeed')
            if 'coordinates' in pitch['hitData'].keys():
                hitX = pitch['hitData']['coordinates'].get('coordX')
                hitY = pitch['hitData']['coordinates'].get('coordY')
        else:
            launch_angle = None
            exit_velo = None
            hitX = None
            hitY = None

        pa_list.append([outside_data['game_id'],
                        outside_data['game_date'],
                        play_id,
                        game_pitcher_id,
                        pitcher_name,
                        pitcher_hand,
                        outside_data['pitcher_height'],
                        hitter_id,
                        hitter_name,
                        hitter_hand,
                        pitch_count,
                        pitch_type,
                        pitch_description,
                        balls,
                        strikes,
                        outs,
                        event,
                        pitch_code,
                        zone,
                        sz_top,
                        sz_bot,
                        velo,
                        arm_angle,
                        ext,
                        plate_time,
                        HB,
                        IVB,
                        spin_rate,
                        spin_dir,
                        pX,
                        pZ,
                        x0,
                        z0,
                        vY0,
                        vZ0,
                        aY,
                        aZ,
                        exit_velo,
                        launch_angle,
                        hitX,
                        hitY
                ])

    return pa_list

def header_stats_chunk(game_id,pitcher_id,ax):
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
    x = r.json()
    game_code = x['game_status_code']

    pitcher_lineup = [x['home_pitcher_lineup'][0]]+[x['away_pitcher_lineup'][0]]+([] if len(x['home_pitcher_lineup'])==1 else x['home_pitcher_lineup'][1:])+([] if len(x['away_pitcher_lineup'])==1 else x['away_pitcher_lineup'][1:])
    home_team = ['home']+['away']+([] if len(x['home_pitcher_lineup'])==1 else ['home']*(len(x['home_pitcher_lineup'])-1))+([] if len(x['away_pitcher_lineup'])==1 else ['away']*(len(x['away_pitcher_lineup'])-1))
    test_list = {}
    for home_away_pitcher in ['home','away']:
        if f'{home_away_pitcher}_pitchers' not in x.keys():
            continue
        for game_pitcher in list(x[f'{home_away_pitcher}_pitchers'].keys()):
            test_list.update({game_pitcher:x[f'{home_away_pitcher}_pitchers'][game_pitcher][0]['pitcher_name']})
    pitcher_lineup = [x for x in pitcher_lineup if str(x) in test_list.keys()]
    if len(test_list.keys())>0:
        pitcher_list = {x:y for x,y in zip(pitcher_lineup,home_team)}
    else:
        pitcher_list = {}

    home_text = pitcher_list[pitcher_id]
    opp_text = 'away' if home_text=='home' else 'home'
    stat_base = x['boxscore']['teams'][home_text]['players'][f'ID{pitcher_id}']['stats']['pitching']
    game_summary = stat_base['summary']
    team = x['scoreboard']['teams'][home_text]['abbreviation']
    opp = x['scoreboard']['teams'][opp_text]['abbreviation']
    starter = stat_base['gamesStarted']
    pitches = stat_base['numberOfPitches']
    innings = stat_base['inningsPitched']
    outs = stat_base['outs']
    earned_runs = stat_base['earnedRuns']
    tbf = stat_base['battersFaced']
    hits = stat_base['hits']
    hit_text = f'{hits} Hits' if hits!=1 else f'{hits} Hit'
    home_runs = stat_base['homeRuns']
    home_run_text = f'{home_runs} HR' if home_runs!=1 else f'{home_runs} HR'
    strikeouts = stat_base['strikeOuts']
    strikeout_text = f'{strikeouts} Ks' if strikeouts!=1 else f'{strikeouts} K'
    walks = stat_base['baseOnBalls']
    ibb = stat_base['intentionalWalks']
    ibb_text = f' ({ibb} IBBs)' if ibb>1 else f' ({ibb} IBB)' if ibb==1 else ''
    walk_text = f'{walks} BBs{ibb_text}' if walks!=1 else f'{walks} BB{ibb_text}'
    win = stat_base['wins']
    loss = stat_base['losses']
    save = stat_base['saves']
    hold = stat_base['holds']
    blown_save = stat_base['blownSaves']
    home_away = 'vs' if home_text=='home' else '@'
    supplemental_decision  = ', QS' if (int(innings[0])>=6) and (int(earned_runs)<=3) else ', BS' if blown_save==1 else ''
    decision = f'(ND{supplemental_decision})' if (win+loss==0) and (starter==1) else f'(W{supplemental_decision})' if win==1 else f'(L{supplemental_decision})' if loss==1 else '(SV)' if save==1 else '(HD)' if hold==1 else ''
    decision = decision if game_code == 'F' else ''
    whiffs = x[f'{home_text}_pitchers'][f'{pitcher_id}'][0]['avg_pitch_speed'][-1]['swinging_strikes']
    whiffs_text = f'{whiffs} Whiffs' if whiffs!=1 else f'{whiffs} Whiff'
    csw = x[f'{home_text}_pitchers'][f'{pitcher_id}'][0]['avg_pitch_speed'][-1]['csw_percent']

    game_score = 40 + (2 * int(outs) + int(strikeouts) - 2 * int(walks) - 2 * int(hits) - 3 * int(earned_runs) - 6 * int(home_runs))
    game_score = (game_score-51)/17*10+75
    if game_score<60:
        game_grade = 'F'
    elif game_score<70:
        game_grade = 'D'
    elif game_score<80:
        game_grade = 'C'
    elif game_score<90:
        game_grade = 'B'
    else:
        game_grade = 'A'

    if game_score%10 <3:
        game_grade_adj = '-'
    elif (game_score%10 >7) | (game_score >= 100):
        game_grade_adj = '+'
    else:
        game_grade_adj = ''

    game_grade = game_grade if game_score <60 else game_grade+game_grade_adj

    hr_text = '' if home_runs==0 else f' ({home_run_text})'
    game_text = f'{innings} IP, {earned_runs} ER, {hit_text}{hr_text}, {walk_text}, {strikeout_text} - {whiffs_text}, {csw}% CSW, {pitches} Pitches'

    # for stat in list(game_stat_dict.keys()):
    ax.text(0.5,
            0.5,
            game_text,
            ha='center',va='center',fontsize=30,color='#b4e2ff'
            )
    ax.set(ylim=(0,1))
    ax.axis('off')
    sns.despine()
    return game_grade

def pull_game(game_id,pitcher_id,pitcher_height):
    game_data = []
    r = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live")
    game = r.json()

    game_date = get_game_date(game_id)
    arm_angles = arm_angle_day(game_date)

    rel_dict = {
        'pitcher_id':pitcher_id,
        'game_id':game_id,
        'pitcher_height':pitcher_height,
        'game_date':game_date,
        'arm_angles':arm_angles
    }

    top_bot = 'top'
    base_outs = 0
    for play in game['liveData']['plays']['allPlays']:
        base_outs = play['playEvents'][-1]['count'].get('outs')
        play_data = pull_play(play, rel_dict,base_outs)
        if play_data:
            game_data+=play_data

    return game_data

def game_line(game_id,pitcher_id):
    game_date = get_game_date(game_id)
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
    x = r.json()
    game_code = x['game_status_code']
    pitcher_lineup = [x['home_pitcher_lineup'][0]]+[x['away_pitcher_lineup'][0]]+([] if len(x['home_pitcher_lineup'])==1 else x['home_pitcher_lineup'][1:])+([] if len(x['away_pitcher_lineup'])==1 else x['away_pitcher_lineup'][1:])
    home_team = ['home']+['away']+([] if len(x['home_pitcher_lineup'])==1 else ['home']*(len(x['home_pitcher_lineup'])-1))+([] if len(x['away_pitcher_lineup'])==1 else ['away']*(len(x['away_pitcher_lineup'])-1))
    test_list = {}
    for home_away_pitcher in ['home','away']:
        if f'{home_away_pitcher}_pitchers' not in x.keys():
            continue
        for game_pitcher in list(x[f'{home_away_pitcher}_pitchers'].keys()):
            test_list.update({game_pitcher:x[f'{home_away_pitcher}_pitchers'][game_pitcher][0]['pitcher_name']})
    pitcher_lineup = [x for x in pitcher_lineup if str(x) in test_list.keys()]
    if len(test_list.keys())>0:
        pitcher_list = {x:y for x,y in zip(pitcher_lineup,home_team)}
    else:
        pitcher_list = {}
    home_text = pitcher_list[pitcher_id]
    opp_text = 'away' if home_text=='home' else 'home'
    stat_base = x['boxscore']['teams'][home_text]['players'][f'ID{pitcher_id}']['stats']['pitching']
    opp = x['scoreboard']['teams'][opp_text]['abbreviation']
    starter = stat_base['gamesStarted']
    innings = stat_base['inningsPitched']
    outs = stat_base['outs']
    earned_runs = stat_base['earnedRuns']
    win = stat_base['wins']
    loss = stat_base['losses']
    save = stat_base['saves']
    hold = stat_base['holds']
    blown_save = stat_base['blownSaves']
    home_away = 'vs' if home_text=='home' else '@'
    supplemental_decision  = ', QS' if (int(innings[0])>=6) and (int(earned_runs)<=3) else ', BS' if blown_save==1 else ''
    decision = f'(ND{supplemental_decision})' if (win+loss==0) and (starter==1) else f'(W{supplemental_decision})' if win==1 else f'(L{supplemental_decision})' if loss==1 else '(SV)' if save==1 else '(HD)' if hold==1 else ''
    decision = decision if game_code == 'F' else ''
    return f'{game_date.strftime('%-b %-d, %Y')} {home_away} {opp} {decision}', game_date.year

def load_prev_pitches(pitcher_id,game_id=None,prev_season=None):
    data_load = []
    pitcher_height = player_height(pitcher_id)
    if prev_season:
        game_list = szn_games(pitcher_id,game_id,current_year=False)
    else:
        if not game_id:
            game_id = load_game_ids(pitcher_id)[-1]
        game_list = szn_games(pitcher_id,game_id)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(pull_game, game_id, pitcher_id, pitcher_height): game_id for game_id in game_list}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="games"):
        # for future in as_completed(futures):
            data_load += future.result()
    return data_load

def xSLGcon(data):
    with open('2025_pl_xSLG_model.pkl', 'rb') as f:
        xSLG_model = pickle.load(f)

    # Apply model (only to pitches with launch_speed and launch_angle)
    data[['out_bbe_pred','single_bbe_pred','double_bbe_pred',
          'triple_bbe_pred','home_run_bbe_pred']] =  xSLG_model.predict_proba(data[xSLG_model.feature_names_in_].astype('float'))

    data['xSLGcon'] = data[['single_bbe_pred','double_bbe_pred','triple_bbe_pred','home_run_bbe_pred']].mul([1,2,3,4]).sum(axis=1).astype('float')
    data['has_data'] = data[xSLG_model.feature_names_in_].isnull().apply(lambda x: not any(x), axis=1)
    for column in ['out_bbe_pred','single_bbe_pred','double_bbe_pred',
                   'triple_bbe_pred','home_run_bbe_pred','xSLGcon']:
        data[column] = np.where(data['has_data'],data[column],None)

    return data['xSLGcon']

def parse_height(height_str: str) -> float:
    """
    Parses the height string in the format 'ft in' and converts it to feet units.

    Parameters:
    height_str: Height string in the format 'ft in'

    Returns:
    Height in ft (float)
    """
    # split the height string into feet and inches
    feet, inches = height_str.split("' ")
    feet = int(feet)
    inches = int(inches.replace('"', ''))

    # convert the height to feet units
    height_in_feet = feet + inches / 12.0
    return height_in_feet

def chunk_list(lst: List[int], chunk_size: int) -> List[List[int]]:
    """
    Splits a list into chunks of specified size.

    Parameters:
    lst: List to be split.
    chunk_size: Size of each chunk.

    Returns:
    List of chunks.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def get_player_heights(player_ids: List[int]) -> pd.DataFrame:
    """
    Fetches the heights of players given their IDs.

    Parameters:
    player_ids: List of player IDs.

    Returns:
    pandas DataFrame of player IDs, heights.
    """
    all_data = []

    for chunk in chunk_list(player_ids, 500):
        player_id_str = ','.join(map(str, chunk))
        url = f'https://statsapi.mlb.com/api/v1/people?personIds={player_id_str}&fields=people,id,nameFirstLast,height,weight'

        response = requests.get(url)

        # raise an HTTPError if the request was unsuccessful
        response.raise_for_status()

        json_data = response.json()['people'] # parse the JSON response
        data = pd.DataFrame(json_data) # convert to DataFrame
        data['height'] = data['height'].apply(parse_height) # convert height to ft
        all_data.append(data)

        # don't overload server
        time.sleep(0.5)

    # concat chunks
    result_df = pd.concat(all_data, ignore_index=True)
    return result_df.set_index('id')['height'].to_dict()

model_constant_dict = {
    'stuff':{
        'game_mean':-0.032034,
        'type_mean':-0.034058,
        'szn_mean':-0.033370,
        'game_stdev':0.008800,
        'type_stdev':0.019083,
        'szn_stdev':0.006730
    },
    'loc':{
        'game_mean':0.025448,
        'type_mean':0.025882,
        'szn_mean':0.025643,
        'game_stdev':0.005888,
        'type_stdev':0.012447,
        'szn_stdev':0.002871
    },
    'plv':{
        'game_mean':0.026163,
        'type_mean':0.025922,
        'szn_mean':0.025552,
        'game_stdev':0.006583,
        'type_stdev':0.013552,
        'szn_stdev':0.003531
    },
}

pitchtype_metrics_dict = {
    'FF':{
        'Velo':[-10e6,90.2,93.2,95.6,98.2,10e6],
        'Ext':[-10e6,5.75,6.25,6.66,7.13,10e6],
        'IVB':[-10e6,10,14.3,16.9,19.2,10e6],
        'HB':[-10e6,1.6,5.9,9.6,13.7,10e6],
        'HAVAA':[-10e6,0.13,0.72,1.2,1.74,10e6],
        'CSW%':[-10e6,0,20,100/3,100/2,10e6],
        'xSLGcon':[-10e6,0.125,0.390,0.725,1.385,10e6],
        'plvStuff+':[-10e6,75,90,105,125,10e6],
        'PLV+':[-10e6,75,90,105,120,10e6]
        },
    'SI':{
        'Velo':[-10e6,89.1,92.4,95.1,97.7,10e6],
        'Ext':[-10e6,5.75,6.25,6.66,7.13,10e6],
        'IVB':[-10e6,0.5,6.3,10.5,15.1,10e6],
        'HB':[-10e6,9.5,13.9,16.4,18.5,10e6],
        'HAVAA':[-10e6,-0.27,0.33,0.83,1.42,10e6],
        'CSW%':[-10e6,0,100/6,100/3,100/2,10e6],
        'xSLGcon':[-10e6,0.15,0.340,0.585,1.095,10e6],
        'plvStuff+':[-10e6,75,90,100,115,10e6],
        'PLV+':[-10e6,75,90,105,120,10e6]
        },
    'FC':{
        'Velo':[-10e6,85,88,90.7,93.8,10e6],
        'Ext':[-10e6,5.75,6.25,6.66,7.13,10e6],
        'IVB':[-10e6,1.95,6.2,9.3,13.2,10e6],
        'HB':[-10e6,-6.1,-3.6,-0.8,2.7,10e6],
        'HAVAA':[-10e6,-0.9,-0.216,0.41,1.12,10e6],
        'CSW%':[-10e6,0,100/6,100/3,100/2,10e6],
        'xSLGcon':[-10e6,0.115,0.320,0.62,1.29,10e6],
        'plvStuff+':[-10e6,80,100,110,125,10e6],
        'PLV+':[-10e6,80,100,110,125,10e6]
        },
    'SL':{
        'Velo':[-10e6,80.9,84.6,87.3,90,10e6],
        'Ext':[-10e6,5.75,6.25,6.66,7.13,10e6],
        'IVB':[-10e6,-4.2,-0.2,3.6,7.6,10e6],
        'HB':[-10e6,-11.9,-5.9,-2.8,0,10e6],
        'CSW%':[-10e6,0,20,100/3,100/2,10e6],
        'xSLGcon':[-10e6,0.105,0.315,0.6,1.265,10e6],
        'plvStuff+':[-10e6,75,95,110,125,10e6],
        'PLV+':[-10e6,80,100,110,120,10e6]
        },
    'ST':{
        'Velo':[-10e6,77.3,80.8,83.4,86.2,10e6],
        'Ext':[-10e6,5.75,6.25,6.66,7.13,10e6],
        'IVB':[-10e6,-5.4,-0.9,3,7.3,10e6],
        'HB':[-10e6,-19.1,-15.5,-12.2,-8.4,10e6],
        'CSW%':[-10e6,0,20,100/3,100/2,10e6],
        'xSLGcon':[-10e6,0.08,0.275,0.575,1.275,10e6],
        'plvStuff+':[-10e6,80,100,110,130,10e6],
        'PLV+':[-10e6,80,100,110,125,10e6]
        },
    'CU':{
        'Velo':[-10e6,73.8,78.2,81.7,85.8,10e6],
        'Ext':[-10e6,5.75,6.25,6.66,7.13,10e6],
        'IVB':[-10e6,-16.6,-12.6,-7.4,-1.5,10e6],
        'HB':[-10e6,-16.3,-11.1,-6.2,-1.3,10e6],
        'CSW%':[-10e6,0,100/6,40,200/3,10e6],
        'xSLGcon':[-10e6,0.105,0.290,0.585,1.25,10e6],
        'plvStuff+':[-10e6,75,90,105,125,10e6],
        'PLV+':[-10e6,80,95,105,120,10e6]
        },
    'CH':{
        'Velo':[-10e6,80.3,84.8,88.1,91,10e6],
        'Ext':[-10e6,5.75,6.25,6.66,7.13,10e6],
        'IVB':[-10e6,-1.5,3.1,7.4,12.1,10e6],
        'HB':[-10e6,8.8,13.2,15.9,18.3,10e6],
        'CSW%':[-10e6,0,100/10,30,100/2,10e6],
        'xSLGcon':[-10e6,0.115,0.285,0.53,1.085,10e6],
        'plvStuff+':[-10e6,70,90,105,125,10e6],
        'PLV+':[-10e6,75,95,105,120,10e6]
        },
    'FS':{
        'Velo':[-10e6,82,84.9,88.1,92.3,10e6],
        'Ext':[-10e6,5.75,6.25,6.66,7.13,10e6],
        'IVB':[-10e6,-2,1.5,5.4,10.3,10e6],
        'HB':[-10e6,4.3,8.8,12.7,15.9,10e6],
        'CSW%':[-10e6,0,100/10,100/3,100/2,10e6],
        'xSLGcon':[-10e6,0.12,0.290,0.56,1.15,10e6],
        'plvStuff+':[-10e6,70,90,105,120,10e6],
        'PLV+':[-10e6,75,90,105,120,10e6]
        },
}

def pitch_models(data):
    category_feats = ['pitcherHand','hitterHand',
                      'balls_before_pitch','strikes_before_pitch']
    model_df = data.copy()

    bip_result_dict = pd.read_csv('bip_dict.csv').set_index('bb_bucket').to_dict(orient='index')

    run_expectancies = pd.read_csv('re_12_vals.csv').set_index(['cleaned_description','count']).to_dict()['delta_re']

    model_df['balls_before_pitch'] = model_df['balls'].copy()
    model_df['strikes_before_pitch'] = model_df['strikes'].copy()
    model_df['count'] = model_df['balls'].astype('str')+'_'+model_df['strikes'].astype('str')

    model_df['sz_z'] = strikezone_z(model_df,'sz_top','sz_bot')

    model_df['pitcherHeight'] = model_df['pitcherId'].map(get_player_heights(model_df['pitcherId'].unique()))
    for stat in ['extension','x0','z0']:
        model_df[stat+'_ratio'] = model_df[stat].astype('float').div(model_df['pitcherHeight'].astype('float'))

    # One-hot encode category columns
    model_df = pd.get_dummies(model_df, columns=category_feats)

    category_feats = ['pitcherHand_L','pitcherHand_R',
                      'hitterHand_L','hitterHand_R',
                      'balls_before_pitch_0','balls_before_pitch_1','balls_before_pitch_2',
                      'balls_before_pitch_3','strikes_before_pitch_0','strikes_before_pitch_1',
                      'strikes_before_pitch_2']
    for feat in category_feats:
        if feat not in model_df.columns.values:
            model_df[feat] = False

    model_df[['take_input','swing_input','called_strike_raw','ball_raw',
                'hit_by_pitch_raw','swinging_strike_raw','contact_raw',
                'foul_strike_raw','in_play_raw','10deg_raw','10-20deg_raw',
                '20-30deg_raw','30-40deg_raw','40-50deg_raw','50+deg_raw','50+deg_pred',
                'called_strike_pred','ball_pred','hit_by_pitch_pred','contact_input',
                'swinging_strike_pred','foul_strike_pred','in_play_input','50+deg_pred',
                'out_pred', 'single_pred', 'double_pred', 'triple_pred', 'home_run_pred']] = None

    for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
        model_df[[launch_angle+'_input',launch_angle+': <90mph_raw',
                launch_angle+': 90-95mph_raw',launch_angle+': 95-100mph_raw',
                launch_angle+': 100-105mph_raw',launch_angle+': 105+mph_raw',
                launch_angle+': <90mph_pred',launch_angle+': 90-95mph_pred',
                launch_angle+': 95-100mph_pred',launch_angle+': 100-105mph_pred',
                launch_angle+': 105+mph_pred']] = None

    model_df = model_df.copy()

    for model_type in ['stuff','loc','plv']:
        for pitch_type in ['Fastball','Breaking Ball','Offspeed']:
            if model_df.loc[model_df['pitch_type_bucket']==pitch_type].shape[0]==0:
                continue
            if model_type != 'stuff':
                # Swing Decision
                model = xgb.XGBClassifier()
                model.load_model(f'model_files/statcast_swing_model_{pitch_type}_{model_type}.json')

                model_df.loc[model_df['pitch_type_bucket']==pitch_type,['take_input','swing_input']] = model.predict_proba(model_df.loc[model_df['pitch_type_bucket']==pitch_type,model.feature_names_in_])

                # # Take Result
                model = xgb.XGBClassifier()
                model.load_model(f'model_files/statcast_take_model_{pitch_type}_{model_type}.json')

                model_df.loc[model_df['pitch_type_bucket']==pitch_type,['called_strike_raw','ball_raw','hit_by_pitch_raw']] = model.predict_proba(model_df.loc[model_df['pitch_type_bucket']==pitch_type,model.feature_names_in_])
                model_df.loc[model_df['pitch_type_bucket']==pitch_type,'called_strike_pred'] = model_df.loc[model_df['pitch_type_bucket']==pitch_type,'called_strike_raw'].mul(model_df.loc[model_df['pitch_type_bucket']==pitch_type,'take_input'])
                model_df.loc[model_df['pitch_type_bucket']==pitch_type,'ball_pred'] = model_df.loc[model_df['pitch_type_bucket']==pitch_type,'ball_raw'].mul(model_df.loc[model_df['pitch_type_bucket']==pitch_type,'take_input'])
                model_df.loc[model_df['pitch_type_bucket']==pitch_type,'hit_by_pitch_pred'] = model_df.loc[model_df['pitch_type_bucket']==pitch_type,'hit_by_pitch_raw'].mul(model_df.loc[model_df['pitch_type_bucket']==pitch_type,'take_input'])

            # Swing Result
            model = xgb.XGBClassifier()
            model.load_model(f'model_files/statcast_contact_model_{pitch_type}_{model_type}.json')

            if model_type == 'stuff':
                model_df.loc[model_df['pitch_type_bucket']==pitch_type,['swinging_strike_pred','contact_input']] = model.predict_proba(model_df.loc[model_df['pitch_type_bucket']==pitch_type,model.feature_names_in_])
            else:
                model_df.loc[model_df['pitch_type_bucket']==pitch_type,['swinging_strike_raw','contact_raw']] = model.predict_proba(model_df.loc[model_df['pitch_type_bucket']==pitch_type,model.feature_names_in_])
                model_df.loc[model_df['pitch_type_bucket']==pitch_type,'contact_input'] = model_df.loc[model_df['pitch_type_bucket']==pitch_type,'contact_raw'].mul(model_df.loc[model_df['pitch_type_bucket']==pitch_type,'swing_input'])
                model_df.loc[model_df['pitch_type_bucket']==pitch_type,'swinging_strike_pred'] = model_df.loc[model_df['pitch_type_bucket']==pitch_type,'swinging_strike_raw'].mul(model_df.loc[model_df['pitch_type_bucket']==pitch_type,'swing_input'])

            # Contact Result
            model = xgb.XGBClassifier()
            model.load_model(f'model_files/statcast_in_play_model_{pitch_type}_{model_type}.json')

            model_df.loc[model_df['pitch_type_bucket']==pitch_type,['foul_strike_raw','in_play_raw']] = model.predict_proba(model_df.loc[model_df['pitch_type_bucket']==pitch_type,model.feature_names_in_])
            model_df.loc[model_df['pitch_type_bucket']==pitch_type,'foul_strike_pred'] = model_df.loc[model_df['pitch_type_bucket']==pitch_type,'foul_strike_raw'].mul(model_df.loc[model_df['pitch_type_bucket']==pitch_type,'contact_input'])
            model_df.loc[model_df['pitch_type_bucket']==pitch_type,'in_play_input'] = model_df.loc[model_df['pitch_type_bucket']==pitch_type,'in_play_raw'].mul(model_df.loc[model_df['pitch_type_bucket']==pitch_type,'contact_input'])

            # Launch Angle Result
            model = xgb.XGBClassifier()
            model.load_model(f'model_files/statcast_launch_angle_model_{pitch_type}_{model_type}.json')

            model_df.loc[model_df['pitch_type_bucket']==pitch_type,['10deg_raw','10-20deg_raw','20-30deg_raw','30-40deg_raw','40-50deg_raw','50+deg_raw']] = model.predict_proba(model_df.loc[model_df['pitch_type_bucket']==pitch_type,model.feature_names_in_])
            for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
                model_df.loc[model_df['pitch_type_bucket']==pitch_type,launch_angle+'_input'] = model_df.loc[model_df['pitch_type_bucket']==pitch_type,launch_angle+'_raw'].mul(model_df.loc[model_df['pitch_type_bucket']==pitch_type,'in_play_input'])
            model_df.loc[model_df['pitch_type_bucket']==pitch_type,'50+deg_pred'] = model_df.loc[model_df['pitch_type_bucket']==pitch_type,'50+deg_raw'].mul(model_df.loc[model_df['pitch_type_bucket']==pitch_type,'in_play_input'])

            # Launch Velo Result
            for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
                model = xgb.XGBClassifier()
                model.load_model(f'model_files/statcast_{launch_angle}_model_{pitch_type}_{model_type}.json')

                model_df.loc[model_df['pitch_type_bucket']==pitch_type,[launch_angle+': <90mph_raw',launch_angle+': 90-95mph_raw',launch_angle+': 95-100mph_raw',launch_angle+': 100-105mph_raw',launch_angle+': 105+mph_raw']] = model.predict_proba(model_df.loc[model_df['pitch_type_bucket']==pitch_type,model.feature_names_in_])
                for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
                    model_df.loc[model_df['pitch_type_bucket']==pitch_type,bucket+'_pred'] = model_df.loc[model_df['pitch_type_bucket']==pitch_type,bucket+'_raw'].mul(model_df.loc[model_df['pitch_type_bucket']==pitch_type,launch_angle+'_input'])

        for outcome in ['out', 'single', 'double', 'triple', 'home_run']:
            # Start with 50+ degrees (popups)
            model_df[outcome+'_pred'] = model_df['50+deg_pred']*bip_result_dict['50+deg'][outcome]

            for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
                for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
                    model_df[outcome+'_pred'] += model_df[bucket+'_pred']*bip_result_dict[bucket][outcome]
        model_df = model_df.copy()


        outcomes = ['swinging_strike','foul_strike','out','single','double','triple','home_run']

        if model_type == 'stuff':
            er_per_pitch = 0
        else:
            outcomes += ['ball','called_strike','hit_by_pitch']
            er_per_pitch = 0.028
        std_plv_runs = 0.048
        model_df['delta_re'] = er_per_pitch
        for stat in outcomes:
            model_df[stat+'_re'] = stat if stat != 'hit_by_pitch' else 'ball' # Code HBP as Ball REs
            model_df[stat+'_re'] = model_df[[stat+'_re','count']].apply(tuple,axis=1).map(run_expectancies)
            model_df['delta_re'] = model_df['delta_re'].add(model_df[stat+'_pred'].fillna(model_df[stat+'_pred'].median()).mul(model_df[stat+'_re']))

        model_df[model_type+'Grade_game'] = -((model_df['delta_re'] - model_constant_dict[model_type]['game_mean']) / model_constant_dict[model_type]['game_stdev']) * 10 + 75
        model_df[model_type+'Grade_szn'] = -((model_df['delta_re'] - model_constant_dict[model_type]['szn_mean']) / model_constant_dict[model_type]['szn_stdev']) * 10 + 75
        if model_type=='stuff':
            model_df['plvStuff+'] = -((model_df['delta_re'] - model_constant_dict[model_type]['type_mean']) / model_constant_dict[model_type]['type_stdev']) * 15 + 100
        elif model_type=='plv':
            model_df['PLV+'] = -((model_df['delta_re'] - model_constant_dict[model_type]['type_mean']) / model_constant_dict[model_type]['type_stdev']) * 15 + 100
    return model_df[['plvStuff+','stuffGrade_game','stuffGrade_szn','locGrade_game','locGrade_szn','PLV+','plvGrade_game','plvGrade_szn']]

### Standardized Strikezone (z-location, in 'strikezones')
def strikezone_z(dataframe,top_column,bottom_column):
    dataframe[['pZ',top_column,bottom_column]] = dataframe[['pZ',top_column,bottom_column]].astype('float')

    # Ratio of 'strikezones' above/below midpoint of strikezone
    dataframe['sz_mid'] = dataframe[[top_column,bottom_column]].mean(axis=1)
    dataframe['sz_height'] = dataframe[top_column].sub(dataframe[bottom_column])

    return dataframe['pZ'].sub(dataframe['sz_mid']).div(dataframe['sz_height'])

### Calculate the differences between each pitch and their avg fastball
def fastball_differences(dataframe,stat):
    dataframe[stat] = dataframe[stat].astype('float')
    temp_df = dataframe.loc[dataframe['pitchType']==dataframe['fastball_type']].groupby(['pitcherId','gameId','pitchType'], as_index=False)[stat].mean().rename(columns={stat:'fb_'+stat})
    dataframe = dataframe.merge(temp_df,
                                left_on=['pitcherId','gameId','fastball_type'],
                                right_on=['pitcherId','gameId','pitchType']).drop(columns=['pitchType_y']).rename(columns={'pitchType_x':'pitchType'})
    return dataframe[stat].sub(dataframe['fb_'+stat])

def pull_game_info(game_id):
    code_dict = {
        'F':0,
        'U':0,
        'O':1,
        'I':1,
        'N':1,
        'P':2,
        'S':2,
        'D':2
    }
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
    x = r.json()
    game_hour = int(x['scoreboard']['datetime']['dateTime'][11:13])
    game_hour = game_hour-4 if game_hour >3 else game_hour+20
    game_minutes = int(x['scoreboard']['datetime']['dateTime'][14:16])
    raw_time = game_hour*60+game_minutes
    am_pm = 'AM' if game_hour <12 else 'PM'
    game_time = f'{game_hour-12}:{game_minutes:>02}{am_pm}' if (am_pm=='PM') & (game_hour!=12) else f'{game_hour}:{game_minutes:>02}{am_pm}'
    ppd = 0 if x['scoreboard']['datetime']['originalDate']==x['scoreboard']['datetime']['officialDate'] else 1
    
    away_team = x['scoreboard']['teams']['away']['abbreviation']
    home_team = x['scoreboard']['teams']['home']['abbreviation']
    game_status_code = x['game_status_code']
    code_map = code_dict[game_status_code]
    if game_status_code  in ['P','S','D']:
        game_info = f'{away_team} @ {home_team}: {game_time}'
        inning_sort = None
    else:
        game_info = f'{away_team} @ {home_team}'
        home_runs = x['scoreboard']['linescore']['teams']['home']['runs']
        away_runs = x['scoreboard']['linescore']['teams']['away']['runs']
        inning = x['scoreboard']['linescore']['currentInning']
        top_bot = x['scoreboard']['linescore']['inningHalf'][0]
        inning_sort = int(inning)*2 - (0 if top_bot=='Bottom' else 1)
        if game_status_code == 'F':
            if home_runs>away_runs:
                game_info = f'FINAL: {away_team} {away_runs} @ **:green[{home_team} {home_runs}]**'
            elif home_runs<away_runs:
                game_info = f'FINAL: **:green[{away_team} {away_runs}]** @ {home_team} {home_runs}'
            else:
                game_info = f'FINAL: {away_team} {away_runs} @ {home_team} {home_runs}'
        else:
            game_info = f'{top_bot}{inning}: {away_team} {away_runs} @ {home_team} {home_runs}'
    return {game_info:[game_id,game_time,raw_time,inning_sort,code_map]}

def generate_games(games_today):
    game_dict = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(pull_game_info, game_id): game_id for game_id in games_today}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="games"):
        # for future in as_completed(futures):
            game_dict.update(future.result())
    game_df = pd.DataFrame.from_dict(game_dict, orient='index',columns=['Game ID','Time','Sort Time','Sort Inning','Sort Code'])
    return game_df.sort_values(['Sort Code','Sort Time','Game ID','Sort Inning'])['Game ID'].to_dict()

st.write('Data (especially pitch types) are subject to change.')
col1, col2, col3 = st.columns([0.25,0.5,0.25])

with col1:
    today = (datetime.now(UTC)-timedelta(hours=16)).date()
    input_date = st.date_input("Select a game date:", today, 
                               min_value=date(2026, 2, 20), max_value=today+timedelta(days=2))
    r = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={input_date}')
    x = r.json()
    if x['totalGames']==0:
        print(f'No games on {input_date}')
    else:
        games_today = []
        for game in range(len(x['dates'][0]['games'])):
            if x['dates'][0]['games'][game]['gamedayType'] in ['E','P']:
                games_today += [x['dates'][0]['games'][game]['gamePk']]
        game_list = generate_games(games_today)
with col2:
    input_game = st.pills('Choose a game (all times EST):',list(game_list.keys()),default=list(game_list.keys())[0])
    game_id = game_list[input_game]
    game_id = int(game_id)
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
    x = r.json()
    game_code = x['game_status_code']
    if (len(x['home_pitcher_lineup'])>0) | (len(x['away_pitcher_lineup'])>0):
        pitcher_lineup = [x['home_pitcher_lineup'][0]]+[x['away_pitcher_lineup'][0]]+([] if len(x['home_pitcher_lineup'])==1 else x['home_pitcher_lineup'][1:])+([] if len(x['away_pitcher_lineup'])==1 else x['away_pitcher_lineup'][1:])
        home_team = [1]+[0]+([] if len(x['home_pitcher_lineup'])==1 else [1]*(len(x['home_pitcher_lineup'])-1))+([] if len(x['away_pitcher_lineup'])==1 else [0]*(len(x['away_pitcher_lineup'])-1))
        test_list = {}
        for home_away_pitcher in ['home','away']:
            if f'{home_away_pitcher}_pitchers' not in x.keys():
                continue
            for pitcher_id in list(x[f'{home_away_pitcher}_pitchers'].keys()):
                test_list.update({pitcher_id:x[f'{home_away_pitcher}_pitchers'][pitcher_id][0]['pitcher_name']})
        pitcher_lineup = [x for x in pitcher_lineup if str(x) in test_list.keys()]
        if len(test_list.keys())>0:
            pitcher_list = {test_list[str(x)]:[str(x),y] for x,y in zip(pitcher_lineup,home_team)}
        else:
            pitcher_list = {}
    else:
        pitcher_list = {}

with col3:
    # Game Line
    if len(list(pitcher_list.keys()))>0:
        pitcher_id = st.selectbox('Choose a pitcher:',list(pitcher_list.keys()))
        pitcher_id = int(pitcher_id)
        vs_past = st.checkbox("Compare to previous results?",value=True,help='If player has no 2025 MLB data, uncheck')
        spring_training = st.checkbox("Is Spring Training Game?",value=True)

# pitcher_id = st.number_input('Enter Pitcher MLBAMID',value=694973)
# game_id = st.number_input('Enter MLB Game ID',value=831490)

if vs_past:
    if spring_training:
        prev_season = True
    else:
        prev_season = False
    szn_load = load_prev_pitches(pitcher_id,game_id,
                                  prev_season=prev_season
                                  )
else:
    prev_season = False
    szn_load = []

@st.cache_data(ttl=600)
def load_data(pitcher_id,game_id,vs_past,szn_load):
    game_df = (
        pd.DataFrame(pull_game(game_id, pitcher_id, 6.25),
                     columns=['gameId','gameDate','playId',
                              'pitcherId','pitcherName','pitcherHand','pitcherHeight',
                              'hitterId','hitterName','hitterHand',
                              'isPitch','pitchType','description','balls','strikes','outs',
                              'event','code','zone','sz_top','sz_bot',
                              'velo','armAngle','extension','plate_time','HB','IVB','spin_rate','spin_dir',
                              'pX','pZ','x0','z0','vY0','vZ0','aY','aZ',
                              'launch_speed','launch_angle','hitX','hitY'])
        .query('isPitch==1')
        .drop_duplicates('playId')
        .dropna(subset=['sz_top','sz_bot','velo','extension','plate_time','HB','IVB','spin_rate',
                        'pX','pZ','x0','z0','vY0','vZ0','aY','aZ'])
        .assign(pitchType = lambda x: x['pitchType'].map(pitchtype_map),
                desc = lambda x: x['code'].map(desc_map),
                ca_str = lambda x: np.where(x['desc']=='called_strike',1,0),
                sw_str = lambda x: np.where(x['desc']=='swinging_strike',1,0),
                csw = lambda x: np.where(x['desc'].isin(['swinging_strike','called_strike']),1,0),
                swing = lambda x: np.where(x['desc'].isin(['swinging_strike','foul_strike','in_play']),1,0),
                strike = lambda x: np.where(x['desc'].isin(['called_strike','swinging_strike','foul_strike','in_play']),1,0))
        .assign(chase = lambda x: np.where((x['zone']==1),None,x['swing']),
                whiff = lambda x: np.where((x['swing']==1),x['sw_str'],None),
                hr = lambda x: np.where(x['desc']=='home_run',1,0))
        .astype({
                'pX':'float',
                'pZ':'float'
            })
        )
    game_df['balls'] = np.clip(game_df['balls'],0,3)
    game_df['strikes'] = np.clip(game_df['strikes'],0,2)
    game_df[['VAA','HAVAA']] = adjusted_vaa(game_df[['pZ','vY0','vZ0','aY','aZ']].astype('float'))
    game_df['usage'] = game_df['isPitch'].groupby([game_df['pitcherId'],game_df['gameId'],game_df['pitchType']]).transform('count') / game_df['isPitch'].groupby([game_df['pitcherId'],game_df['gameId']]).transform('count')
    game_df['vRHH'] = np.where(game_df['hitterHand']=='R',1,None)
    game_df['vLHH'] = np.where(game_df['hitterHand']=='L',1,None)
    game_df['xSLGcon'] = xSLGcon(game_df)
    
    game_df['HB_acc'] = game_df['HB'].div(game_df['plate_time']**2)
    game_df['IVB_acc'] = game_df['IVB'].div(game_df['plate_time']**2)
    game_df['Break_acc'] = (game_df['HB_acc'].astype('float')**2+game_df['IVB_acc'].astype('float')**2)**0.5
    
    fastballs = ['FF','FC','FT','SI']
    fastball_df = (game_df
                   .loc[game_df['pitchType'].isin(fastballs)]
                   .groupby(['pitcherId','gameId'], as_index=False)
                   ['pitchType']
                   .agg(pd.Series.mode)
                   .rename(columns={'pitchType':'fastball_type'})
                   .copy()
                  )
    
    # Add most common Fastball type
    game_df = game_df.merge(fastball_df,on=['pitcherId','gameId'], how='left')
    game_df['fastball_type'] = game_df['fastball_type'].fillna('NA').apply(lambda x: x if len(x[0])==1 else x[0])
    
    ### Pitch Type Grouping
    game_df['pitch_type_bucket'] = 'Other'
    game_df.loc[(game_df['pitchType']==game_df['fastball_type']) |
                 game_df['pitchType'].isin(['FF','FT','SI']),'pitch_type_bucket'] = 'Fastball'
    game_df.loc[(game_df['pitchType']!=game_df['fastball_type']) &
                 game_df['pitchType'].isin(['SL','ST','CU', 'FC']),'pitch_type_bucket'] = 'Breaking Ball'
    game_df.loc[game_df['pitchType'].isin(['CH', 'FS','KN','SC']),'pitch_type_bucket'] = 'Offspeed'
    
    for stat in ['HB_acc','IVB_acc','plate_time','velo']:
        game_df[stat+'_diff'] = fastball_differences(game_df,stat)
    game_df['Break_diff'] = (game_df['HB_acc_diff'].astype('float')**2+game_df['IVB_acc_diff'].astype('float')**2)**0.5
    
    game_df[['plvStuff+','stuffGrade_game','stuffGrade_szn','locGrade_game','locGrade_szn','PLV+','plvGrade_game','plvGrade_szn']] = pitch_models(game_df)

    game_group = (
        game_df
        .astype({'xSLGcon':'float'})
        .groupby(['pitcherId','pitcherName','pitchType'])
        [['isPitch','usage','vRHH','vLHH','armAngle','velo','extension','IVB','HB','HAVAA','strike','whiff','csw',
          'sw_str','hr','plvStuff+','PLV+','xSLGcon'
          ]]
        .agg({
            'isPitch':'sum',
            'usage':'mean',
            'vRHH':'sum',
            'vLHH':'sum',
            'armAngle':'mean',
            'velo':'mean',
            'extension':'mean',
            'IVB':'mean',
            'HB':'mean',
            'HAVAA':'mean',
            'strike':'mean',
            'whiff':'mean',
            'csw':'mean',
            'sw_str':'mean',
            'hr':'sum',
            'xSLGcon':'mean',
            'plvStuff+':'mean',
            'PLV+':'mean'
            })
        # .dropna(subset='armAngle')
        .sort_values('isPitch',ascending=False)
        .reset_index()
        .assign(Type = lambda x: x['pitchType'].map(pitch_names),
                pitches_vR = lambda x: x['vRHH'],
                vRHH = lambda x: x['vRHH'].div(x['vRHH'].sum()).astype('float')*100,
                vLHH = lambda x: np.where(x['vLHH'].sum(skipna=False)==0,None,x['vLHH'].div(x['vLHH'].sum())).astype('float')*100,
                usage = lambda x: x['usage']*100,
                strike = lambda x: x['strike'].astype('float')*100,
                whiff = lambda x: x['whiff'].astype('float')*100,
                sw_str = lambda x: x['sw_str'].astype('float')*100,
                csw = lambda x: x['csw']*100)
        [['pitchType','Type','isPitch','pitches_vR','vLHH','usage','vRHH','armAngle','velo',
          'extension','IVB','HB','HAVAA','strike','sw_str','csw','hr','xSLGcon','plvStuff+','PLV+']]
        # .round(1)
        .rename(columns={
            'isPitch':'#',
            'usage':'Usage%',
            'vRHH':'vsR',
            'vLHH':'vsL',
            'armAngle':'Arm Angle',
            'velo':'Velo',
            'extension':'Ext',
            'strike':'Str%',
            'sw_str':'SwStr%',
            'csw':'CSW%',
            'hr':'HR'
        })
        .round({'vsL':1,'Usage%':1,'vsR':1,'Velo':1,'Ext':1,'IVB':1,'HB':1,'HAVAA':1,
                'Str%':1,'SwStr%':1,'CSW%':1,'xSLGcon':3,'plvStuff+':0,'PLV+':0
                })
        .astype({'plvStuff+':'int','PLV+':'int','pitches_vR':'int'})
    )
  
    szn_df = None
    szn_group = None
    szn_comp = None
    if vs_past:
        szn_df = (
            pd.DataFrame(szn_load,
                        columns=['gameId','gameDate','playId',
                                  'pitcherId','pitcherName','pitcherHand','pitcherHeight',
                                  'hitterId','hitterName','hitterHand',
                                  'isPitch','pitchType','description','balls','strikes','outs',
                                  'event','code','zone','sz_top','sz_bot',
                                  'velo','armAngle','extension','plate_time','HB','IVB','spin_rate','spin_dir',
                                  'pX','pZ','x0','z0','vY0','vZ0','aY','aZ',
                                  'launch_speed','launch_angle','hitX','hitY'])
            .query('isPitch==1')
            .drop_duplicates('playId')
            .dropna(subset=['sz_top','sz_bot','velo','extension','plate_time','HB','IVB','spin_rate',
                            'pX','pZ','x0','z0','vY0','vZ0','aY','aZ'])
            .assign(pitchType = lambda x: x['pitchType'].map(pitchtype_map),
                    #HB = lambda x: np.where(x['pitcherHand']=='L',x['HB'].mul(-1),x['HB']),
                    desc = lambda x: x['code'].map(desc_map),
                    ca_str = lambda x: np.where(x['desc']=='called_strike',1,0),
                    sw_str = lambda x: np.where(x['desc']=='swinging_strike',1,0),
                    csw = lambda x: np.where(x['desc'].isin(['swinging_strike','called_strike']),1,0),
                    swing = lambda x: np.where(x['desc'].isin(['swinging_strike','foul_strike','in_play']),1,0))
            .assign(chase = lambda x: np.where((x['zone']==1),None,x['swing']),
                    whiff = lambda x: np.where((x['swing']==1),x['sw_str'],None))
            )
        szn_df[['VAA','HAVAA']] = adjusted_vaa(szn_df[['pZ','vY0','vZ0','aY','aZ']].astype('float'))
        szn_df['usage'] = szn_df['isPitch'].groupby([szn_df['pitcherId'],szn_df['gameId'],szn_df['pitchType']]).transform('count') / szn_df['isPitch'].groupby([szn_df['pitcherId'],szn_df['gameId']]).transform('count')
        szn_df['vRHH'] = np.where(szn_df['hitterHand']=='R',1,None)
        szn_df['vLHH'] = np.where(szn_df['hitterHand']=='L',1,None)

        szn_group = (
            szn_df
            .groupby(['pitcherId','pitcherName','pitchType'])
            [['isPitch','vRHH','vLHH','armAngle','velo','extension','IVB','HB','HAVAA','zone','chase','whiff','csw',
              # 'xwOBAcon'
              ]]
            .agg({
                'isPitch':'sum',
                'vRHH':'sum',
                'vLHH':'sum',
                'zone':'mean',
                'armAngle':'mean',
                'velo':'mean',
                'extension':'mean',
                'IVB':'mean',
                'HB':'mean',
                'HAVAA':'mean',
                'chase':'mean',
                'whiff':'mean',
                'csw':'mean',
                #'xwOBAcon':'mean'
                })
            # .dropna(subset='armAngle')
            .sort_values('isPitch',ascending=False)
            .reset_index()
            .assign(Type = lambda x: x['pitchType'].map(pitch_names),
                    vRHH = lambda x: x['vRHH'].div(x['vRHH'].sum()).astype('float')*100,
                    vLHH = lambda x: x['vLHH'].div(x['vLHH'].sum()).astype('float')*100,
                    usage = lambda x: x['isPitch'].div(x['isPitch'].sum())*100,
                    chase = lambda x: x['chase'].astype('float')*100,
                    zone = lambda x: x['zone']*100,
                    whiff = lambda x: x['whiff'].astype('float')*100,
                    csw = lambda x: x['csw']*100)
            [['pitchType','Type','isPitch','vLHH','usage','vRHH','armAngle','velo',
              'extension','IVB','HB','HAVAA','zone','chase','whiff','csw']]
            .sort_values('usage',ascending=False)
            .round(1)
            .rename(columns={
                'isPitch':'#',
                'usage':'Usage%',
                'vRHH':'vsR',
                'vLHH':'vsL',
                'armAngle':'Arm Angle',
                'velo':'Velo',
                'extension':'Ext',
                'zone':'Zone%',
                'chase':'Chase%',
                'whiff':'Whiff%',
                'csw':'CSW%'
            })
        )

        szn_comp = pd.merge(game_group,szn_group,how='left',on=['pitchType','Type'],suffixes=['','_szn']).sort_values('Usage%',ascending=False)
        for stat in ['vsL','vsR','Velo']:
            szn_comp[stat+'_diff'] = szn_comp[stat].sub(szn_comp[stat+'_szn'].fillna(szn_comp[stat]))
            szn_comp[stat+'_arrow'] = np.where(szn_comp[stat+'_diff'] >0,'↑','↓')
            szn_comp[stat+'_arrow'] = np.where(szn_comp[stat+'_diff']==0,'',szn_comp[stat+'_arrow'])

    return game_df, game_group, szn_df, szn_group, szn_comp

def gaussian_filter(kernel_size, sigma=1, muu=0):
    # Initializing value of x, y as grid of kernel size in the range of kernel size
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2 + y**2)

    # Normal part of the Gaussian function
    normal = 1 / (2 * np.pi * sigma**2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst - muu)**2 / (2.0 * sigma**2))) * normal

    return gauss

def fastball_stats(table_df,ax):
    suffix_dict = {
        'Velo':'',
        'Ext':"'",
        'IVB':'"',
        'HB':'"',
        'HAVAA':'°'
        }

    fastball_comp = (
        table_df
        .loc[table_df['pitchType'].isin(['FF','SI','FC']),['pitchType','#','Type','Velo','Ext','IVB','HB','HAVAA']]
        .head(1)
    )
    fastball_type = fastball_comp['pitchType'].item()
    stat_list = ['Velo','Ext','IVB','HB','HAVAA']
    for stat in stat_list:
        x_val = stat_list.index(stat)
        stat_val = fastball_comp[stat].item()
        fastball_comp[stat+'_color'] = pd.cut(fastball_comp[stat],
                                              bins=pitchtype_metrics_dict[fastball_type][stat],
                                              labels=range(5))
        color_val = fastball_comp[stat+'_color'].item()
        stat_color = diverge_palette[color_val]
        ax.text(x_val,
                0.3,
                f'{stat_val}{suffix_dict[stat]}',
                ha='center',va='center',fontsize=30,color=stat_color
                )
        ax.text(x_val,
                0.55,
                stat.replace(' ','\n'),
                fontsize=24,color=pl_line_color,ha='center',va='bottom')

    ax.set(xlim=(-0.75,len(stat_list)-0.25),
           ylim=(0,1)
           )
    ax.axis('off')
    sns.despine()
    return fastball_comp['Type'].item(),fastball_comp['pitchType'].item()

def usage_chunk(table_df,ax,vs_past):
    pitch_list = list(table_df['pitchType'].unique())
    max_bar = table_df[['vsR','vsL']].max().max()
    bar_lim = max_bar+max_bar/3
    fill_width = bar_lim/3
    usage_max = table_df['Usage%'].max()
    usage_min = table_df['Usage%'].min()
    table_df['usage_scale'] = table_df['Usage%'].sub(usage_min).div(usage_max-usage_min).mul(0.8).add(0.2)
    sns.barplot(table_df.assign(vsR = lambda x: x['vsR'].add(fill_width)),
                x='vsR',
                y='Type',
                hue='pitchType',
                palette=marker_colors,
                saturation=1,
                legend=False,
                zorder=0
                )
    sns.barplot(table_df.assign(vsL = lambda x: x['vsL'].add(fill_width).mul(-1)),
                x='vsL',
                y='Type',
                hue='pitchType',
                palette=marker_colors,
                saturation=1,
                legend=False,
                zorder=0
                )
    # Round the bars
    new_patches = []
    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                                abs(bb.width), abs(bb.height),
                                boxstyle="round,pad=0.01,rounding_size=2.5",
                                ec=pl_background, fc=color,
                                mutation_aspect=.1
                                )
        patch.remove()
        new_patches.append(p_bbox)

    for patch in new_patches:
        ax.add_patch(patch)

    sns.barplot(table_df.assign(blank = fill_width),
                x='blank',
                y='Type',
                color=pl_background,
                saturation=1,
                legend=False
                )
    sns.barplot(table_df.assign(blank = -fill_width),
                x='blank',
                y='Type',
                color=pl_background,
                saturation=1,
                legend=False
                )
    text_flip_thresh = 3/5
    for pitch_type in pitch_list:
        usage_val = table_df.loc[table_df['pitchType']==pitch_type,'Usage%'].item()
        usage_adjust = np.clip(usage_val,0,33) / 33
        font_size = usage_adjust * 16 + 16

        ax.text(0,
                pitch_list.index(pitch_type)+0.05,
                f'{pitch_type} {usage_val:.0f}%',
                color=marker_colors[pitch_type],ha='center',va='center',fontsize=font_size,
                )

        vsR_val = table_df.loc[table_df['pitchType']==pitch_type,'vsR'].item()
        r_text = ax.text(vsR_val + bar_lim/2.75,
                pitch_list.index(pitch_type),
                f'{vsR_val:.0f}%' if (vsR_val > 0.5) | (vsR_val == 0) else f'< 1%',
                color=marker_colors[pitch_type],
                ha='left',
                va='center',
                fontsize=20)

        if vs_past:
            vsR_arrow = table_df.loc[table_df['pitchType']==pitch_type,'vsR_arrow'].item()
            r_text = ax.annotate(f'{vsR_arrow}', xycoords=r_text, xy=(1, 0.5), va="center",
                                color=marker_colors[pitch_type],fontsize=20)
        vsL_val = table_df.loc[table_df['pitchType']==pitch_type,'vsL'].item()
        l_text = ax.text(-vsL_val - bar_lim/(2.2 if vs_past else 2.5),
                pitch_list.index(pitch_type),
                f'{vsL_val:.0f}%' if (vsL_val > 0.5) | (vsL_val == 0) else f'<1%',
                color=marker_colors[pitch_type],
                ha='right',
                va='center',
                fontsize=20)

        if vs_past:
            vsL_arrow = table_df.loc[table_df['pitchType']==pitch_type,'vsL_arrow'].item()
            l_text = ax.annotate(f'{vsL_arrow}', xycoords=l_text, xy=(1, 0.5), va="center",
                                color=marker_colors[pitch_type],fontsize=20)

    label_adj = -1.5/len(pitch_list)-0.5
    ax.text(fill_width*1.15,label_adj,'vs RHB%',fontsize=20,color=pl_line_color,ha='left',va='bottom',weight='regular')
    ax.text(-fill_width*1.15,label_adj,'vs LHB%',fontsize=20,color=pl_line_color,ha='right',va='bottom',weight='regular')
    x_lim = max(np.abs(np.array(ax.get_xlim()))) * 4/3
    y_lim = (ax.get_ylim()[0],ax.get_ylim()[1]*1.1)
    ax.set(
           xlim=(-x_lim,x_lim),
          ylim=(len(pitch_list)-0.5,label_adj)
          )
    ax.axis('off')
    sns.despine(left=True)

def header_chunk(table_df,ax):
    pitch_list = list(table_df['pitchType'].unique())
    bar_width = 2/len(pitch_list)
    for pitch_type in pitch_list:
        ax.text(0.225,pitch_list.index(pitch_type),table_df.loc[table_df['pitchType']==pitch_type,'Type'].item(),
                      ha='left',va='center',fontsize=20,color=marker_colors[pitch_type]
                      )
        ax.add_artist(mpatches.FancyBboxPatch((0.1, pitch_list.index(pitch_type)-bar_width/2), 0.075, bar_width, ec="none",
                                              color=marker_colors[pitch_type],
                                boxstyle=mpatches.BoxStyle("Round", pad=0),mutation_scale=.1
                                              )
        )
        num_thrown = table_df.loc[table_df['pitchType']==pitch_type,'#'].item()
        ax.text(0.9,pitch_list.index(pitch_type),f'{num_thrown:,.0f}',
                ha='center',va='center',fontsize=20)
      
    label_adj = -2/(len(pitch_list)-0.5)
    ax.text(0.225,label_adj,'Type',fontsize=16,color=pl_line_color,ha='left',va='bottom')
    ax.text(0.9,label_adj,'#',fontsize=16,color=pl_line_color,ha='center',va='bottom')
    ax.set(ylim=(len(pitch_list)-0.5,label_adj))
    ax.axis('off')
    sns.despine()

def stats_chunk(table_df,ax,vs_past):
    color_min = '#4BBFDF'
    color_max = '#ff5757'
    velo_diff_palette = sns.blend_palette([color_min,'w',color_max],n_colors=13)

    if vs_past:
        table_df['Velo_diff_group'] = pd.cut(table_df['Velo_diff'],
                                            bins=[-200,-2,-1,-0.5,0.5,1,2,200],
                                            labels=[0,2,4,6,8,10,12])
    stat_list = ['Velo', 'IVB', 'HB', 'Str%', 'SwStr%','CSW%',
                 'xSLGcon','plvStuff+','PLV+'
                 ]

    pitch_list = list(table_df['pitchType'].unique())
    width_dict = {
        'Arm Angle':1,
        'Velo':0,
        'Ext':0.75,
        'IVB':0.9,
        'HB':0.65,
        'HAVAA':0.9,
        'Str%':0.75,
        'Chase%':1.05,
        'SwStr%':0.75,
        'CSW%':0.75,
        'xSLGcon':0.75,
        'plvStuff+':0.75,
        'PLV+':0.7
        }
    width_dict = {x:y for x,y in width_dict.items() if x in stat_list}

    suffix_dict = {
        'Arm Angle':'°',
        'Velo':'',
        'Ext':"'",
        'IVB':'"',
        'HB':'"',
        'HAVAA':'°',
        'Str%':'%',
        'Chase%':'%',
        'SwStr%':'%',
        'CSW%':'%',
        'xSLGcon':'',
        'plvStuff+':'',
        'PLV+':''
        }

    label_adj = -2/(len(pitch_list)-0.5)
    for pitch_type in pitch_list:
        x_val = 0.15
        for stat in stat_list:
            x_val += width_dict[stat]
            y_val = pitch_list.index(pitch_type)
            ax.text(x_val,
                    label_adj,
                    stat.replace(' ','\n'),
                    fontsize=16,color=pl_line_color,ha='center',va='bottom')
            stat_val = table_df.loc[table_df['pitchType']==pitch_type,stat].item()
            if (stat == 'Velo') & vs_past:
                velo_change = table_df.loc[table_df['pitchType']==pitch_type,'Velo_diff'].item()
                vel_diff_group = table_df.loc[table_df['pitchType']==pitch_type,'Velo_diff_group'].item()
                text = ax.text(x_val-0.15,
                               y_val,
                               f'{stat_val}{suffix_dict[stat]}',
                               ha='center',va='center',fontsize=20,color='w'
                               )
                text = ax.annotate(f' ({velo_change:+.1f})', xycoords=text, xy=(1, 0.5), va="center",
                                   color=velo_diff_palette[vel_diff_group],
                                   fontsize=16)
            else:
                if (stat in ['Velo','CSW%','xSLGcon','PLV+','plvStuff+']) & (pitch_type in pitchtype_metrics_dict.keys()):
                    if np.isnan(stat_val):
                        stat_color = 'w'
                        color_val = 2
                    else:
                        table_df[stat+'_color'] = pd.cut(table_df[stat],
                                                        bins=pitchtype_metrics_dict[pitch_type][stat],
                                                        labels=range(5))
                        color_val = table_df.loc[table_df['pitchType']==pitch_type,stat+'_color'].item()
                        if stat == 'xSLGcon':
                            color_val = 4 - color_val
                        stat_color = diverge_palette[color_val]
                else:
                    stat_color = 'w'
                if np.isnan(stat_val):
                    stat_val = '-'
                else:
                    stat_val = int(stat_val) if abs(stat_val-50)==50 else stat_val
                ax.text(x_val,
                        y_val,
                        f'{stat_val}{suffix_dict[stat]}',
                        ha='center',va='center',fontsize=20,color=stat_color
                        )

    ax.set(xlim=(-0.5,sum(width_dict.values())+0.5),
          ylim=(len(pitch_list)-0.5,label_adj))
    ax.axis('off')
    sns.despine()

def gradient_image(ax, extent, direction=0.3, cmap_range=(0, 1), **kwargs):
    """
    Draw a gradient image based on a colormap.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    extent
        The extent of the image as (xmin, xmax, ymin, ymax).
        By default, this is in Axes coordinates but may be
        changed using the *transform* keyword argument.
    direction : float
        The direction of the gradient. This is a number in
        range 0 (=vertical) to 1 (=horizontal).
    cmap_range : float, float
        The fraction (cmin, cmax) of the colormap that should be
        used for the gradient, where the complete colormap is (0, 1).
    **kwargs
        Other parameters are passed on to `.Axes.imshow()`.
        In particular useful is *cmap*.
    """
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    # added origin = lower, elsewise text is flipped upside down
    im = ax.imshow(X, extent=extent,
                    interpolation='bicubic',
                    vmin=0, vmax=1, origin='lower', **kwargs)
    return im
  
def gradient_bar(ax, x, y, width=0.5, bottom=0):
    for left, top in zip(x, y):
        right = left + width
        gradient_image(ax, extent=(left, right, bottom, top),
                        cmap=pl_highlight_cmap, cmap_range=(0, 1))

logo = load_logo()

def letter_grade(val):
    return pd.cut([val],
                  bins=[-100,60,63,67,70,73,77,80,83,87,90,93,97,300],
                  labels=['F', 'D-', 'D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'])[0]

def generate_chart(pitcher_id,game_id,game_df,game_group,szn_df,szn_comp,vs_past):
    arm_angle = game_df['armAngle'].mean()
    
    fig = plt.figure(figsize=(15,20))
    # Divide card into tiles
    grid = plt.GridSpec(6, 21,height_ratios=[0.9,1.4,1.125,1.225,5,3.25],hspace=0.3,wspace=0)
    
    logo_ax = fig.add_axes([0.725,0.91,0.25,0.25], anchor='SW', zorder=1)
    logo_ax.imshow(logo)
    logo_ax.axis('off')
    
    title_text_ax = fig.add_axes([0.03,0.91,3/4,0.08], anchor='SW', zorder=1)
    game_text = name_chunk(pitcher_id,game_id,title_text_ax)
    
    game_stats_ax = fig.add_axes([0.02,0.8425,.96,0.037], anchor='SW', zorder=1)
    start_grade = header_stats_chunk(game_id,pitcher_id,game_stats_ax)
    
    start_grade_ax = fig.add_axes([0.01,0.705,.15,0.1], anchor='SW', zorder=1)
    pitch_model_ax = fig.add_axes([0.16,0.705,.275,0.1], anchor='SW', zorder=1)
    
    # Grade only
    stuff_grade = letter_grade(game_df['stuffGrade_game'].mean())
    vs_r_location_grade = letter_grade(game_df.loc[game_df['hitterHand']=='R','locGrade_game'].mean())
    vs_l_location_grade = letter_grade(game_df.loc[game_df['hitterHand']=='L','locGrade_game'].mean())
    location_grade = letter_grade(game_df['locGrade_game'].mean())
    plv_grade = letter_grade(game_df['plvGrade_game'].mean())
    
    # Drop Shadow for Start Grade
    # if start_grade[0] != 'C':
    #     kernel_size = 100
    #     x = np.linspace(0,1,kernel_size)
    #     y = np.linspace(0,1,kernel_size)
    #     X, Y = np.meshgrid(x, y)
    #     start_grade_ax.contourf(X, Y,
    #                             gaussian_filter(kernel_size,sigma=0.4),
    #                             200,
    #                             alpha=1,
    #                             zorder=0,
    #                             cmap=sns.blend_palette([pl_background,
    #                                         sns.blend_palette([pl_background,grade_colors[start_grade]],n_colors=100)[15]],
    #                                       as_cmap=True),
    #                             extent=[0,1,
    #                                     0,1])
    
    start_grade_ax.text(0.5,0.5,start_grade,ha='center',va='center',fontsize=90,color=grade_colors[start_grade])
    start_grade_ax.set(xlim=(0,1),ylim=(0,1))
    start_grade_ax.axis('off')
    
    pitch_model_ax.text(0.175,0.8,'Stuff',ha='center',va='center',fontsize=22,color=pl_line_color)
    pitch_model_ax.text(0.175,0.45,stuff_grade,ha='center',va='center',fontsize=50,color=grade_colors[stuff_grade])
    
    pitch_model_ax.text(0.5,0.8,'Locations',ha='center',va='center',fontsize=20,color=pl_line_color)
    pitch_model_ax.text(0.5,0.45,location_grade,ha='center',va='center',fontsize=50,color=grade_colors[location_grade])
    
    pitch_model_ax.text(0.825,0.8,'PLV',ha='center',va='center',fontsize=22,color=pl_line_color)
    pitch_model_ax.text(0.825,0.45,plv_grade,ha='center',va='center',fontsize=50,color=grade_colors[plv_grade])
    
    pitch_model_ax.set(xlim=(0,1))
    pitch_model_ax.axis('off')
    
    hand = bio_text(pitcher_id)[2]
    
    ax1 = fig.add_axes([0.03,0.275,0.423,0.287], anchor='SW', zorder=1)
    circle1 = plt.Circle((0, 0), 6, color=pl_white,fill=False,alpha=0.25,linestyle='--',linewidth=2)
    ax1.add_patch(circle1)
    circle2 = plt.Circle((0, 0), 12, color=pl_white,fill=False,alpha=0.5,linewidth=2)
    ax1.add_patch(circle2)
    circle3 = plt.Circle((0, 0), 18, color=pl_white,fill=False,alpha=0.25,linestyle='--',linewidth=2)
    ax1.add_patch(circle3)
    circle4 = plt.Circle((0, 0), 24, color=pl_white,fill=False,alpha=0.5,linewidth=2)
    ax1.add_patch(circle4)
    ax1.axvline(0,ymin=6/62,ymax=56/62,color=pl_white,alpha=0.5,zorder=0.5,linewidth=2)
    ax1.axhline(0,xmin=6/62,xmax=56/62,color=pl_white,alpha=0.5,zorder=0.5,linewidth=2)
    
    for dist in [12,24]:
        label_dist = dist-0.25
        ax1.text(label_dist,-0.5,f'{dist}"',ha='right',va='top',fontsize=14,color=pl_white,alpha=0.75,zorder=1)
        ax1.text(-label_dist+0.25,-0.5,f'{dist}"',ha='left',va='top',fontsize=14,color=pl_white,alpha=0.75,zorder=1)
        ax1.text(0.5,label_dist-0.5,f'{dist}"',ha='left',va='top',fontsize=14,color=pl_white,alpha=0.75,zorder=1)
        ax1.text(0.5,-label_dist+0.5,f'{dist}"',ha='left',va='bottom',fontsize=14,color=pl_white,alpha=0.75,zorder=1)
    
    if hand=='R':
        ax1.text(29,0,'Arm\nSide',ha='center',va='center',fontsize=16,color=pl_white,alpha=1,zorder=1)
        ax1.text(-29,0,'Glove\nSide',ha='center',va='center',fontsize=16,color=pl_white,alpha=1,zorder=1)
    else:
        ax1.text(29,0,'Glove\nSide',ha='center',va='center',fontsize=16,color=pl_white,alpha=1,zorder=1)
        ax1.text(-29,0,'Arm\nSide',ha='center',va='center',fontsize=16,color=pl_white,alpha=1,zorder=1)
    
    ax1.text(0,27,'Rise',ha='center',va='center',fontsize=16,color=pl_white,alpha=1,zorder=1)
    ax1.text(0,-27,'Drop',ha='center',va='center',fontsize=16,color=pl_white,alpha=1,zorder=1)
    
    pitch_list = list(game_df['pitchType'].unique())
    if vs_past:
        sns.kdeplot(szn_df.loc[szn_df['pitchType'].isin(pitch_list)].assign(HB = lambda x: np.where(x['pitcherHand']=='L',x['HB'].mul(-1),x['HB'])),
                    x='HB',
                    y='IVB',
                    hue='pitchType',
                    palette=marker_colors,
                    levels=[0.1,1],
                    fill=True,
                    bw_adjust=2.5,
                    cut=2,
                    alpha=0.25,
                    legend=False,
                    ax=ax1)
    sns.scatterplot(game_df.assign(HB = lambda x: np.where(x['pitcherHand']=='L',x['HB'].mul(-1),x['HB'])),
                    x='HB',
                    y='IVB',
                    hue='pitchType',
                    palette=marker_colors,
                    edgecolor=pl_background,
                    s=300,
                    linewidth=0.5,
                    alpha=1,
                    zorder=10,
                    ax=ax1,
                    legend=False)
    
    chart_lim = 29
    arm_rads = np.deg2rad(arm_angle)
    x_val = np.cos(arm_rads) * (1 if hand=='R' else -1) * chart_lim
    y_val = np.sin(arm_rads) * chart_lim
    ax1.plot([0,x_val],
             [0,y_val],
             color='w',linestyle='--')
    ax1.plot([0,-x_val],
             [0,-y_val],
             color='w',linestyle='--',alpha=0.1)
    ax1.text(x_val,
             y_val,
             f'{arm_angle:.0f}°',
             ha='center',va='center',
             fontsize=16,color=pl_white,fontweight='light',
             bbox=dict(facecolor=pl_background, edgecolor='w', boxstyle='round',linewidth=1,alpha=0.75))
    
    ax1.set(xlim=(-chart_lim-2,chart_lim+2),
           ylim=(-chart_lim-2,chart_lim+2),
           aspect=1)
    ax1.axis('off')
    sns.despine(left=True,bottom=True)
    
    sz_bot = 1.5
    sz_top = 3.5
    x_ft = 2.5
    y_bot = -0.5
    y_lim = 6.5
    plate_y = -.25
    alpha_val = 0.5
    
    ax2 = fig.add_axes([0.445,0.275,0.2675,0.287], anchor='SW', zorder=1)
    sns.scatterplot(data=(game_df.loc[(game_df['hitterHand']=='L')].assign(pX = lambda x: x['pX']*-1)),
                    x='pX',
                    y='pZ',
                    hue='pitchType',
                    palette=marker_colors,
                    edgecolor=pl_background,
                    s=300,
                    linewidth=0.5,
                    alpha=1,
                    legend=False,
                    zorder=10,
                    ax=ax2)
    
    # Inner Strike zone
    ax2.plot([-9.5/12,9.5/12], [1.5+2/3,1.5+2/3], color=pl_background, linewidth=3.5, alpha=alpha_val)
    ax2.plot([-9.5/12,9.5/12], [1.5+4/3,1.5+4/3], color=pl_background, linewidth=3.5, alpha=alpha_val)
    ax2.axvline(10/36, ymin=(sz_bot-y_bot+0.05)/(y_lim-1-y_bot), ymax=(sz_top-y_bot-0.05)/(y_lim-1-y_bot), color=pl_background, linewidth=3.5, alpha=alpha_val)
    ax2.axvline(-10/36, ymin=(sz_bot-y_bot+0.05)/(y_lim-1-y_bot), ymax=(sz_top-y_bot-0.05)/(y_lim-1-y_bot), color=pl_background, linewidth=3.5, alpha=alpha_val)
    ax2.plot([-10/12,10/12], [1.5+2/3,1.5+2/3], color=pl_white, linewidth=2, alpha=alpha_val,zorder=2)
    ax2.plot([-10/12,10/12], [1.5+4/3,1.5+4/3], color=pl_white, linewidth=2, alpha=alpha_val,zorder=2)
    ax2.axvline(10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot-0.025)/(y_lim-1-y_bot), color=pl_white, linewidth=3, alpha=alpha_val,zorder=2)
    ax2.axvline(-10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot-0.025)/(y_lim-1-y_bot), color=pl_white, linewidth=3, alpha=alpha_val,zorder=2)
    
    # Outer Strike Zone
    zone_outline_shadow = plt.Rectangle((-10/12, sz_bot), 20/12, 2,
                                 color=pl_background,fill=False,alpha=alpha_val, linewidth=3,zorder=1)
    ax2.add_patch(zone_outline_shadow)
    zone_outline = plt.Rectangle((-10/12, sz_bot), 20/12, 2, color=pl_white,fill=False,linewidth=2,alpha=alpha_val)
    ax2.add_patch(zone_outline)
    
    # Plate
    ax2.plot([-8.5/12,8.5/12], [plate_y,plate_y], color=pl_white, linewidth=2,zorder=0)
    ax2.plot([-8.5/12,-8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=2,zorder=0)
    ax2.plot([8.5/12,8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=2,zorder=0)
    ax2.plot([8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=2,zorder=0)
    ax2.plot([-8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=2,zorder=0)
    
    ax2.set(xlim=(-2,2),
           ylim=(y_bot,y_lim-1),
           aspect=1)
    ax2.axis('off')
    
    ax3 = fig.add_axes([0.7225,0.275,0.2675,0.287], anchor='SW', zorder=1)
    # Inner Strike zone
    ax3.plot([-9.5/12,9.5/12], [1.5+2/3,1.5+2/3], color=pl_background, linewidth=3.5, alpha=alpha_val)
    ax3.plot([-9.5/12,9.5/12], [1.5+4/3,1.5+4/3], color=pl_background, linewidth=3.5, alpha=alpha_val)
    ax3.axvline(10/36, ymin=(sz_bot-y_bot+0.05)/(y_lim-1-y_bot), ymax=(sz_top-y_bot-0.05)/(y_lim-1-y_bot), color=pl_background, linewidth=3.5, alpha=alpha_val)
    ax3.axvline(-10/36, ymin=(sz_bot-y_bot+0.05)/(y_lim-1-y_bot), ymax=(sz_top-y_bot-0.05)/(y_lim-1-y_bot), color=pl_background, linewidth=3.5, alpha=alpha_val)
    ax3.plot([-10/12,10/12], [1.5+2/3,1.5+2/3], color=pl_white, linewidth=2, alpha=alpha_val,zorder=2)
    ax3.plot([-10/12,10/12], [1.5+4/3,1.5+4/3], color=pl_white, linewidth=2, alpha=alpha_val,zorder=2)
    ax3.axvline(10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot-0.025)/(y_lim-1-y_bot), color=pl_white, linewidth=3, alpha=alpha_val,zorder=2)
    ax3.axvline(-10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot-0.025)/(y_lim-1-y_bot), color=pl_white, linewidth=3, alpha=alpha_val,zorder=2)
    
    # Outer Strike Zone
    zone_outline_shadow = plt.Rectangle((-10/12, sz_bot), 20/12, 2,
                                 color=pl_background,fill=False,alpha=alpha_val, linewidth=3,zorder=1)
    ax3.add_patch(zone_outline_shadow)
    zone_outline = plt.Rectangle((-10/12, sz_bot), 20/12, 2, color=pl_white,fill=False,linewidth=2,alpha=alpha_val)
    ax3.add_patch(zone_outline)
    
    
    sns.scatterplot(data=(game_df.loc[(game_df['hitterHand']=='R')].assign(pX = lambda x: x['pX']*-1)),
                    x='pX',
                    y='pZ',
                    hue='pitchType',
                    palette=marker_colors,
                    edgecolor=pl_background,
                    s=300,
                    linewidth=0.5,
                    alpha=1,
                    legend=False,
                   zorder=10,
                   ax=ax3)
    
    # Plate
    ax3.plot([-8.5/12,8.5/12], [plate_y,plate_y], color=pl_white, linewidth=2, zorder=0)
    ax3.plot([-8.5/12,-8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=2, zorder=0)
    ax3.plot([8.5/12,8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=2, zorder=0)
    ax3.plot([8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=2, zorder=0)
    ax3.plot([-8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=2, zorder=0)
    
    ax3.set(xlim=(-2,2),
           ylim=(y_bot,y_lim-1),
           aspect=1)
    ax3.axis('off')
    
    l_loc_ax = fig.add_axes([0.435,0.48,0.1,0.1], anchor='SW', zorder=1)
    l_loc_ax.text(0.5,0.7,'Locations',ha='center',va='center',fontsize=15,color=pl_line_color)
    l_loc_ax.text(0.5,0.45,vs_l_location_grade,ha='center',va='center',fontsize=50,
                  color=grade_colors[vs_l_location_grade])
    l_loc_ax.set(xlim=(0,1),ylim=(0,1))
    l_loc_ax.axis('off')
    
    r_loc_ax = fig.add_axes([0.7125,0.48,0.1,0.1], anchor='SW', zorder=1)
    
    r_loc_ax.text(0.5,0.7,'Locations',ha='center',va='center',fontsize=15,color=pl_line_color)
    r_loc_ax.text(0.5,0.45,vs_r_location_grade,ha='center',va='center',fontsize=50,
                  color=grade_colors[vs_r_location_grade])
    r_loc_ax.set(xlim=(0,1),ylim=(0,1))
    r_loc_ax.axis('off')
    
    
    fastball_ax = fig.add_axes([0.01,0.6,.425,0.082], anchor='SW', zorder=1)
    fastball_name, fastball_code = fastball_stats(game_group,fastball_ax)
    
    stat_header_ax = fig.add_axes([0.01,0.015,.25,0.205], anchor='SW', zorder=1)
    header_chunk(game_group,stat_header_ax)
    
    stat_table_ax = fig.add_axes([0.26,0.015,0.73,0.205], anchor='SW', zorder=1)
    stats_chunk(game_group.merge(szn_comp[['pitchType','Velo_diff']],how='left',on='pitchType') if vs_past else game_group,
                stat_table_ax,vs_past)
    
    usage_ax = fig.add_axes([0.445,0.593,.545,0.195], anchor='SW', zorder=1)
    usage_chunk(szn_comp if vs_past else game_group,usage_ax,vs_past)
    
    if fastball_name=='Four-Seam':
        fig.text(0.06,0.69,f'Primary Fastball:',color='w',fontsize=24,va='center',ha='left')
        fig.text(0.25,0.69,fastball_name,color=marker_colors[fastball_code],fontsize=28,va='center',ha='left')
        fig.add_artist(lines.Line2D([0.01, 0.04], [0.69, 0.69],linewidth=3,color=pl_text,alpha=0.75))
        fig.add_artist(lines.Line2D([0.41,0.4325], [0.69, 0.69],linewidth=3,color=pl_text,alpha=0.75))
    else:
        fig.text(0.093,0.69,f'Primary Fastball:',color='w',fontsize=24,va='center',ha='left')
        fig.text(0.283,0.69,fastball_name,color=marker_colors[fastball_code],fontsize=28,va='center',ha='left')
        fig.add_artist(lines.Line2D([0.01, 0.073], [0.69, 0.69],linewidth=3,color=pl_text,alpha=0.75))
        fig.add_artist(lines.Line2D([0.375,0.4325], [0.69, 0.69],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.01, 0.4325], [0.713, 0.713],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.01, 0.01], [0.59, 0.688],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.435, 0.435], [0.59, 0.813],linewidth=3,color=pl_text,alpha=0.75))
    
    #Grade Only
    fig.text(0.085,0.815,'Start', ha='center',va='center',color='w',fontsize=30)
    fig.text(0.2975,0.815,'Models', ha='center',va='center',color='w',fontsize=30)
    fig.add_artist(lines.Line2D([0.01, 0.03], [0.815, 0.815],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.14, 0.23], [0.815, 0.815],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.01, 0.01], [0.715, 0.813],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.16, 0.16], [0.715, 0.813],linewidth=3,color=pl_text,alpha=0.75))
    
    fig.text(0.5,0.9,game_text,color='w',fontsize=24,va='center',ha='center',font=italic)
    
    fig.add_artist(mpatches.FancyBboxPatch((0.02, 0.85), 0.96, 0.023,
                                           ec=pl_text,
                                           fc=pl_background,
                                           alpha=0.75,
                                           zorder=0,
                                           linewidth=3,
                                           boxstyle=mpatches.BoxStyle("Round", pad=0.2),
                                           mutation_scale=0.05
                                                  )
            )
    
    fig.text(0.715,0.815,'Usage',color='w',fontsize=30,va='center',ha='center')
    fig.add_artist(lines.Line2D([0.775, 0.99], [0.815, 0.815],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.365, 0.655], [0.815, 0.815],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.99, 0.99], [0.59, 0.813],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.01, 0.99], [0.588, 0.588],linewidth=3,color=pl_text,alpha=0.75))
    if vs_past:
        if prev_season:
            fig.text(0.98,0.595,'Arrows are vs\n2025 Usage',alpha=0.5,ha='right')
        else:
            fig.text(0.98,0.595,'Arrows are vs\nprior 2025 Usage',alpha=0.5,ha='right')
    
    # Left align movement
    fig.text(0.57375,0.565,'vs LHB',color='w',fontsize=30,va='center',ha='center')
    fig.text(0.85125,0.565,'vs RHB',color='w',fontsize=30,va='center',ha='center')
    fig.text(0.2225,0.565,'Movement',color='w',fontsize=30,va='center',ha='center')
    if vs_past:
        if prev_season:
            fig.text(0.02,0.285,'Shaded Regions\nare 2025 Shapes',alpha=0.5)
        else:
            fig.text(0.02,0.285,'Shaded Regions are\nprior 2025 Shapes',alpha=0.5)
    fig.add_artist(lines.Line2D([0.01, 0.12], [0.565, 0.565],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.325, 0.505], [0.565, 0.565],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.6425, 0.7825], [0.565, 0.565],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.92, 0.99], [0.565, 0.565],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.435, 0.435], [0.28, 0.563],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.7125, 0.7125], [0.28, 0.563],linewidth=3,color=pl_text,alpha=0.75))
    
    fig.add_artist(lines.Line2D([0.01, 0.01], [0.28, 0.565],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.99, 0.99], [0.28, 0.565],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.01, 0.99], [0.278, 0.278],linewidth=3,color=pl_text,alpha=0.75))
    
    fig.text(0.5,0.255,'Pitch Type Metrics',color='w',fontsize=30,va='center',ha='center')
    fig.add_artist(lines.Line2D([0.01, 0.35], [0.255, 0.255],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.65, 0.99], [0.255, 0.255],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.01, 0.01], [0.017, 0.253],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.99, 0.99], [0.017, 0.253],linewidth=3,color=pl_text,alpha=0.75))
    fig.add_artist(lines.Line2D([0.01, 0.99], [0.015, 0.015],linewidth=3,color=pl_text,alpha=0.75))
    
    fig.add_artist(lines.Line2D([0, 1, 1, 0], [1, 1, 0, 0],linewidth=3,color='w',alpha=0))
    
    grid.tight_layout(fig,pad=2)
    sns.despine(left=True,bottom=True)
    st.pyplot(fig, width='content')
if st.button('Generate Chart'):
    game_df, game_group, szn_df, szn_group, szn_comp = load_data(pitcher_id,game_id,vs_past,szn_load)
    generate_chart(pitcher_id,game_id,game_df,game_group,szn_df,szn_comp,vs_past)
