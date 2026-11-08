import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Tipster Pro AI", page_icon="⚽", layout="centered")

st.title("⚽ Motore Tipster Totale - Pro Edition")
st.write("Analisi intelligente dei match basata su machine learning e forza degli avversari.")

# --- 1. DOWNLOAD DATI (Dal tuo primo codice) ---
@st.cache_data(ttl=3600) # Salva in cache per non riscaricare a ogni click
def load_data():
    leagues = {
        'Serie_A': 'I1', 'Serie_B': 'I2',
        'Premier_League': 'E0', 'Championship': 'E1', 'League_One': 'E2', 'League_Two': 'E3', 'National_League': 'EC',
        'La_Liga': 'SP1', 'Segunda_Division': 'SP2',
        'Bundesliga': 'D1', 'Bundesliga_2': 'D2',
        'Ligue_1': 'F1', 'Ligue_2': 'F2',
        'Eredivisie_Olanda': 'N1', 'Jupiler_League_Belgio': 'B1', 'Liga_Portugal': 'P1',
        'Turchia_Super_Lig': 'T1', 'Grecia_Super_League': 'G1',
        'Scozia_Premiership': 'SC0', 'Scozia_Championship': 'SC1', 'Scozia_League_One': 'SC2', 'Scozia_League_Two': 'SC3'
    }
    seasons = ['2425', '2526', '2627']
    all_data = []

    for league_name, league_code in leagues.items():
        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
            try:
                df = pd.read_csv(url, on_bad_lines='skip')
                df['League'] = league_name
                df['Season'] = season
                all_data.append(df)
            except Exception:
                pass

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        master_df['Date'] = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce')
        master_df = master_df.sort_values('Date').reset_index(drop=True)
        if 'Home_xG' not in master_df.columns: master_df['Home_xG'] = 0.0
        if 'Away_xG' not in master_df.columns: master_df['Away_xG'] = 0.0
        return master_df
    return pd.DataFrame()

with st.spinner("⏳ Caricamento e aggiornamento database in corso..."):
    master_df = load_data()

# --- 2. ELABORAZIONE STATISTICHE (Dal tuo secondo codice) ---
def compute_smart_pro_stats(df):
    team_history = {}
    team_season_stats = {}

    h_gf, h_ga, h_xg, h_corners, h_cards, h_ht, h_pts = [], [], [], [], [], [], []
    a_gf, a_ga, a_xg, a_corners, a_cards, a_ht, a_pts = [], [], [], [], [], [], []

    df = df.sort_values('Date').reset_index(drop=True)

    for idx, row in df.iterrows():
        h_team, a_team = row['HomeTeam'], row['AwayTeam']
        season = row['Season']
        h_key, a_key = (season, h_team), (season, a_team)

        if h_key not in team_season_stats: team_season_stats[h_key] = {'gf': [], 'ga': [], 'points': 0, 'matches': 0}
        if a_key not in team_season_stats: team_season_stats[a_key] = {'gf': [], 'ga': [], 'points': 0, 'matches': 0}

        if not pd.isna(row['FTHG']) and not pd.isna(row['FTAG']):
            fthg, ftag = row['FTHG'], row['FTAG']
            team_season_stats[h_key]['gf'].append(fthg)
            team_season_stats[h_key]['ga'].append(ftag)
            team_season_stats[a_key]['gf'].append(ftag)
            team_season_stats[a_key]['ga'].append(fthg)
            team_season_stats[h_key]['matches'] += 1
            team_season_stats[a_key]['matches'] += 1
            if fthg > ftag: team_season_stats[h_key]['points'] += 3
            elif fthg < ftag: team_season_stats[a_key]['points'] += 3
            else:
                team_season_stats[h_key]['points'] += 1
                team_season_stats[a_key]['points'] += 1

    def get_opponent_multiplier(opp_key, stats_dict):
        if opp_key not in stats_dict or stats_dict[opp_key]['matches'] < 3:
            return 1.0
        st_val = stats_dict[opp_key]
        avg_pts = st_val['points'] / st_val['matches']
        if avg_pts >= 1.8: return 1.15
        elif avg_pts <= 1.0: return 0.90
        return 1.0

    for idx, row in df.iterrows():
        h_team, a_team = row['HomeTeam'], row['AwayTeam']
        season = row['Season']
        h_key, a_key = (season, h_team), (season, a_team)

        for key in [h_key, a_key]:
            if key not in team_history:
                team_history[key] = {'matches': []}

        h_current_pts = sum([m['pts'] for m in team_history[h_key]['matches']]) if team_history[h_key]['matches'] else 0
        a_current_pts = sum([m['pts'] for m in team_history[a_key]['matches']]) if team_history[a_key]['matches'] else 0

        h_pts.append(h_current_pts)
        a_pts.append(a_current_pts)

        def calc_weighted_rolling(history_list):
            if not history_list:
                return 1.3, 1.1, 1.4, 5.0, 2.0, 0.5
            last_5 = history_list[-5:]
            gf_w = np.mean([m['gf'] * m['opp_mult'] for m in last_5])
            ga_w = np.mean([m['ga'] / m['opp_mult'] for m in last_5])
            xg_w = np.mean([m['xg'] * m['opp_mult'] for m in last_5])
            corn_w = np.mean([m['corners'] for m in last_5])
            card_w = np.mean([m['cards'] for m in last_5])
            ht_w = np.mean([m['ht'] for m in last_5])
            return gf_w, ga_w, xg_w, corn_w, card_w, ht_w

        h_vals = calc_weighted_rolling(team_history[h_key]['matches'])
        a_vals = calc_weighted_rolling(team_history[a_key]['matches'])

        h_gf.append(h_vals[0]); h_ga.append(h_vals[1]); h_xg.append(h_vals[2])
        h_corners.append(h_vals[3]); h_cards.append(h_vals[4]); h_ht.append(h_vals[5])
        a_gf.append(a_vals[0]); a_ga.append(a_vals[1]); a_xg.append(a_vals[2])
        a_corners.append(a_vals[3]); a_cards.append(a_vals[4]); a_ht.append(a_vals[5])

        if not pd.isna(row['FTHG']) and not pd.isna(row['FTAG']):
            fthg, ftag = row['FTHG'], row['FTAG']
            hc, ac = row['HC'], row['AC']
            hy, ay = row['HY'], row['AY']
            h_ht_g, a_ht_g = row['HTHG'], row['HTAG']

            h_opp_mult = get_opponent_multiplier(a_key, team_season_stats)
            a_opp_mult = get_opponent_multiplier(h_key, team_season_stats)

            h_match_pts = 3 if fthg > ftag else (1 if fthg == ftag else 0)
            a_match_pts = 3 if ftag > fthg else (1 if fthg == ftag else 0)

            team_history[h_key]['matches'].append({'gf': fthg, 'ga': ftag, 'xg': row['Home_xG'], 'corners': hc, 'cards': hy, 'ht': h_ht_g, 'opp_mult': h_opp_mult, 'pts': h_match_pts})
            team_history[a_key]['matches'].append({'gf': ftag, 'ga': fthg, 'xg': row['Away_xG'], 'corners': ac, 'cards': ay, 'ht': a_ht_g, 'opp_mult': a_opp_mult, 'pts': a_match_pts})

    df['Home_GF_roll5'] = h_gf; df['Home_GA_roll5'] = h_ga; df['Home_xG_roll5'] = h_xg
    df['Home_Corners_roll5'] = h_corners; df['Home_Cards_roll5'] = h_cards; df['Home_HT_roll5'] = h_ht; df['Home_Points'] = h_pts
    df['Away_GF_roll5'] = a_gf; df['Away_GA_roll5'] = a_ga; df['Away_xG_roll5'] = a_xg
    df['Away_Corners_roll5'] = a_corners; df['Away_Cards_roll5'] = a_cards; df['Away_HT_roll5'] = a_ht; df['Away_Points'] = a_pts
    df['Diff_Points'] = df['Home_Points'] - df['Away_Points']
    return df

with st.spinner("⚙️ Elaborazione metriche avanzate..."):
    processed_df = compute_smart_pro_stats(master_df)

# --- 3. ADDESTRAMENTO MODELLI IA ---
if 'Target_1X2' not in processed_df.columns:
    processed_df['Target_1X2'] = processed_df.apply(lambda r: 0 if r['FTHG'] > r['FTAG'] else (1 if r['FTHG'] == r['FTAG'] else 2), axis=1)
if 'Target_Over25' not in processed_df.columns:
    processed_df['Target_Over25'] = ((processed_df['FTHG'] + processed_df['FTAG']) > 2.5).astype(int)
if 'Target_GG' not in processed_df.columns:
    processed_df['Target_GG'] = ((processed_df['FTHG'] > 0) & (processed_df['FTAG'] > 0)).astype(int)

features = [
    'Home_GF_roll5', 'Home_GA_roll5', 'Home_xG_roll5', 'Home_Corners_roll5', 'Home_Cards_roll5', 'Home_HT_roll5',
    'Away_GF_roll5', 'Away_GA_roll5', 'Away_xG_roll5', 'Away_Corners_roll5', 'Away_Cards_roll5', 'Away_HT_roll5',
    'Diff_xG', 'Diff_GF_GA', 'Diff_Points'
]

train_df = processed_df.dropna(subset=['Home_GF_roll5', 'Away_GF_roll5', 'Target_1X2']).copy()
train_df['Diff_xG'] = train_df['Home_xG_roll5'] - train_df['Away_xG_roll5']
train_df['Diff_GF_GA'] = train_df['Home_GF_roll5'] - train_df['Away_GA_roll5']
X = train_df[features]

model_1x2 = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8).fit(X, train_df['Target_1X2'])
model_over25 = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8).fit(X, train_df['Target_Over25'])
model_gg = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8).fit(X, train_df['Target_GG'])

# --- 4. INTERFACCIA GRAFICA STREAMLIT ---
st.success("✅ Modelli IA pronti all'uso!")

leagues_list = sorted(processed_df['League'].unique())
selected_league = st.selectbox("🏆 Seleziona Campionato:", leagues_list)

def get_teams_for_league(league_name):
    df_league = processed_df[processed_df['League'] == league_name]
    if df_league.empty: return []
    latest_season = df_league['Season'].max()
    df_latest = df_league[df_league['Season'] == latest_season]
    return sorted(list(set(df_latest['HomeTeam'].unique()).union(set(df_latest['AwayTeam'].unique()))))

teams = get_teams_for_league(selected_league)
col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("🏠 Squadra di Casa:", teams, index=0 if len(teams) > 0 else 0)
with col2:
    a_team = st.selectbox("✈️ Squadra Ospite:", teams, index=1 if len(teams) > 1 else 0)

def get_latest_pro_stats(df, team):
    team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    if team_matches.empty: return 1.3, 1.1, 1.4, 5.0, 2.0, 0.5, 0
    last_match = team_matches.iloc[-1]
    if last_match['HomeTeam'] == team:
        return last_match['Home_GF_roll5'], last_match['Home_GA_roll5'], last_match['Home_xG_roll5'], last_match['Home_Corners_roll5'], last_match['Home_Cards_roll5'], last_match['Home_HT_roll5'], last_match['Home_Points']
    else:
        return last_match['Away_GF_roll5'], last_match['Away_GA_roll5'], last_match['Away_xG_roll5'], last_match['Away_Corners_roll5'], last_match['Away_Cards_roll5'], last_match['Away_HT_roll5'], last_match['Away_Points']

if st.button("🚀 Genera Analisi Tipster Pro", type="primary"):
    if h_team == a_team:
        st.error("❌ Seleziona due squadre diverse!")
    else:
        league_df = processed_df[processed_df['League'] == selected_league]
        h_gf, h_ga, h_xg, h_corn, h_card, h_ht, h_pts = get_latest_pro_stats(league_df, h_team)
        a_gf, a_ga, a_xg, a_corn, a_card, a_ht, a_pts = get_latest_pro_stats(league_df, a_team)

        diff_xg = h_xg - a_xg
        diff_gf_ga = h_gf - a_ga
        diff_points = h_pts - a_pts

        sample_input = [[h_gf, h_ga, h_xg, h_corn, h_card, h_ht, a_gf, a_ga, a_xg, a_corn, a_card, a_ht, diff_xg, diff_gf_ga, diff_points]]

        p_1x2 = model_1x2.predict_proba(sample_input)[0]
        p_o25 = model_over25.predict_proba(sample_input)[0]
        p_gg = model_gg.predict_proba(sample_input)[0]

        p_ov, p_un, p_g = p_o25[1], p_o25[0], p_gg[1]
        expected_corners = h_corn + a_corn
        expected_cards = h_card + a_card

        consigli_mercato = []
        mercati_validi = []

        max_1x2 = max(p_1x2)
        fav_1x2_idx = np.argmax(p_1x2)
        if max_1x2 >= 0.62:
            fav_1x2_str = "1 (Vittoria Casa)" if fav_1x2_idx == 0 else ("X (Pareggio)" if fav_1x2_idx == 1 else "2 (Vittoria Ospite)")
            consigli_mercato.append(f"**1X2**: 🟢 {fav_1x2_str} ({max_1x2*100:.1f}%)")
            mercati_validi.append(fav_1x2_str)
        else:
            consigli_mercato.append(f"**1X2**: ⚠️ Sotto soglia (Max: {max_1x2*100:.1f}%)")

        if p_ov >= 0.62:
            consigli_mercato.append(f"**Gol Totali**: 🟢 **Over 2.5** ({p_ov*100:.1f}%)")
            mercati_validi.append("Over 2.5")
        elif p_un >= 0.62:
            consigli_mercato.append(f"**Gol Totali**: 🔴 **Under 2.5** ({p_un*100:.1f}%)")
            mercati_validi.append("Under 2.5")
        else:
            consigli_mercato.append(f"**Gol Totali**: ⚪ Incerto (Over: {p_ov*100:.1f}%)")

        if p_g >= 0.62:
            consigli_mercato.append(f"**Entrambe a Segno**: 🟢 **GOAL (Sì)** ({p_g*100:.1f}%)")
            mercati_validi.append("GOAL (Sì)")
        elif (1 - p_g) >= 0.62:
            consigli_mercato.append(f"**Entrambe a Segno**: 🔴 **NO GOAL** ({(1-p_g)*100:.1f}%)")
            mercati_validi.append("NO GOAL")
        else:
            consigli_mercato.append(f"**Entrambe a Segno**: ⚪ Incerto (GG: {p_g*100:.1f}%)")

        if expected_corners >= 10.5:
            consigli_mercato.append(f"**Calci d'Angolo**: 🟢 **Over 9.5 Corner** (Stimati: {expected_corners:.1f})")
            mercati_validi.append("Over 9.5 Corner")
        elif expected_corners <= 8.0:
            consigli_mercato.append(f"**Calci d'Angolo**: 🔴 **Under 10.5 Corner** (Stimati: {expected_corners:.1f})")
            mercati_validi.append("Under 10.5 Corner")
        else:
            consigli_mercato.append(f"**Calci d'Angolo**: ⚪ Linea neutra (Stimati: {expected_corners:.1f})")

        if expected_cards >= 5.2:
            consigli_mercato.append(f"**Cartellini**: 🟢 **Over 4.5 Cartellini** (Stimati: {expected_cards:.1f})")
            mercati_validi.append("Over 4.5 Cartellini")
        else:
            consigli_mercato.append(f"**Cartellini**: ⚪ Sotto soglia (Stimati: {expected_cards:.1f})")

        st.markdown("---")
        st.subheader(f"📊 Risultato Analisi: {h_team} vs {a_team}")
        
        for c in consigli_mercato:
            st.write(f"- {c}")

        if len(mercati_validi) == 0:
            st.error("VERDETTO: NO BET (Nessun mercato raggiunge i filtri di qualità)")
        else:
            segno_scelto = " + ".join(mercati_validi[:2])
            st.success(f"VERDETTO TOP TIPSTER ➔ Consigliato: [{segno_scelto}]")

        with st.expander("📈 Visualizza percentuali pure del modello"):
            st.write(f"- **1X2** -> 1: {p_1x2[0]*100:.1f}% | X: {p_1x2[1]*100:.1f}% | 2: {p_1x2[2]*100:.1f}%")
            st.write(f"- **Over 2.5**: {p_ov*100:.1f}% | Under 2.5: {p_un*100:.1f}%")
            st.write(f"- **Goal (GG)**: {p_g*100:.1f}% | No Goal: {(1-p_g)*100:.1f}%")