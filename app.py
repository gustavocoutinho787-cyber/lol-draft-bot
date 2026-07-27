"""
Interface web local (Streamlit) para prever a probabilidade de vitória de um
draft, sem precisar editar JSON nem usar o terminal.

Rodar com:
    streamlit run app.py

Abre automaticamente no navegador em http://localhost:8501 — só funciona
enquanto esse comando estiver rodando no seu computador.
"""

import json
from pathlib import Path

import streamlit as st

from predict import load_champion_winrates, load_model, vectorize_single

DATA_DIR = Path(__file__).resolve().parent / "data"
LEAGUES = ["LPL", "LCK", "LEC", "CBLOL"]


@st.cache_resource
def get_model(league: str):
    return load_model(league)


@st.cache_data
def get_champion_list(league: str) -> list[str]:
    winrates = load_champion_winrates(league)
    return sorted(winrates.keys())


st.set_page_config(page_title="LoL Draft Bot", page_icon="🎮")
st.title("🎮 LoL Draft Bot")
st.caption("Previsão de vitória a partir do draft — estimativa estatística, não garantia.")

league = st.selectbox("Liga", LEAGUES)

try:
    model, columns = get_model(league)
    champion_winrates = load_champion_winrates(league)
except FileNotFoundError:
    st.error(
        f"Não encontrei um modelo treinado para {league}. "
        f"Rode `python build_features.py --league {league}` e "
        f"`python train.py --league {league}` primeiro."
    )
    st.stop()

champion_options = get_champion_list(league)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Time 1")
    team1_picks = [
        st.selectbox(f"Pick {i+1} (Time 1)", champion_options, key=f"t1_{i}")
        for i in range(5)
    ]
with col2:
    st.subheader("Time 2")
    team2_picks = [
        st.selectbox(f"Pick {i+1} (Time 2)", champion_options, key=f"t2_{i}")
        for i in range(5)
    ]

if st.button("Prever", type="primary"):
    if len(set(team1_picks)) < 5 or len(set(team2_picks)) < 5 or set(team1_picks) & set(team2_picks):
        st.warning("Cada campeão só pode ser escolhido uma vez no draft inteiro (sem repetir entre os times).")
    else:
        X = vectorize_single(team1_picks, team2_picks, columns, champion_winrates)
        proba_team1 = model.predict_proba(X)[0, 1]

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Time 1", f"{proba_team1:.1%}")
        c2.metric("Time 2", f"{1 - proba_team1:.1%}")
        st.progress(float(proba_team1))
        st.caption(
            "Para achar valor de aposta, compare essa probabilidade com a probabilidade "
            "implícita da odds da casa (1 / odd_decimal). Só há indício de edge positivo "
            "se a sua probabilidade estimada for maior que a implícita por uma margem consistente."
        )
