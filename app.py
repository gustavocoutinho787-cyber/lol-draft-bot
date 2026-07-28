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

from predict import load_champion_winrates, load_kills_model, load_model, vectorize_single

DATA_DIR = Path(__file__).resolve().parent / "data"
LEAGUES = ["LPL", "LCK", "LEC", "CBLOL"]
ROLES = ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]
ROLE_LABELS = {"TOP": "Top", "JUNGLE": "Jungle", "MID": "Mid", "BOT": "ADC", "SUPPORT": "Support"}


@st.cache_resource
def get_model(league: str):
    return load_model(league)


@st.cache_resource
def get_kills_model(league: str):
    return load_kills_model(league)


@st.cache_data
def get_champion_list(league: str) -> list[str]:
    winrates = load_champion_winrates(league)
    return sorted(winrates.keys())


@st.cache_data
def get_champion_roles(league: str) -> dict[str, list[str]]:
    path = DATA_DIR / f"champion_roles_{league}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_champions_for_role(all_champions: list[str], roles_by_champ: dict[str, list[str]], role: str) -> list[str]:
    """Coloca os campeões mais comuns dessa rota primeiro na lista (mais fácil
    de achar o padrão), mas SEMPRE mantém todos os campeões disponíveis —
    a rota vem de um recorte antigo de dados e não deve travar picks fora do
    padrão (ex: mago no ADC, que virou comum na meta atual)."""
    if not roles_by_champ:
        return all_champions
    matched = [c for c in all_champions if role in roles_by_champ.get(c, [])]
    rest = [c for c in all_champions if c not in matched]
    return sorted(matched) + sorted(rest)


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
champion_roles = get_champion_roles(league)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Time 1")
    team1_picks = [
        st.selectbox(
            ROLE_LABELS[role] + " (Time 1)",
            get_champions_for_role(champion_options, champion_roles, role),
            key=f"t1_{role}",
        )
        for role in ROLES
    ]
with col2:
    st.subheader("Time 2")
    team2_picks = [
        st.selectbox(
            ROLE_LABELS[role] + " (Time 2)",
            get_champions_for_role(champion_options, champion_roles, role),
            key=f"t2_{role}",
        )
        for role in ROLES
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

        kills_model, kills_columns = get_kills_model(league)
        if kills_model is not None:
            X_kills = vectorize_single(team1_picks, team2_picks, kills_columns, champion_winrates)
            total_kills = kills_model.predict(X_kills)[0]
            st.metric("Total de kills estimado no jogo", f"{total_kills:.1f}")

        st.caption(
            "Para achar valor de aposta, compare essa probabilidade com a probabilidade "
            "implícita da odds da casa (1 / odd_decimal). Só há indício de edge positivo "
            "se a sua probabilidade estimada for maior que a implícita por uma margem consistente."
        )
