"""
Roda o modelo treinado em um draft novo (ainda sem resultado) e devolve a
probabilidade estimada de vitória do Team1.

Formato esperado do JSON de entrada (--config):

{
  "league": "LPL",
  "team1_picks": ["Renata Glasc", "Kindred", "Azir", "Kalista", "Poppy"],
  "team2_picks": ["Jarvan IV", "Viego", "Orianna", "Xayah", "Braum"]
}
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODEL_DIR = DATA_DIR / "models"


def load_model(league: str):
    payload = joblib.load(MODEL_DIR / f"model_{league}.joblib")
    return payload["model"], payload["columns"]


def vectorize_single(team1_picks, team2_picks, columns):
    row = {col: 0 for col in columns}
    for col in columns:
        if col.startswith("champ_") and col.endswith("_t1"):
            champ = col[len("champ_"):-len("_t1")]
            row[col] = int(champ in team1_picks)
        elif col.startswith("champ_") and col.endswith("_t2"):
            champ = col[len("champ_"):-len("_t2")]
            row[col] = int(champ in team2_picks)
    return pd.DataFrame([row], columns=columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON com o draft a prever")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    league = config["league"]

    model, columns = load_model(league)
    X = vectorize_single(config["team1_picks"], config["team2_picks"], columns)

    proba_team1 = model.predict_proba(X)[0, 1]
    print(f"Probabilidade estimada de vitória do Team1: {proba_team1:.1%}")
    print(f"Probabilidade estimada de vitória do Team2: {1 - proba_team1:.1%}")
    print(
        "\nPara achar valor de aposta, compare essa probabilidade com a "
        "probabilidade implícita da odds da casa (1 / odd_decimal). Se a sua "
        "probabilidade estimada for maior que a implícita por uma margem "
        "consistente, há indício de edge positivo."
    )


if __name__ == "__main__":
    main()
