"""
Treina um modelo de regressão para prever o total de kills (Team1Kills +
Team2Kills) de uma partida a partir do draft — mesmas features de
build_features.py (composição de picks + win rate histórico dos campeões).

Requer que os jogos tenham sido enriquecidos com kills antes:
    python fetch_golgg.py --league LPL --add-kills

Usa LightGBM (regressão) com validação cruzada por tempo, pelo mesmo motivo
de train.py: dados de esports têm forte componente temporal.
"""

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "data" / "models"


def train(league: str):
    features_path = DATA_DIR / f"features_{league}.csv"
    df = pd.read_csv(features_path)

    df = df.dropna(subset=["total_kills"])
    if df.empty:
        raise SystemExit(
            f"Nenhum jogo de {league} tem dado de kills ainda. Rode primeiro:\n"
            f"  python fetch_golgg.py --league {league} --add-kills\n"
            f"  python build_features.py --league {league}"
        )

    y = df["total_kills"]
    X = df.drop(columns=["team1_win", "game_id", "total_kills"])

    tscv = TimeSeriesSplit(n_splits=5)
    metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        metrics.append({"fold": fold, "mae": mae, "media_real": float(y_test.mean())})
        print(f"fold {fold}: MAE={mae:.2f} kills  (média real do fold: {y_test.mean():.1f})")

    # Modelo final treinado em tudo, para uso em produção
    final_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10
    )
    final_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"model_kills_{league}.joblib"
    joblib.dump({"model": final_model, "columns": list(X.columns)}, model_path)

    metrics_path = MODEL_DIR / f"metrics_kills_{league}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nModelo de kills salvo em {model_path}")
    print(f"Métricas de validação salvas em {metrics_path}")
    print(
        "\nIMPORTANTE: o erro médio (MAE) mostra o quanto a previsão erra em média, "
        "em número de kills — use como referência de incerteza, não como valor exato."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="LPL")
    args = parser.parse_args()
    train(args.league)


if __name__ == "__main__":
    main()
