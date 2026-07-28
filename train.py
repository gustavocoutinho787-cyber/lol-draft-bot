"""
Treina um modelo de previsão de vitória (team1_win) a partir das features de
draft geradas em build_features.py.

Usa LightGBM (gradient boosting) com validação cruzada por tempo, já que dados
de esports têm forte componente temporal (meta muda a cada patch) — treinar
com dados futuros para prever o passado infla a acurácia de forma irreal.
"""

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "data" / "models"


def train(league: str):
    features_path = DATA_DIR / f"features_{league}.csv"
    df = pd.read_csv(features_path)

    y = df["team1_win"]
    # total_kills é um resultado da partida (como team1_win), não uma feature —
    # incluí-lo como entrada vazaria informação pós-jogo para o modelo.
    X = df.drop(columns=["team1_win", "game_id", "total_kills"])

    tscv = TimeSeriesSplit(n_splits=5)
    metrics = []

    model = None
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
        )
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds > 0.5)
        ll = log_loss(y_test, preds)
        brier = brier_score_loss(y_test, preds)
        metrics.append({"fold": fold, "accuracy": acc, "log_loss": ll, "brier": brier})
        print(f"fold {fold}: acc={acc:.3f}  log_loss={ll:.3f}  brier={brier:.3f}")

    # Modelo final treinado em tudo, para uso em produção
    final_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10
    )
    final_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"model_{league}.joblib"
    joblib.dump({"model": final_model, "columns": list(X.columns)}, model_path)

    metrics_path = MODEL_DIR / f"metrics_{league}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nModelo salvo em {model_path}")
    print(f"Métricas de validação salvas em {metrics_path}")
    print(
        "\nIMPORTANTE: acurácia por si só não diz se há valor de aposta — "
        "o que importa é comparar a probabilidade prevista com a probabilidade "
        "implícita da odds (1/odd), e só apostar quando houver edge positivo "
        "consistente ao longo do tempo."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="LPL")
    args = parser.parse_args()
    train(args.league)


if __name__ == "__main__":
    main()
