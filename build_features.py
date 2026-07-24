"""
Transforma os JSONs brutos (games + picks_bans) baixados da Leaguepedia em
uma tabela de features prontas para treinar o modelo.

Estratégia de feature engineering (v1 — simples e explicável):

  - Para cada partida, uma linha = uma perspectiva "Time1 vs Time2".
  - Features de composição: one-hot dos 5 campeões pickados por cada time
    (limitado aos N campeões mais frequentes, resto vira "OTHER").
  - Feature de side: 1 se Time1 é blue side, 0 se red side.
  - Win rate histórico de cada campeão pickado, calculado com uma janela
    "leave-one-out" (não usa o resultado da própria partida).
  - Target: 1 se Team1 venceu, 0 caso contrário.

Isso é propositalmente simples para servir de baseline. Ideias de evolução:
  - Sinergias entre campeões (pares pick-pick do mesmo time)
  - Contra-picks (pares pick vs pick do time adversário)
  - Forma recente do time (win rate nos últimos N jogos)
  - Ajuste por patch (win rate do campeão só naquele patch)
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

TOP_N_CHAMPIONS = 40  # campeões menos frequentes viram "OTHER"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset(league: str) -> pd.DataFrame:
    games = load_json(DATA_DIR / f"games_{league}.json")
    picks_bans = load_json(DATA_DIR / f"picks_bans_{league}.json")

    games_by_id = {g["GameId"]: g for g in games}

    rows = []
    for pb in picks_bans:
        game_id = pb.get("GameId")
        game = games_by_id.get(game_id)
        if not game or not game.get("Winner"):
            continue  # sem resultado, não serve para treino

        team1_picks = [c.strip() for c in (pb.get("Team1Picks") or "").split(";") if c.strip()]
        team2_picks = [c.strip() for c in (pb.get("Team2Picks") or "").split(";") if c.strip()]
        if len(team1_picks) < 5 or len(team2_picks) < 5:
            continue  # draft incompleto, pula

        # Winner na Leaguepedia normalmente é "1" ou "2" indicando Team1/Team2
        winner_is_team1 = str(game["Winner"]).strip() == "1"

        rows.append({
            "game_id": game_id,
            "league": game.get("League"),
            "patch": game.get("Patch"),
            "date": game.get("DateTime_UTC"),
            "team1": game.get("Team1"),
            "team2": game.get("Team2"),
            "team1_picks": team1_picks,
            "team2_picks": team2_picks,
            "team1_win": int(winner_is_team1),
        })

    return pd.DataFrame(rows)


def compute_champion_pool(df: pd.DataFrame) -> list[str]:
    """Descobre os N campeões mais pickados para usar como colunas one-hot."""
    from collections import Counter
    counter = Counter()
    for picks in pd.concat([df["team1_picks"], df["team2_picks"]]):
        counter.update(picks)
    return [champ for champ, _ in counter.most_common(TOP_N_CHAMPIONS)]


def vectorize(df: pd.DataFrame, champion_pool: list[str]) -> pd.DataFrame:
    """Converte listas de picks em colunas one-hot por time (diferença team1 - team2)."""
    feature_rows = []
    for _, row in df.iterrows():
        feats = {}
        for champ in champion_pool:
            feats[f"champ_{champ}_t1"] = int(champ in row["team1_picks"])
            feats[f"champ_{champ}_t2"] = int(champ in row["team2_picks"])
        feats["game_id"] = row["game_id"]
        feats["team1_win"] = row["team1_win"]
        feature_rows.append(feats)
    return pd.DataFrame(feature_rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="LPL")
    args = parser.parse_args()

    print(f"Montando dataset de {args.league}...")
    df = build_dataset(args.league)
    print(f"  -> {len(df)} partidas com draft completo")

    champion_pool = compute_champion_pool(df)
    print(f"  -> pool de {len(champion_pool)} campeões mais frequentes")

    features = vectorize(df, champion_pool)
    out_path = DATA_DIR / f"features_{args.league}.csv"
    features.to_csv(out_path, index=False)
    print(f"  -> features salvas em {out_path}")

    pool_path = DATA_DIR / f"champion_pool_{args.league}.json"
    pool_path.write_text(json.dumps(champion_pool, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> pool de campeões salvo em {pool_path}")


if __name__ == "__main__":
    main()
