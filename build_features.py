"""
Transforma os JSONs brutos (games + picks_bans) baixados da Leaguepedia em
uma tabela de features prontas para treinar o modelo.

Estratégia de feature engineering (v1 — simples e explicável):

  - Para cada partida, uma linha = uma perspectiva "Time1 vs Time2".
  - Features de composição: one-hot dos 5 campeões pickados por cada time
    (limitado aos N campeões mais frequentes, resto vira "OTHER").
  - Win rate histórico médio dos campeões de cada time, numa janela expansiva
    (só usa jogos ANTERIORES em ordem cronológica — não vaza resultado futuro).
  - Target (vitória): 1 se Team1 venceu, 0 caso contrário.
  - Target (kills, opcional): total_kills = Team1Kills + Team2Kills, só
    disponível para jogos onde rodou `fetch_golgg.py --add-kills` (fica NaN
    nos outros) — usado por train_kills.py para prever kills totais do jogo.

Isso é propositalmente simples para servir de baseline. Ideias de evolução:
  - Sinergias entre campeões (pares pick-pick do mesmo time)
  - Contra-picks (pares pick vs pick do time adversário)
  - Feature de side (blue/red) — não incluída ainda porque nem toda fonte de
    dados informa o lado (a Leaguepedia não expõe isso nos campos atuais)
  - Ajuste por patch (win rate do campeão só naquele patch)
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

TOP_N_CHAMPIONS = 40  # campeões menos frequentes viram "OTHER"
WINRATE_PRIOR_WEIGHT = 15  # suaviza a win rate de campeões com poucos jogos observados


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

        team1_kills = game.get("Team1Kills")
        team2_kills = game.get("Team2Kills")
        total_kills = (
            team1_kills + team2_kills if team1_kills is not None and team2_kills is not None else None
        )

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
            "total_kills": total_kills,
        })

    df = pd.DataFrame(rows)
    # Ordem cronológica é essencial: a win rate expansiva e o TimeSeriesSplit
    # do train.py só fazem sentido se as linhas estiverem em ordem de data.
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df


def compute_expanding_champion_winrate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Para cada linha, calcula a win rate média dos campeões de cada time usando
    só jogos ANTERIORES (janela expansiva) — evita vazar o resultado da própria
    partida ou de partidas futuras. Suaviza com um prior de 50% para campeões
    com poucas observações (WINRATE_PRIOR_WEIGHT).

    Devolve o df com as colunas team1_champ_wr_avg/team2_champ_wr_avg preenchidas,
    e um dicionário {campeão: win_rate_final} para usar depois em predict.py.
    """
    wins: dict[str, int] = {}
    games: dict[str, int] = {}

    def smoothed_wr(champ: str) -> float:
        w = wins.get(champ, 0)
        g = games.get(champ, 0)
        return (w + WINRATE_PRIOR_WEIGHT * 0.5) / (g + WINRATE_PRIOR_WEIGHT)

    team1_wr_avg = []
    team2_wr_avg = []
    for _, row in df.iterrows():
        team1_wr_avg.append(sum(smoothed_wr(c) for c in row["team1_picks"]) / len(row["team1_picks"]))
        team2_wr_avg.append(sum(smoothed_wr(c) for c in row["team2_picks"]) / len(row["team2_picks"]))

        team1_win = row["team1_win"]
        for c in row["team1_picks"]:
            games[c] = games.get(c, 0) + 1
            wins[c] = wins.get(c, 0) + team1_win
        for c in row["team2_picks"]:
            games[c] = games.get(c, 0) + 1
            wins[c] = wins.get(c, 0) + (1 - team1_win)

    df = df.copy()
    df["team1_champ_wr_avg"] = team1_wr_avg
    df["team2_champ_wr_avg"] = team2_wr_avg

    final_winrates = {c: smoothed_wr(c) for c in games}
    return df, final_winrates


def compute_champion_pool(df: pd.DataFrame) -> list[str]:
    """Descobre os N campeões mais pickados para usar como colunas one-hot."""
    from collections import Counter
    counter = Counter()
    for picks in pd.concat([df["team1_picks"], df["team2_picks"]]):
        counter.update(picks)
    return [champ for champ, _ in counter.most_common(TOP_N_CHAMPIONS)]


def vectorize(df: pd.DataFrame, champion_pool: list[str]) -> pd.DataFrame:
    """Converte listas de picks em colunas one-hot por time (diferença team1 - team2),
    mais as features numéricas de win rate histórico dos campeões de cada time."""
    feature_rows = []
    for _, row in df.iterrows():
        feats = {}
        for champ in champion_pool:
            feats[f"champ_{champ}_t1"] = int(champ in row["team1_picks"])
            feats[f"champ_{champ}_t2"] = int(champ in row["team2_picks"])
        feats["team1_champ_wr_avg"] = row["team1_champ_wr_avg"]
        feats["team2_champ_wr_avg"] = row["team2_champ_wr_avg"]
        feats["game_id"] = row["game_id"]
        feats["team1_win"] = row["team1_win"]
        feats["total_kills"] = row["total_kills"]  # NaN se o jogo ainda não tem kills (--add-kills)
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

    df, final_winrates = compute_expanding_champion_winrate(df)

    champion_pool = compute_champion_pool(df)
    print(f"  -> pool de {len(champion_pool)} campeões mais frequentes")

    features = vectorize(df, champion_pool)
    out_path = DATA_DIR / f"features_{args.league}.csv"
    features.to_csv(out_path, index=False)
    print(f"  -> features salvas em {out_path}")

    pool_path = DATA_DIR / f"champion_pool_{args.league}.json"
    pool_path.write_text(json.dumps(champion_pool, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> pool de campeões salvo em {pool_path}")

    winrates_path = DATA_DIR / f"champion_winrates_{args.league}.json"
    winrates_path.write_text(json.dumps(final_winrates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> win rates finais de campeões salvas em {winrates_path} (usadas pelo predict.py)")


if __name__ == "__main__":
    main()
