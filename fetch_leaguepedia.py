"""
Baixa dados de partidas profissionais de LoL (jogos + picks/bans) da Leaguepedia,
usando a Cargo API pública do MediaWiki.

A Leaguepedia expõe os dados via tabelas Cargo. As duas principais que usamos:

  - ScoreboardGames: uma linha por PARTIDA (times, vencedor, patch, liga, data)
  - PicksAndBansS7 : uma linha por PICK/BAN de uma partida (side, número da
                      ordem, campeão, se é pick ou ban)

Docs da API: https://lol.fandom.com/wiki/Help:Leaguepedia_API
Explorador de tabelas: https://lol.fandom.com/wiki/Special:CargoTables

NOTA: nomes exatos de campos podem mudar entre temporadas/versões da Cargo
schema da Leaguepedia. Rode com --debug na primeira vez para inspecionar o
JSON bruto antes de confiar no parsing.
"""

import argparse
import json
import time
from pathlib import Path

import requests

API_URL = "https://lol.fandom.com/api.php"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

HEADERS = {
    "User-Agent": "lol-draft-bot/0.1 (uso pessoal, contato: preencha-seu-email)"
}


def cargo_query(tables, fields, where=None, join_on=None, order_by=None,
                 limit=500, offset=0):
    params = {
        "action": "cargoquery",
        "format": "json",
        "tables": tables,
        "fields": fields,
        "limit": limit,
        "offset": offset,
    }
    if where:
        params["where"] = where
    if join_on:
        params["join_on"] = join_on
    if order_by:
        params["order_by"] = order_by

    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"Cargo API error: {payload['error']}")
    return [row["title"] for row in payload.get("cargoquery", [])]


def fetch_games(league: str, years: list[str], debug: bool = False):
    """Baixa metadados de partidas (times, vencedor, patch, data) para a liga/anos dados."""
    all_rows = []
    for year in years:
        offset = 0
        while True:
            where = f'League="{league}" AND YEAR(DateTime_UTC)={year}'
            rows = cargo_query(
                tables="ScoreboardGames",
                fields=(
                    "GameId,DateTime_UTC,League,Patch,Team1,Team2,Winner,"
                    "Team1Score,Team2Score,Gamelength"
                ),
                where=where,
                order_by="DateTime_UTC",
                limit=500,
                offset=offset,
            )
            if debug and offset == 0:
                print(f"[debug] exemplo de linha (games, {year}):", json.dumps(rows[:1], indent=2, ensure_ascii=False))
            if not rows:
                break
            all_rows.extend(rows)
            offset += 500
            time.sleep(0.5)  # ser gentil com a API pública
    return all_rows


def fetch_picks_and_bans(league: str, years: list[str], debug: bool = False):
    """Baixa picks/bans (campeão, ordem, lado, pick ou ban) por partida."""
    all_rows = []
    for year in years:
        offset = 0
        while True:
            where = f'League="{league}" AND YEAR(DateTime_UTC)={year}'
            rows = cargo_query(
                tables="PicksAndBansS7",
                fields=(
                    "GameId,Team1,Team2,Team1Picks,Team2Picks,"
                    "Team1Bans,Team2Bans,Team1Players,Team2Players,DateTime_UTC"
                ),
                where=where,
                order_by="DateTime_UTC",
                limit=500,
                offset=offset,
            )
            if debug and offset == 0:
                print(f"[debug] exemplo de linha (picks/bans, {year}):", json.dumps(rows[:1], indent=2, ensure_ascii=False))
            if not rows:
                break
            all_rows.extend(rows)
            offset += 500
            time.sleep(0.5)
    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Baixa dados da Leaguepedia")
    parser.add_argument("--league", default="LPL", help="Código da liga (ex: LPL, LCK, LEC, CBLOL)")
    parser.add_argument("--year", nargs="+", default=["2025"], help="Ano(s), ex: --year 2024 2025")
    parser.add_argument("--debug", action="store_true", help="Imprime exemplo de payload bruto")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Baixando partidas de {args.league} para {args.year}...")
    games = fetch_games(args.league, args.year, debug=args.debug)
    games_path = DATA_DIR / f"games_{args.league}.json"
    games_path.write_text(json.dumps(games, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {len(games)} partidas salvas em {games_path}")

    print(f"Baixando picks/bans de {args.league} para {args.year}...")
    pb = fetch_picks_and_bans(args.league, args.year, debug=args.debug)
    pb_path = DATA_DIR / f"picks_bans_{args.league}.json"
    pb_path.write_text(json.dumps(pb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {len(pb)} registros de draft salvos em {pb_path}")


if __name__ == "__main__":
    main()
