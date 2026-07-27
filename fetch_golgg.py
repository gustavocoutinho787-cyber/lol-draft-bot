"""
Baixa dados de partidas profissionais de LoL (jogos + picks/bans) do gol.gg,
via scraping de HTML — gol.gg não tem API pública como a Leaguepedia.

Use como alternativa a fetch_leaguepedia.py quando a Cargo API da Leaguepedia
estiver com rate limit. A saída (games_{league}.json, picks_bans_{league}.json)
segue o mesmo formato usado por build_features.py, então o resto do pipeline
não precisa mudar.

Fluxo de scraping:
  1. tournament/tournament-matchlist/<torneio>/  -> lista de séries (bo1/bo3/bo5)
  2. game/stats/<id>/page-summary/               -> ids dos jogos individuais da série
  3. game/stats/<id>/page-game/                  -> times, lado, vencedor, bans e picks

NOTA: os nomes exatos dos torneios no gol.gg (ex: "LPL Spring 2024") precisam
ser passados via --tournaments. Rode com --debug para ver o que foi
encontrado em cada etapa antes de confiar no parsing.
"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://gol.gg"
DATA_DIR = Path(__file__).resolve().parent / "data"
HEADERS = {
    "User-Agent": "lol-draft-bot/0.1 (uso pessoal, contato: preencha-seu-email)"
}
REQUEST_DELAY = 0.5  # ser gentil com o site
MAX_RETRIES = 4
ROLES = ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]


def _normalize_champ_name(name: str) -> str:
    """Remove espaços/apóstrofos/pontos para comparar nomes de fontes diferentes
    do gol.gg (ex: filename 'XinZhao' vs nome exibido 'Xin Zhao')."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def get_soup(path: str) -> BeautifulSoup:
    """Baixa uma página com retry (falhas de rede transitórias, tipo timeout,
    são comuns em scrapes longos de centenas de páginas — sem retry, uma única
    falha derruba o processo inteiro e perde todo o progresso já feito)."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(f"{BASE_URL}{path}", headers=HEADERS, timeout=30)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return BeautifulSoup(resp.text, "lxml")
        except (requests.exceptions.RequestException,) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt  # 2s, 4s, 8s
                time.sleep(wait)
    raise last_exc


def list_series(tournament: str, debug: bool = False) -> list[dict]:
    """Lista as séries (confrontos) de um torneio, com o game id do primeiro jogo de cada uma."""
    slug = quote(tournament)
    soup = get_soup(f"/tournament/tournament-matchlist/{slug}/")
    table = soup.find("table")
    if table is None:
        return []

    series = []
    for row in table.find_all("tr")[1:]:  # pula o cabeçalho
        link = row.find("a", href=True)
        if not link or "/game/stats/" not in link["href"]:
            continue
        first_game_id = link["href"].split("/game/stats/")[1].split("/")[0]
        cells = row.find_all("td")
        patch = cells[5].get_text(strip=True) if len(cells) > 5 else None
        date = cells[6].get_text(strip=True) if len(cells) > 6 else None
        series.append({"first_game_id": first_game_id, "patch": patch, "date": date})

    if debug:
        print(f"[debug] {tournament}: {len(series)} séries, exemplo: {series[:1]}")
    return series


def list_games_in_series(first_game_id: str) -> list[str]:
    """Dado o id do primeiro jogo de uma série, retorna os ids de TODOS os jogos da série."""
    soup = get_soup(f"/game/stats/{first_game_id}/page-summary/")
    game_ids = []
    for a in soup.select("a[href*='page-game']"):
        gid = a["href"].split("/game/stats/")[1].split("/")[0]
        if gid not in game_ids:
            game_ids.append(gid)
    return game_ids or [first_game_id]


def _extract_side_drafts(soup: BeautifulSoup) -> list[list[str]]:
    """Retorna as listas de picks de cada time, na ordem em que aparecem na página
    (primeiro bloco = lado blue, segundo bloco = lado red)."""
    picks_blocks = []
    for label in soup.find_all("div", class_="col-2"):
        if label.get_text(strip=True) != "Picks":
            continue
        sib = label.find_next_sibling("div", class_="col-10")
        if not sib:
            continue
        champs = []
        for a in sib.select("a"):
            img = a.find("img")
            name = (img.get("alt") or img.get("title")) if img else None
            if name:
                champs.append(name)
        picks_blocks.append(champs)
    return picks_blocks


def scrape_game(game_id: str, league: str, patch: str | None, date: str | None,
                 debug: bool = False):
    """Baixa uma página de jogo individual e devolve (game, picks_bans) ou (None, None)
    se o jogo ainda não tiver resultado/draft completo."""
    soup = get_soup(f"/game/stats/{game_id}/page-game/")

    blue_header = soup.select_one(".blue-line-header")
    red_header = soup.select_one(".red-line-header")
    if not blue_header or not red_header:
        return None, None

    blue_name, _, blue_result = blue_header.get_text(" ", strip=True).rpartition(" - ")
    red_name, _, red_result = red_header.get_text(" ", strip=True).rpartition(" - ")

    picks_blocks = _extract_side_drafts(soup)
    if len(picks_blocks) < 2 or len(picks_blocks[0]) < 5 or len(picks_blocks[1]) < 5:
        return None, None  # draft incompleto ou página em formato inesperado

    winner_is_blue = blue_result.strip().upper() == "WIN"

    if debug:
        print(f"[debug] game {game_id}: {blue_name} (blue) vs {red_name} (red), "
              f"vencedor={'blue' if winner_is_blue else 'red'}")
        print(f"[debug]   picks blue={picks_blocks[0]} picks red={picks_blocks[1]}")

    game = {
        "GameId": game_id,
        "League": league,
        "Patch": patch,
        "DateTime_UTC": date,
        "Team1": blue_name.strip(),
        "Team2": red_name.strip(),
        "Winner": "1" if winner_is_blue else "2",
    }
    picks_bans = {
        "GameId": game_id,
        "Team1Picks": ";".join(picks_blocks[0][:5]),
        "Team2Picks": ";".join(picks_blocks[1][:5]),
    }
    return game, picks_bans


def fetch_champion_roles(tournament: str, known_champions: set[str],
                          debug: bool = False) -> dict[str, list[str]]:
    """Baixa a página agregada de picks & bans de um torneio no gol.gg, que
    lista os campeões mais jogados por rota (TOP/JUNGLE/MID/BOT/SUPPORT), e
    devolve {campeão: [rotas]}.

    Um campeão pode aparecer em mais de uma rota (picks flex, ex: Renekton
    top/jungle) — nesse caso fica associado a todas em que apareceu. Os nomes
    de campeão nessa página vêm do filename do ícone (ex: "XinZhao"), sem
    espaço/apóstrofo, então são casados com known_champions por normalização
    em vez de usados direto (para bater com o nome usado no resto dos dados).
    """
    slug = quote(tournament)
    soup = get_soup(f"/tournament/tournament-picksandbans/{slug}/")
    table = soup.find("table", class_="table_list")
    if table is None:
        return {}

    normalized_lookup = {_normalize_champ_name(c): c for c in known_champions}

    roles_by_champ: dict[str, list[str]] = {}
    for row in table.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 2:
            continue
        label = tds[0].get_text(strip=True)
        if label not in ROLES:
            continue
        for img in tds[1].find_all("img"):
            filename = img.get("src", "").rsplit("/", 1)[-1].replace(".png", "")
            champ = normalized_lookup.get(_normalize_champ_name(filename))
            if not champ:
                continue  # campeão não visto nos dados de partidas já baixados
            roles_by_champ.setdefault(champ, [])
            if label not in roles_by_champ[champ]:
                roles_by_champ[champ].append(label)

    if debug:
        print(f"[debug] rotas extraídas para {len(roles_by_champ)} campeões de {tournament}")
    return roles_by_champ


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(league: str, all_games: list, all_picks_bans: list):
    """Deduplica por GameId (relevante com --merge, se algum torneio se sobrepuser),
    ordena por data (vários torneios podem ser passados fora de ordem cronológica,
    ex: playoffs de um split cruzando com o split seguinte — e train.py depende da
    ordem das linhas para validação respeitando o tempo) e salva em disco."""
    seen = set()
    dedup_games, dedup_picks_bans = [], []
    for game, pb in zip(all_games, all_picks_bans):
        gid = game["GameId"]
        if gid in seen:
            continue
        seen.add(gid)
        dedup_games.append(game)
        dedup_picks_bans.append(pb)

    order = sorted(range(len(dedup_games)), key=lambda i: dedup_games[i]["DateTime_UTC"] or "")
    sorted_games = [dedup_games[i] for i in order]
    sorted_picks_bans = [dedup_picks_bans[i] for i in order]

    games_path = DATA_DIR / f"games_{league}.json"
    pb_path = DATA_DIR / f"picks_bans_{league}.json"
    games_path.write_text(json.dumps(sorted_games, ensure_ascii=False, indent=2), encoding="utf-8")
    pb_path.write_text(json.dumps(sorted_picks_bans, ensure_ascii=False, indent=2), encoding="utf-8")
    return games_path, pb_path


def main():
    parser = argparse.ArgumentParser(description="Baixa dados do gol.gg (scraping de HTML)")
    parser.add_argument("--league", required=True, help="Nome curto para salvar os arquivos (ex: LPL)")
    parser.add_argument("--tournaments", nargs="+", default=[],
                         help="Slugs exatos de torneio no gol.gg, ex: \"LPL Spring 2024\" \"LPL Summer 2024\". "
                              "Pode ser omitido se só quiser rodar --roles-from sobre dados já baixados.")
    parser.add_argument("--limit-series", type=int, default=None,
                         help="Limita quantas séries processar por torneio (útil para testar)")
    parser.add_argument("--merge", action="store_true",
                         help="Soma aos dados já salvos em vez de sobrescrever (útil para "
                              "completar torneios que faltaram numa rodada anterior)")
    parser.add_argument("--roles-from", default=None,
                         help="Slug de torneio (ex: \"LPL Spring 2024\") para extrair a rota "
                              "(TOP/JUNGLE/MID/BOT/SUPPORT) de cada campeão e salvar em "
                              "champion_roles_{league}.json — usado pelo app.py para organizar "
                              "os menus de pick por rota")
    parser.add_argument("--debug", action="store_true", help="Imprime detalhes de cada etapa")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_games = []
    all_picks_bans = []

    if args.merge:
        games_path = DATA_DIR / f"games_{args.league}.json"
        pb_path = DATA_DIR / f"picks_bans_{args.league}.json"
        if games_path.exists() and pb_path.exists():
            all_games = load_json(games_path)
            all_picks_bans = load_json(pb_path)
            print(f"--merge: {len(all_games)} jogos já salvos carregados de {games_path}")

    if not args.tournaments and not args.roles_from:
        parser.error("passe --tournaments, --roles-from, ou ambos")

    for tournament in args.tournaments:
        print(f"Torneio: {tournament}")
        series_list = list_series(tournament, debug=args.debug)
        if args.limit_series:
            series_list = series_list[: args.limit_series]
        print(f"  -> {len(series_list)} séries encontradas")

        for i, series in enumerate(series_list, 1):
            game_ids = list_games_in_series(series["first_game_id"])
            for game_id in game_ids:
                game, picks_bans = scrape_game(
                    game_id, args.league, series["patch"], series["date"], debug=args.debug
                )
                if game:
                    all_games.append(game)
                    all_picks_bans.append(picks_bans)
            if i % 10 == 0 or i == len(series_list):
                print(f"  ... {i}/{len(series_list)} séries processadas ({len(all_games)} jogos até agora)")

        # Salva um checkpoint depois de cada torneio: um scrape completo pode
        # levar dezenas de minutos e centenas de requisições — se uma falha de
        # rede (mesmo com retry) matar o processo, o progresso já feito não se perde.
        _save(args.league, all_games, all_picks_bans)

    if args.tournaments:
        games_path, pb_path = _save(args.league, all_games, all_picks_bans)
        print(f"  -> {len(all_games)} partidas salvas em {games_path}")
        print(f"  -> {len(all_picks_bans)} registros de draft salvos em {pb_path}")

    if args.roles_from:
        known_champions = set()
        winrates_path = DATA_DIR / f"champion_winrates_{args.league}.json"
        if winrates_path.exists():
            known_champions.update(load_json(winrates_path).keys())
        for pb in all_picks_bans:
            known_champions.update((pb.get("Team1Picks") or "").split(";"))
            known_champions.update((pb.get("Team2Picks") or "").split(";"))
        known_champions.discard("")

        print(f"Buscando rotas dos campeões em: {args.roles_from}")
        roles_by_champ = fetch_champion_roles(args.roles_from, known_champions, debug=args.debug)
        roles_path = DATA_DIR / f"champion_roles_{args.league}.json"
        roles_path.write_text(json.dumps(roles_by_champ, ensure_ascii=False, indent=2), encoding="utf-8")
        sem_rota = known_champions - set(roles_by_champ.keys())
        print(f"  -> rotas de {len(roles_by_champ)} campeões salvas em {roles_path}")
        if sem_rota:
            print(f"  -> {len(sem_rota)} campeões sem rota identificada (ficam disponíveis em todas): "
                  f"{sorted(sem_rota)[:10]}{'...' if len(sem_rota) > 10 else ''}")


if __name__ == "__main__":
    main()
