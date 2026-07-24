# LoL Draft Bot — Previsão de vitória a partir do draft

Bot que analisa o draft (pick/ban) de partidas profissionais de League of Legends
e estima a probabilidade de vitória de cada time. A ideia é, no futuro, comparar
essa probabilidade com as odds das casas de apostas para achar apostas de valor
("edge positivo").

## Escopo atual

- Foco inicial: **LPL** (fácil de estender para LCK, LEC, CBLOL etc.)
- Fonte de dados: **Leaguepedia** (API pública Cargo/MediaWiki, gratuita)
- Modelo: previsão de win probability a partir do draft completo (picks, bans, side)
- Odds de apostas: **fora de escopo por enquanto** — módulo fica pronto para
  receber essa integração depois

## Estrutura

```
lol-draft-bot/
├── src/
│   ├── data/
│   │   └── fetch_leaguepedia.py   # baixa partidas + drafts da Leaguepedia
│   ├── features/
│   │   └── build_features.py      # transforma drafts em features numéricas
│   └── model/
│       ├── train.py               # treina modelo de win probability
│       └── predict.py             # roda inferência em um draft novo
├── data/                          # dados brutos e processados (csv)
├── requirements.txt
└── README.md
```

## Como rodar

```bash
pip install -r requirements.txt

# 1. Baixar dados históricos de partidas da LPL
python src/data/fetch_leaguepedia.py --league LPL --year 2024 2025

# 2. Construir features a partir do draft
python src/features/build_features.py

# 3. Treinar o modelo
python src/model/train.py

# 4. Prever um draft específico
python src/model/predict.py --config exemplo_draft.json
```

## Notas importantes

- A **Riot Esports API** oficial (dados "de verdade" de LPL/LCK/LEC ao vivo) é
  restrita a parceiros — não há acesso público self-service. A Leaguepedia é a
  alternativa aberta mais completa para dados históricos de draft.
- Para odds ao vivo, o caminho mais comum é uma API de odds paga (ex: The Odds
  API, Pinnacle API) ou scraping — nenhuma decisão foi tomada sobre isso ainda.
- Este é um projeto de **análise/apoio à decisão**. As previsões do modelo são
  estimativas estatísticas, não garantias — aposte com responsabilidade.
