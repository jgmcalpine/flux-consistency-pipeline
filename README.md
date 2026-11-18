# flux-consistency-pipeline

Keep characters, costumes and locations on-model across AI-generated shots made with
[FLUX](https://bfl.ai) by Black Forest Labs.

The idea: treat a film like a software build. The **story bible** is the source code,
generated stills and clips are build artifacts, and **automated QA** is the test suite
that rejects drift before it reaches video.

```
bible (YAML) -> canonical refs -> keyframes -> still QA -> animate -> video QA -> rough cut
                      \______________ every paid call goes through the ledger ____/
```

> Status: **M0, repo and guardrails.** No API calls yet. See the roadmap for M1-M7.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add your BFL key later (M2); DRY_RUN stays true until then

fluxpipe validate             # check the story bible
fluxpipe prompts              # see the prompt every shot will use
fluxpipe budget               # estimated spend vs. the cap
pytest
```

## Layout

| Path                         | What it is                                                |
| ---------------------------- | --------------------------------------------------------- |
| `bible/`                     | Stage 1: characters, locations and the shot list, as YAML |
| `src/fluxpipe/bible.py`      | Loads and validates the bible                             |
| `src/fluxpipe/prompts.py`    | Builds every prompt from one template                     |
| `src/fluxpipe/ledger.py`     | SQLite log of every paid call; enforces `SPEND_CAP_USD`   |
| `src/fluxpipe/config.py`     | Settings and the estimated price table                    |
| `src/fluxpipe/bfl_client.py` | The only code that talks to the BFL API                   |
| `src/fluxpipe/qa/`           | Face, palette, costume and video checks (M5-M6)           |
| `assets/`                    | Generated output and the ledger database (git-ignored)    |

## Guardrails

- **Spend cap.** The ledger writes a row _before_ each paid call and refuses the call if
  estimated spend would pass `SPEND_CAP_USD`. Failed calls still count, since they may be billed.
- **Dry run.** With `DRY_RUN=true`, calls are logged but never sent.
- **Secrets.** The API key lives in `.env`, which git ignores.

Prices in `config.py` are estimates. Check them against the BFL dashboard after the first real calls.

## Licensing notes

InsightFace's pretrained models (used for face QA in M5) are for non-commercial use.
