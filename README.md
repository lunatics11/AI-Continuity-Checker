# Continuity Checker (Kuku AI Generalist, Problem 3)

AI video tools draw every shot from its own prompt. If a script calls a necklace a *pendant* in Ep 1 and a *locket* in Ep 3, the viewer sees two objects. This tool reads the screenplays, finds details that clash across episodes, and writes one locked description per character/prop, plus a ready-to-paste prompt for every camera shot.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env              # put GEMINI_API_KEY in .env
streamlit run app.py              # the UI
python pipeline.py samples        # no UI: saves outputs/report.json + prompt_block.txt
python pipeline.py test           # plants 4 known errors, counts catches
```
Swap the model in `.env` (`MODEL=`, `PROVIDER=`). Answers are cached in `.cache/`, so repeat runs are free.

## Files
| file | what |
|---|---|
| `pipeline.py` | the whole engine: `ask()` (the only model call), parse, extract, compare, build, prompts, clip check, self-test |
| `app.py` | Streamlit UI (black / red / white) |
| `docs/` | solution doc (PDF + source) and interview study guide |
| `outputs/` | raw outputs from real runs |
| `samples/` | Kuku's 3 sample episodes (confidential, git-ignored) |

## How it works (LangGraph: parse -> extract -> compare -> build)
| step | who | what |
|---|---|---|
| parse | code | one regex turns a screenplay into rows: scene, speaker, time, text |
| extract | AI, 1 call per episode | list visible facts, each with an exact quote |
| verify | code | drops any fact whose quote is not really in the script |
| compare | AI, 1 call | merge aliases, flag `contradiction` / `wording_drift` / name drift, write locked descriptions |
| build | code | keeps only conflicts that cite 2+ real quotes; builds the prompts |
| clip check | vision AI, optional | 6 frames vs the reference sheet + one line saying what the shot should show |

A human makes the final call.

## Results (real runs, `gemini-3.1-flash-lite`)
- Episodes 1-3: 37 facts, 1 conflict (pendant / amethyst / locket), 3 name drifts (Clarissa, matriarch, father).
- Self-test: **3 of 4** planted errors caught; missed Clarissa's gown colour change. One run, 4 plants, one small model: a rough signal, not accuracy.

## Honest limits
- Output varies a little between runs; treat it as a to-check list.
- A small model extracts few facts (37 from 3 episodes), so details it never extracts cannot be compared.
- Scripts are checked as text only: it cannot see how an object looks.
- The clip check needs the "what is this shot meant to show" line, or later-scene clothes get flagged wrongly.
- Compare runs in one call: for 50+ episodes it must be chunked / incremental. Not built.
- Free Gemini tiers are often busy: `ask()` retries, then uses a backup model.
- `samples/` is Kuku's confidential pack. Do not publish it.
