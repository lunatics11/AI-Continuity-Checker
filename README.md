# Continuity Checker (Kuku AI Generalist, Problem 3)

AI video tools draw every shot from its own prompt. If a script calls a necklace a *pendant* in Ep 1 and a *locket* in Ep 3, the viewer sees two objects. This tool reads the screenplays, finds details that clash across episodes, and writes one locked description per character/prop, plus a ready-to-paste prompt for every camera shot.

## Run it
```bash
pip install -r requirements.txt
cp .env              # put GEMINI_API_KEY in .env
streamlit run app.py              # the UI
python pipeline.py samples        # no UI: saves outputs/report.json + prompt_block.txt
python pipeline.py test           # plants 4 known errors, counts catches
```
