# Continuity Checker (Kuku AI Generalist, Problem 3)

An AI-powered system for intelligent visual consistency, scene validation and cross-episode continuity in AI-generated microdramas.
AI video tools generate every shot from its own prompt. If a script calls a necklace a *pendant* in Ep 1 and a *locket* in Ep 3, the viewer sees two objects. This tool reads the screenplays, finds details that clash across episodes, and writes one locked description per character/prop.
Check out the deployed version:
https://ai-continuity-checker.streamlit.app

## Run it
```bash
pip install -r requirements.txt
cp .env              # put GEMINI_API_KEY in .env
streamlit run app.py              # the UI
python pipeline.py samples        # no UI: saves outputs/report.json + prompt_block.txt
python pipeline.py test           # plants 4 known errors, counts catches
```
