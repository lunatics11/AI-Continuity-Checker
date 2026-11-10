"""Continuity Checker: parse -> extract -> compare -> build.
    python pipeline.py samples    run on a folder of screenplays, save outputs/
    python pipeline.py test       plant 4 known errors and count how many get caught
CODE parses, verifies quotes and builds.  AI reads and judges.  A HUMAN decides."""
import os, re, sys, json, time, base64, hashlib
from pathlib import Path
from typing import Literal, TypedDict
from concurrent.futures import ThreadPoolExecutor
import cv2
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
MODEL = os.getenv("MODEL", "gemini-3.1-flash-lite")          
FALLBACK = os.getenv("FALLBACK_MODEL", "gemini-3.8-flash")
CACHE = HERE / ".cache"
CACHE.mkdir(exist_ok=True)
Kind = Literal["character", "prop", "costume", "location"]


# model function
def ask(prompt, schema, images=()):
    """Send a prompt (+ optional JPEG frames), get back a validated object. Answers are cached, so reruns are free."""
    f = CACHE / (hashlib.sha256((MODEL + prompt).encode() + b"".join(images)).hexdigest() + ".json")
    if f.exists():
        return schema.model_validate_json(f.read_text())
    parts = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(i).decode()}} for i in images]
    for model in (MODEL, FALLBACK):                          
        llm = init_chat_model(model, model_provider=os.getenv("PROVIDER", "google_genai"), temperature=0).with_structured_output(schema)
        for wait in (0, 5, 10):
            time.sleep(wait)
            try:
                out = llm.invoke([{"role": "user", "content": parts}])
                f.write_text(out.model_dump_json())
                return out
            except Exception as e:
                err = e
                if "NOT_FOUND" in str(e):                    # retired model name: go to the next model
                    break
    raise RuntimeError(f"Models busy or out of free quota. Try again in a minute. Last error: {str(err)[:200]}")


# Parsing 
def parse_ep(name, text):
    """Screenplay -> (title, rows). Each row = (scene, speaker or CAMERA/ACTION/SOUND FX, time, text)."""
    title = (re.search(r"EPISODE NUMBER:\s*(.+)", text) or [None, name])[1].strip()
    rows = []
    for chunk in re.split(r"(?m)^(?=SCENE \d+)", text)[1:]:
        scene = chunk.split("\n")[0].strip()
        for label, m, s, body in re.findall(r"\*\*(.+?):\*\*\s*\nStart: \((\d+):(\d+\.?\d*)\)[^\n]*\n(.+(?:\n(?!\n|\*\*).+)*)", chunk):
            rows.append((scene, re.sub(r"\s*\(.*", "", label), f"{int(m):02d}:{float(s):04.1f}", " ".join(body.split())))
    return title, rows


def speakers(rows):
    """speaker -> (lines spoken, what the camera showed just before they first spoke)"""
    out, last = {}, ""
    for _, who, _, text in rows:
        if who in ("CAMERA", "ACTION"):
            last = text
        elif who != "SOUND FX":
            n, ctx = out.get(who, (0, last[:140]))
            out[who] = (n + 1, ctx)
    return out


#Extract: AI lists visible facts, CODE keeps only facts whose quote is really in the script
class Fact(BaseModel):
    entity: str = Field(description="the person/object/place as the script names it, e.g. \"Juliana's pendant\"")
    kind: Kind
    attribute: str = Field(description="one visible attribute: eye colour, hair, shape, material, colour, scar, uniform...")
    value: str
    quote: str = Field(description="the exact words copied from the script that state this")
    at: str = Field(description="the [SCENE mm:ss] tag of that line")


class Facts(BaseModel):
    facts: list[Fact]


EXTRACT = """You are a script supervisor for AI-generated vertical microdramas. An AI video tool will draw every shot
from the script, so any VISIBLE detail (what a person, costume, prop or place looks like) must stay identical.

Read the episode. List every visible fact about characters, costumes, props and locations.
Rules: facts about how things LOOK only (no feelings, no plot). Copy the quote EXACTLY from the script. Use the line's [SCENE mm:ss] tag.
Do not invent anything. Be thorough: every colour, material, shape, hair, eye colour, accessory, uniform, age, scar, building detail.
A normal episode has 20-40 such facts. Include facts about every named or described person, not only the lead.

EPISODE: {title}
{script}"""


# Compare AI judges which' facts clash and writes one locked description per thing
class Entity(BaseModel):
    canonical: str = Field(description="one clean name, e.g. \"Juliana\" or \"Mother's pendant\"")
    kind: Kind
    aliases: list[str] = Field(description="other names/labels the script uses for the same thing")
    locked: str = Field(description="ONE description to paste in every video prompt: only visible details, stated in the script, no conflicts")


class Conflict(BaseModel):
    entity: str
    attribute: str
    kind: Literal["contradiction", "wording_drift"] = Field(description="contradiction = both cannot be true; wording_drift = same thing described so differently a video model may draw two different things")
    fact_ids: list[int] = Field(description="at least two fact ids that clash")
    severity: Literal["high", "medium", "low"]
    why: str = Field(description="one plain sentence")
    fix: str = Field(description="one sentence telling the writer what to lock it to")


class NameDrift(BaseModel):
    canonical: str
    labels: list[str] = Field(description="the different speaker labels for the same character")
    why: str


class Compared(BaseModel):
    entities: list[Entity]
    conflicts: list[Conflict]
    name_drift: list[NameDrift]


COMPARE = """You are the continuity supervisor of an AI microdrama series. Below are visual facts extracted from several episodes
(id | episode | time | entity | attribute | value), and the speaker labels used in each episode.

1. entities: merge facts about the same character/prop/costume/location (aliases!) and write one locked description from the script facts.
2. conflicts: flag facts about the same thing that CLASH. "contradiction" if both cannot be true (blue vs green eyes).
   "wording_drift" if one thing gets different names or looks across episodes (pendant vs locket vs medallion). Never flag two
   different things, never flag details that merely add up (an oval shape AND a purple stone can both be true), never invent ids.
3. name_drift: one character that appears under several speaker labels (use the "first shown as" context to tell who a label is,
   e.g. a label like "BLOND NOBLEMAN" shown right after a shot of "Juliana's father" is the father).
Be precise rather than noisy: a short list a human can check in one minute.

FACTS:
{facts}

SPEAKER LABELS:
{labels}"""


#Pipeline
class State(TypedDict, total=False):
    files: dict      # filename -> screenplay text
    eps: list        # [(title, rows)]
    facts: list
    out: dict
    result: dict


def parse(s):
    return {"eps": [parse_ep(n, t) for n, t in sorted(s["files"].items())]}


def extract(s):
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()

    def one(ep):
        title, rows = ep
        script = "\n".join(f"[{sc} {t}] {who}: {tx}" for sc, who, t, tx in rows)
        got = ask(EXTRACT.format(title=title, script=script), Facts).facts
        return [dict(f.model_dump(), ep=title) for f in got if norm(f.quote) and norm(f.quote) in norm(script)]   # anti-hallucination

    with ThreadPoolExecutor(4) as pool:
        per_ep = list(pool.map(one, s["eps"]))
    return {"facts": [dict(f, id=i) for i, f in enumerate(f for p in per_ep for f in p)]}


def compare(s):
    facts = "\n".join(f"{f['id']} | {f['ep']} | {f['at']} | {f['entity']} | {f['attribute']} | {f['value']}" for f in s["facts"])
    labels = "\n".join(f"{t} | {w} ({n} lines) | first shown as: {c}" for t, rows in s["eps"] for w, (n, c) in speakers(rows).items())
    return {"out": ask(COMPARE.format(facts=facts, labels=labels), Compared).model_dump()}


def build(s):
    """CODE checks the AI's answer: a conflict must cite at least 2 real, different quotes."""
    by_id = {f["id"]: f for f in s["facts"]}
    conflicts = []
    for c in s["out"]["conflicts"]:
        evidence = [by_id[i] for i in c["fact_ids"] if i in by_id]
        if len({(f["ep"], f["quote"]) for f in evidence}) >= 2:
            conflicts.append({**c, "evidence": evidence})
    conflicts.sort(key=lambda c: ["high", "medium", "low"].index(c["severity"]))
    shots = [dict(ep=t, scene=sc, at=tm, text=tx) for t, rows in s["eps"] for sc, who, tm, tx in rows if who == "CAMERA"]
    return {"result": {**s["out"], "conflicts": conflicts, "shots": shots, "n_facts": len(s["facts"]), "episodes": [t for t, _ in s["eps"]]}}


g = StateGraph(State)
g.add_sequence([parse, extract, compare, build])             # parse -> extract -> compare -> build
g.add_edge(START, "parse"); g.add_edge("build", END)
GRAPH = g.compile()


def run(files, on_step=None):
    """files = {filename: screenplay text}  ->  result dict (entities, conflicts, name_drift, shots)"""
    state = {}
    for update in GRAPH.stream({"files": files}, stream_mode="updates"):
        for node, delta in update.items():
            state.update(delta)
            if on_step:
                on_step(node)
    return state["result"]


#outputs
def prompt_block(res):
    """Paste this in front of every shot prompt."""
    lines = ["LOCKED DESCRIPTIONS - keep these exactly the same in every shot:"]
    for e in res["entities"]:
        also = f" (also called: {', '.join(e['aliases'])})" if e["aliases"] else ""
        lines.append(f"- {e['canonical']} [{e['kind']}]{also}: {e['locked']}")
    return "\n".join(lines)


def shot_prompt(res, shot):
    """One ready-to-paste video prompt: the shot + only the reference-sheet entries whose names appear in it."""
    low, used = shot["text"].lower(), []
    for e in res["entities"]:
        names = [e["canonical"], *e["aliases"]]
        if e["kind"] == "character":                         # also match a first name, but not a possessive ("Juliana's father")
            names += [n.split()[0] for n in names if len(n.split()[0]) > 3 and n[0].isupper() and "'" not in n.split()[0]]
        if any(n.lower() in low for n in names):
            used.append(f"- {e['canonical']}: {e['locked']}")
    return f"SHOT: {shot['text']}\n\nKEEP EXACTLY THE SAME:\n" + ("\n".join(used) or "- (no tracked character or prop in this shot)")


# vision model
class Mismatch(BaseModel):
    frames: list[int] = Field(description="frame numbers (1-based) where it is visible")
    entity: str
    expected: str = Field(description="what the reference sheet says")
    seen: str = Field(description="what the frames actually show")
    severity: Literal["high", "medium", "low"]
    fix_prompt: str = Field(description="one sentence to regenerate the shot")


class ClipReport(BaseModel):
    mismatches: list[Mismatch]
    summary: str


CLIP = """These {n} frames (numbered 1..{n}) come from ONE AI-generated shot of a microdrama. Compare them with the reference sheet below.
List only what you can CLEARLY see: (a) a visible detail that contradicts the reference sheet, or (b) something that looks different between frames
(face, outfit, prop, hands). Do not guess, do not mention things the reference sheet says nothing about. If all is fine, return an empty list.
The reference sheet describes the characters in general. If the SHOT CONTEXT explains a change (a later scene, different clothes, injury, dirt), that is
NOT an error. Only flag a reference-sheet detail that should still hold, such as face, hair, eye colour, age and props that appear in this shot.

SHOT CONTEXT (what this shot is meant to show): {context}

{sheet}"""


def check_clip(video_bytes, sheet, context=""):
    """Sample 6 frames, ask the vision model. Returns (frames, report)."""
    (CACHE / "clip.mp4").write_bytes(video_bytes)
    video = cv2.VideoCapture(str(CACHE / "clip.mp4"))
    total, frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)), []
    for i in range(6):
        video.set(cv2.CAP_PROP_POS_FRAMES, int((i + .5) * total / 6))
        ok, img = video.read()
        if ok:
            frames.append(cv2.imencode(".jpg", cv2.resize(img, (512, int(512 * img.shape[0] / img.shape[1]))))[1].tobytes())
    sheet = sheet or "No reference sheet was given. Only report details that look different between the frames (face, outfit, prop, hands)."
    return frames, ask(CLIP.format(n=len(frames), sheet=sheet, context=context or "not given"), ClipReport, images=frames)


PLANTS = [("003", "wide blue eyes", "wide green eyes"), ("002", "gold-topped cane", "silver-topped cane"),
          ("002", "shimmering gold gown", "emerald green gown"), ("003", "her white gown", "her crimson gown")]


def self_test(files):
    """Plant 4 known errors, run the tool, count the catches (a catch = a conflict that quotes the planted words)."""
    bad = dict(files)
    for num, old, new in PLANTS:
        name = next(k for k in bad if f"episode-{num}" in k)
        assert old in bad[name], f"cannot plant: {old!r} not found"
        bad[name] = bad[name].replace(old, new, 1)
    conflicts = run(bad)["conflicts"]
    hits = [any(new in f["quote"].lower() for c in conflicts for f in c["evidence"]) for _, _, new in PLANTS]
    for (_, old, new), hit in zip(PLANTS, hits):
        print("caught " if hit else "MISSED ", f"{old} -> {new}")
    print(f"\nPLANTED ERRORS CAUGHT: {sum(hits)}/{len(PLANTS)}")
    print("conflicts on the clean scripts (check these by hand):", [c["entity"] + " / " + c["attribute"] for c in run(files)["conflicts"]])


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "samples"
    folder = HERE / "samples" if arg == "test" else Path(arg)
    files = {p.name: p.read_text() for p in sorted(folder.glob("*.txt"))}
    if arg == "test":
        self_test(files)
    else:
        res = run(files, on_step=lambda n: print("done:", n))
        (HERE / "outputs").mkdir(exist_ok=True)
        (HERE / "outputs" / "report.json").write_text(json.dumps(res, indent=2))
        (HERE / "outputs" / "prompt_block.txt").write_text(prompt_block(res))
        print(f"{len(res['episodes'])} episodes, {res['n_facts']} facts, {len(res['conflicts'])} conflicts, {len(res['name_drift'])} name drifts")
        print(prompt_block(res))
