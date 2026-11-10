from concurrent.futures import ThreadPoolExecutor
import base64
import streamlit as st
import streamlit.components.v1 as components
import pipeline as P

st.set_page_config(page_title="Continuity Checker", layout="wide")
show = lambda h: st.markdown(h, unsafe_allow_html=True)

show("""<style>
.stApp { background:#000; } header[data-testid="stHeader"], #MainMenu, footer { visibility:hidden; height:0; }
.block-container { padding-top:2rem; max-width:1100px; }
h1 span { color:#E50914; }
.card { border:1px solid #333; border-left:4px solid #E50914; border-radius:6px; padding:12px 16px; margin:10px 0; }
.card.medium { border-left-color:#fff; } .card.low { border-left-color:#666; }
.card h4 { margin:0 0 4px; font-size:1.05rem; }
.quote { border:1px solid #333; border-radius:4px; padding:6px 10px; margin:6px 0; font-size:.92rem; }
.quote small { color:#E50914; display:block; }
.tag { color:#E50914; font-weight:700; font-size:.75rem; margin-right:8px; }
.stButton>button, .stDownloadButton>button { background:#E50914; color:#fff; border:0; border-radius:6px; font-weight:700; }
.stButton>button:hover, .stDownloadButton>button:hover { background:#B20710; color:#fff; }
</style>""")
show('<div style="color:#E50914;font-weight:800;letter-spacing:.25em;font-size:.8rem">KUKU · AI MICRODRAMA TOOLKIT</div>')
st.markdown("# Continuity <span>Checker</span>", unsafe_allow_html=True)
st.write("Catches script details that clash across episodes, so every AI-generated frames carry the contiguity. This tool finds those clashes before you generate and writes one fixed description for each character and prop.")


def card(sev, head, why="", quotes=(), fix=""):
    q = "".join(f'<div class="quote"><small>{a}</small>{b}</div>' for a, b in quotes)
    show(f'<div class="card {sev}"><h4>{head}</h4>{why}{q}' + (f"<p><b>Fix:</b> {fix}</p>" if fix else "") + "</div>")


# input
samples = sorted((P.HERE / "samples").glob("*.txt"))
sample_clip = P.HERE / "samples" / "Sample_clip.mp4"
left, right = st.columns(2, gap="large")
uploads = left.file_uploader("Screenplay files (.txt), prefered by sequence", type="txt", accept_multiple_files=True)
use_sample = left.toggle("Use the 3 sample episodes", value=not uploads, disabled=not samples)
clip_up = right.file_uploader("Optional: an AI-generated clip (.mp4)", type=["mp4", "mov"])
use_clip = right.toggle("Use the sample clip", value=False, disabled=not sample_clip.exists())
files = {u.name: u.getvalue().decode("utf-8", "ignore") for u in uploads} or ({p.name: p.read_text() for p in samples} if use_sample else {})
clip_bytes = clip_up.getvalue() if clip_up else (sample_clip.read_bytes() if use_clip else None)
st.caption(f"Model: {P.MODEL}")

steps = {"parse": "Parsing screenplays", "extract": "Finding visual facts", "compare": "Comparing episodes", "build": "Building the reference sheet"}


@st.cache_resource
def worker():
    return ThreadPoolExecutor(1)


if st.button("Run check"):
    if not files:
        st.error("Add at least one screenplay, or switch on the samples.")
    else:
        job = {"done": [], "res": None, "err": None}

        def work():
            try:
                job["res"] = P.run(files, on_step=job["done"].append)
            except Exception as e:
                job["err"] = e

        worker().submit(work)
        st.session_state["job"] = job


def progress():
    job = st.session_state.get("job")
    if not job:
        return
    if job["res"] is None and job["err"] is None:
        st.info("Working: " + (", ".join(steps[n] for n in job["done"]) or "starting") + " ...")
        return
    if job["err"]:
        st.error(str(job["err"])[:400])
    else:
        st.session_state["res"] = job["res"]
    st.session_state["job"] = None
    st.rerun()


job_running = bool(st.session_state.get("job"))
st.fragment(progress, run_every=2 if job_running else None)()

def clip_section(sheet):
    """Works with or without a script: with no reference sheet it only compares the frames with each other."""
    st.caption("A vision model checks 6 frames of the clip chosen at the top. "
               + ("It compares them with the reference sheet." if sheet else "No script was run, so it only checks that the frames match each other."))
    if clip_bytes:
        # the video is embedded in the page itself: st.video can fail to load on Streamlit Cloud
        b64 = base64.b64encode(clip_bytes).decode()
        components.html(f'<video src="data:video/mp4;base64,{b64}" controls style="height:260px;border-radius:6px"></video>', height=270)
    else:
        st.info("Add a clip at the top of the page, or switch on the sample clip.")
    context = st.text_input("What is this shot meant to show?", placeholder="e.g. After the wedding. Juliana, muddy, with a bandaged hand, replants a daisy.")
    if clip_bytes and st.button("Check clip"):
        with st.spinner("Looking at the frames..."):
            try:
                st.session_state["clip"] = P.check_clip(clip_bytes, sheet, context)
            except Exception as e:
                st.session_state["clip"] = None
                st.error(str(e)[:300])
    if st.session_state.get("clip"):
        frames, rep = st.session_state["clip"]
        # frames are embedded in the page (st.image can show broken icons on Streamlit Cloud)
        show("".join(f'<img src="data:image/jpeg;base64,{base64.b64encode(f).decode()}" width="110" style="margin:0 6px 6px 0;border-radius:4px">' for f in frames))
        st.markdown(f"**{rep.summary}**")
        for m in rep.mismatches:
            card(m.severity, f'<span class="tag">{m.severity.upper()} · FRAMES {", ".join(map(str, m.frames))}</span>{m.entity}',
                 quotes=[("Expected", m.expected), ("Seen", m.seen)], fix=m.fix_prompt)


res = st.session_state.get("res")
if clip_bytes:  
    st.markdown("### Check a clip")
    clip_section(P.prompt_block(res) if res else "")
if not res:
    st.stop()

# results─
for col, (n, label) in zip(st.columns(4), [(len(res["episodes"]), "Episodes"), (res["n_facts"], "Facts found"),
                                           (len(res["conflicts"]), "Conflicts"), (len(res["name_drift"]), "Name drifts")]):
    col.metric(label, n)

#radio 
view = st.radio("Section", ["Conflicts", "Name drift", "Reference sheet", "Prompts"], horizontal=True, label_visibility="collapsed", key="view")

if view == "Conflicts":
    for c in res["conflicts"]:
        card(c["severity"], f'<span class="tag">{c["severity"].upper()} · {c["kind"].replace("_", " ").upper()}</span>{c["entity"]} · {c["attribute"]}',
             c["why"], [(f'{f["ep"]} · {f["at"]}', f'“{f["quote"]}”') for f in c["evidence"]], c["fix"])
    if not res["conflicts"]:
        st.info("No conflicts found.")

elif view == "Name drift":
    st.caption("The same character written under different speaker labels. A video model may cast them as different people.")
    for d in res["name_drift"]:
        card("medium", d["canonical"], d["why"], [("Labels used", " · ".join(d["labels"]))])

elif view == "Reference sheet":
    for i in range(0, len(res["entities"]), 3):
        for col, e in zip(st.columns(3), res["entities"][i:i + 3]):
            also = f"<br><small>also: {', '.join(e['aliases'])}</small>" if e["aliases"] else ""
            col.markdown(f'<div class="card medium"><small>{e["kind"].upper()}</small><h4>{e["canonical"]}</h4>{e["locked"]}{also}</div>', unsafe_allow_html=True)

elif view == "Prompts":
    st.caption("Paste this block in front of every shot prompt, or copy a ready-made prompt for one camera shot below.")
    block = P.prompt_block(res)
    st.code(block, language="text")
    st.download_button("Download prompt block", block, "prompt_block.txt")
    ep = st.selectbox("Episode", res["episodes"])
    mine = [s for s in res["shots"] if s["ep"] == ep]
    for s in mine[:12]:
        with st.expander(f"{s['scene']} · {s['at']} — {s['text'][:70]}..."):
            st.code(P.shot_prompt(res, s), language="text")
    st.caption(f"{len(mine)} shots in this episode, first 12 shown.")
    st.download_button("Download all shot prompts for this episode", "\n\n----------\n\n".join(f"[{s['scene']} {s['at']}]\n" + P.shot_prompt(res, s) for s in mine), "shot_prompts.txt")

st.caption("Code does the parsing, quote checking and final prompts. The model only reads and judges. A person makes the final call.")
