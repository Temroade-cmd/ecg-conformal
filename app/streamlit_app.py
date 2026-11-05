"""App de démo ECG Conformal.

    streamlit run app/streamlit_app.py

L'app télécharge le modèle publié en release sur GitHub (le modèle ONNX, les scores de calibration
et quelques ECG du test), puis utilise le même code conformal que l'évaluation (src/ecgc/conformal.py).
"""

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st

# Streamlit Cloud installe seulement app/requirements.txt, pas le package ecgc (il demanderait PyTorch)
# on ajoute donc src/ au chemin pour importer conformal.py directement, il n'a besoin que de numpy
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from ecgc import conformal as cp
import ecg_tools
import plots

REPO = "https://github.com/Temroade-cmd/ecg-conformal"
MODEL_TAG = os.environ.get("ECGC_MODEL_TAG", "model-v1")
MODEL_DIR = Path(os.environ.get("ECGC_MODEL_DIR", HERE / "model"))
FILES = ["ecg_resnet.onnx", "meta.json", "samples.json"]
PORTFOLIO = "https://portfolio-temtem.netlify.app"
LINKEDIN = "https://www.linkedin.com/in/tamer-benturquia-648b25301"

st.set_page_config(page_title="ECG Conformal", page_icon=str(HERE / "favicon.png"), layout="wide")
st.html(f"<style>{(HERE / 'style.css').read_text()}</style>")


def band_svg():
    # petite bande d'ECG qui défile en haut, comme sur mon portfolio (animation en CSS seulement)
    beat = [(0, 15), (14, 15), (18, 12), (22, 15), (30, 15), (33, 18), (36, 2), (39, 25), (42, 15),
            (52, 15), (60, 10), (68, 15), (100, 15)]
    pts = []
    for k in range(48):  # deux moitiés identiques pour que la boucle ne se voie pas
        pts += [(x + 100 * k, y) for x, y in beat]
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4800 30" preserveAspectRatio="none">'
           f'<path d="{d}" fill="none" stroke="#000" stroke-width="1.2" vector-effect="non-scaling-stroke"/></svg>')
    return f'<div class="band"><img alt="" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"></div>'



def figure(svg, css_class="", alt=""):
    # on affiche le SVG comme une image, dans un bloc qui défile sur les petits écrans
    data = base64.b64encode(svg.encode()).decode()
    st.html(f'<div class="fig {css_class}"><img alt="{alt}" src="data:image/svg+xml;base64,{data}"></div>')


def caption(html):
    st.html(f'<p class="cap">{html}</p>')


# --- modèle et données ---

def download_model():
    # la première fois, on télécharge les fichiers de la release, ensuite ils restent sur le disque
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = MODEL_DIR / name
        if not target.exists():
            part = target.with_suffix(target.suffix + ".part")  # un téléchargement coupé ne laisse pas de fichier cassé
            urllib.request.urlretrieve(f"{REPO}/releases/download/{MODEL_TAG}/{name}", part)
            part.rename(target)


@st.cache_resource  # chargé une seule fois pour tous les visiteurs
def load_bundle(version):
    download_model()
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    samples = json.loads((MODEL_DIR / "samples.json").read_text())
    session = ort.InferenceSession(str(MODEL_DIR / "ecg_resnet.onnx"), providers=["CPUExecutionProvider"])
    cal = [{"pos": np.asarray(meta["calibration"][k]["pos"]), "neg": np.asarray(meta["calibration"][k]["neg"])} for k in meta["classes"]]
    X = np.asarray([s["signal"] for s in samples], dtype=np.float32)
    return meta, samples, session, cal, X


def run(session, x):
    return session.run(["prob"], {"ecg": x.astype(np.float32)})[0]


@st.cache_data(show_spinner=False, max_entries=64)
def predict(signal, version):
    session = load_bundle(version)[2]
    return run(session, signal[None])[0]


def decide(probs, cal, alpha):
    # réponse conforme pour un ECG, et seuils pour dessiner les zones des barres
    inc_pos, inc_neg = cp.mondrian_sets(probs[None, :], cal, alpha)
    states = cp.states(inc_pos, inc_neg)[0]
    bounds = [cp.decision_bounds(c["pos"], c["neg"], alpha) for c in cal]
    return states, bounds


# --- en-tête ---

st.html(
    f'<div class="runhead"><span>ecg-conformal</span><span>'
    f'<a href="{REPO}">code</a><a href="{PORTFOLIO}">portfolio</a><a href="{LINKEDIN}">T. Benturquia</a></span></div>'
    + band_svg()
    + '<h1 class="title">A 12-lead ECG classifier that says when it cannot tell.</h1>'
    + '<p class="abstract"><b>Abstract</b>A probability of 0.74 does not say how often the model is wrong when it says 0.74. '
    "This network reads ECGs from PTB-XL and, for each of five diagnoses, either commits or answers that it cannot tell. "
    "You set the error rate. It is checked on patients the model never saw.</p>"
)

version = MODEL_TAG  # passé aux fonctions en cache, un nouveau modèle a un nouveau nom de release
meta, samples, session, cal, X = load_bundle(version)
classes, names = meta["classes"], [meta["class_names"][k] for k in meta["classes"]]

tab_read, tab_method, tab_results = st.tabs(["1  Read an ECG", "2  Method", "3  Results"])


# --- onglet 1, lire un ECG ---

with tab_read:
    # ECG envoyé par l'utilisateur, gardé dans la session
    with st.expander("Use your own ECG"):
        st.markdown(
            "A CSV with 12 columns in the order I, II, III, aVR, aVL, aVF, V1 to V6 (or a header with those names), "
            "in millivolts, at least 10 seconds. Nothing is stored. "
            "Keep in mind the guarantee was calibrated on PTB-XL, another device or hospital may not respect it."
        )
        up_col, fs_col = st.columns([3, 1])
        with up_col:
            upload = st.file_uploader("ECG file", type=["csv", "txt"], label_visibility="collapsed")
        with fs_col:
            fs_in = st.selectbox("Sampling rate", [100, 250, 500, 1000], index=2, format_func=lambda v: f"{v} Hz")
        if upload is not None:
            try:
                sig, notes = ecg_tools.read_csv_ecg(upload.getvalue(), fs_in)
                st.session_state["mine"] = sig
                for n in notes:
                    st.caption(n)
                st.caption("Loaded. Pick “your ECG” below.")
            except ValueError as e:
                st.error(str(e))

    ids = [s["ecg_id"] for s in samples]
    options = ids + (["your ECG"] if "mine" in st.session_state else [])

    # par défaut on ouvre sur un ECG où le modèle hésite au moins une fois à 90%, c'est plus parlant
    start = 0
    for j in range(len(ids)):
        if "uncertain" in decide(predict(X[j], version), cal, 0.1)[0]:
            start = j
            break

    st.html('<p class="lbl">Record from the PTB-XL test fold</p>')
    picked = st.pills("Record", options, default=ids[start], label_visibility="collapsed")
    if picked is None:
        picked = ids[start]
    if picked == "your ECG":
        signal, truth, title = st.session_state["mine"], None, "Your ECG"
    else:
        rec = samples[ids.index(picked)]
        signal = X[ids.index(picked)]
        truth = [k in rec["labels"] for k in classes]
        title = f"PTB-XL record {picked}"
    probs = predict(signal, version)

    c1, c2 = st.columns([2.2, 2], vertical_alignment="bottom")
    with c1:
        level = st.slider("Guarantee", min_value=80, max_value=99, value=90, format="%d%%")
    with c2:
        target = st.selectbox("Diagnosis to inspect", range(len(classes)), index=int(np.argmax(probs)),
                              format_func=lambda k: names[k])
    alpha = 1 - level / 100
    states, bounds = decide(probs, cal, alpha)

    # figure 1, l'ECG
    figure(plots.ecg_sheet(signal, meta["leads"], meta["fs"]), "sheet", "12-lead ECG")
    caption(f"<b>Fig. 1.</b> {title}. 10 s at 100 Hz, 25 mm/s, 10 mm/mV. The long strip at the bottom is lead II.")

    # quelques mesures simples sur le rythme, et l'accord avec le cardiologue
    m = ecg_tools.rhythm(signal)
    decided = [i for i, s_ in enumerate(states) if s_ in ("present", "absent")]
    agree = None if truth is None else sum((states[i] == "present") == truth[i] for i in decided)
    cells = [
        (f"{m['hr']:.0f}<small>bpm</small>" if m else "n/a", "heart rate, lead II"),
        (f"{m['rr_ms']:.0f}<small>ms</small>" if m else "n/a", "RR interval" + ("" if not m else (", regular" if m["regular"] else ", irregular"))),
        (f"{len(decided)}<small>of 5</small>", f"answers given at {level}%"),
        (f"{agree}<small>of {len(decided)}</small>" if agree is not None else "n/a", "agree with the cardiologist"),
    ]
    st.html('<div class="stats">' + "".join(f'<div><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in cells) + "</div>")

    # figure 2 la réponse, figure 3 où tombe cet ECG
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.html('<div class="legend"><span><i class="sw-a"></i>absent</span><span><i class="sw-u"></i>can\'t tell</span>'
                '<span><i class="sw-p"></i>present</span><span>○ this ECG</span></div>')
        figure(plots.readout(probs, bounds, states, names, truth), "", "Model answer per diagnosis")
        miss = 100 - level
        rate = f"1 time in {100 // miss}" if 100 % miss == 0 else f"{miss} times in 100"
        caption(f"<b>Fig. 2.</b> Answer per diagnosis at {level}%. A real finding is ruled out at most {rate}, "
                f"and an absent one is reported at most {rate}. Grey and red zones are where the model commits, "
                "the hatched zone is where it cannot tell.")
    with right:
        figure(plots.position(cal[target], probs[target], bounds[target], names[target]), "",
               "This ECG among calibration ECGs")
        caption(f"<b>Fig. 3.</b> {names[target]}. Scores of the {meta['results']['n_calibration']:,} calibration ECGs, "
                "with and without the diagnosis, and where this ECG falls. The decision only depends on how unusual "
                "its score is for each group.")


# --- onglet 2, méthode ---

with tab_method:
    st.html("""<div class="prose">
<h3>Data</h3>
<p>PTB-XL holds 21,799 clinical 12-lead ECGs of 10 seconds, annotated by cardiologists. Each record is grouped into
five diagnostic classes, normal, myocardial infarction, ST/T change, conduction disturbance and hypertrophy, and one
ECG can have several. The dataset comes split into ten folds with no patient shared between folds.</p>
<table class="plain"><thead><tr><th>Folds</th><th>Role</th></tr></thead><tbody>
<tr><td>1 to 7</td><td>training</td></tr><tr><td>8</td><td>choosing the best epoch</td></tr>
<tr><td>9</td><td>calibration</td></tr><tr><td>10</td><td>test, used once</td></tr></tbody></table>
<h3>Model</h3>
<p>A 1D residual network of three blocks (about half a million parameters) reads the twelve leads at 100 Hz and gives
one probability per diagnosis. It follows the residual baseline of the PTB-XL benchmark (Strodthoff et al., 2021).</p>
<h3>From a probability to an answer</h3>
<ol>
<li>On the calibration fold, keep two lists for each diagnosis, the scores of ECGs that truly have it and of those that do not.</li>
<li>For a new ECG, measure how unusual its score is in each list. This gives two conformal p-values.</li>
<li>Keep every answer whose p-value is above the error rate. One answer left means the model commits. Two left means it cannot tell.</li>
</ol>
</div>""")
    st.latex(r"p_{\text{present}} = \frac{1 + \#\{\,i \in \text{positives} : 1 - \hat p_i \ge 1 - \hat p\,\}}{n_{\text{pos}} + 1}"
             r"\qquad p_{\text{absent}} = \frac{1 + \#\{\,i \in \text{negatives} : \hat p_i \ge \hat p\,\}}{n_{\text{neg}} + 1}")
    st.html("""<div class="prose">
<p>If calibration and new ECGs come from the same population, a real finding is then ruled out with probability at
most the error rate, and so is an absent one, separately. Rare diagnoses are protected as much as frequent ones.</p>
<h3>What it does not do</h3>
<p>The guarantee is an average over ECGs, not a promise about one patient. It assumes the new ECG looks like the
calibration fold, so another device, hospital or population needs recalibration. The shaded explanation shows what the
network reacts to, not why a cardiologist would agree. This is a research demo, not a medical device.</p>
</div>""")


# --- onglet 3, résultats ---

with tab_results:
    r = meta["results"]
    mac = r["macro_auroc"]
    st.html('<div class="stats">'
            f'<div><div class="v">{mac["value"]:.3f}</div><div class="k">macro AUROC, test</div></div>'
            f'<div><div class="v">{mac["ci95"][0]:.3f}<small>to {mac["ci95"][1]:.3f}</small></div><div class="k">95% bootstrap interval</div></div>'
            '<div><div class="v">0.930</div><div class="k">reference, Strodthoff 2021</div></div>'
            f'<div><div class="v">{r["n_test"]:,}</div><div class="k">test ECGs, fold 10</div></div>'
            "</div>")

    pc = r["mondrian"]["0.1"]
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    rows = "".join(
        f"<tr><td>{meta['class_names'][k]}</td><td>{r['auroc_per_class'][k]:.3f}</td><td>{pct(pc[k]['coverage_pos'])}</td>"
        f"<td>{pct(pc[k]['coverage_neg'])}</td><td>{pct(pc[k]['decided_rate'])}</td><td>{pct(pc[k]['accuracy_when_decided'])}</td></tr>"
        for k in classes
    )
    st.html('<div class="prose" style="max-width:860px"><table><thead><tr><th>Diagnosis</th><th>AUROC</th>'
            "<th>real findings kept</th><th>absences kept</th><th>answers given</th><th>right when answering</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")
    caption("<b>Table 1.</b> Test fold, guarantee at 90%. “Kept” rates should sit near 90%, "
            "“answers given” is the share of decisions that are not “can't tell”.")

    crc = r["crc"]["0.1"]
    st.html(f'<div class="prose"><h3>A simpler output</h3><p>Instead of one answer per diagnosis, a single threshold can be '
            f"calibrated so that the list of diagnoses above it misses at most 10% of an ECG's true diagnoses on average. "
            f"The threshold comes out at {crc['threshold']:.2f}. On the test fold the list misses {pct(crc['fnr'])} of true "
            f"diagnoses, with {crc['mean_set_size']:.2f} diagnoses listed per ECG for {crc['mean_true_labels']:.2f} true ones "
            "(conformal risk control, Angelopoulos et al., 2024).</p></div>")


st.html(f'<div class="foot"><span>Research demo, not a medical device.</span>'
        f'<span>Data PTB-XL 1.0.3, PhysioNet, CC BY 4.0</span>'
        f'<span><a href="{REPO}">github</a> &nbsp; <a href="{PORTFOLIO}">portfolio</a> &nbsp; <a href="{LINKEDIN}">linkedin</a></span></div>')
