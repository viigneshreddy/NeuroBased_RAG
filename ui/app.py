 ui/app.py
# A tiny web UI to demo EEG -> State -> RAG -> LLM

import sys
from pathlib import Path
# Ensure project root is on the import path when running with `streamlit run ui/app.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from eeg.simulate_eeg import simulate_eeg_sample
from eeg.classifier import classify_eeg_state
from rag.rag_pipeline import generate_response

# ---------- Page setup ----------
st.set_page_config(page_title="Adaptive Neuro-RAG Demo", page_icon="🧠", layout="centered")
st.title("🧠 Adaptive Neuro-RAG (EEG-Guided Assistant)")

# ---------- Session state (persist across interactions) ----------
if "eeg_history" not in st.session_state:
    st.session_state.eeg_history = []     # store last few classified states
if "last_eeg" not in st.session_state:
    st.session_state.last_eeg = None      # store last EEG raw sample
if "last_state" not in st.session_state:
    st.session_state.last_state = None    # store last classified state

# ---------- Sidebar ----------
st.sidebar.header("Controls")

if st.sidebar.button("🧪 Simulate EEG Sample"):
    # 1) simulate raw EEG bands
    eeg = simulate_eeg_sample()
    st.session_state.last_eeg = eeg

    # 2) classify to a simple state
    state = classify_eeg_state(
        delta=eeg["delta"], theta=eeg["theta"], alpha=eeg["alpha"],
        beta=eeg["beta"], gamma=eeg["gamma"]
    )
    st.session_state.last_state = state

    # 3) keep a short history (last 5)
    st.session_state.eeg_history.append(state)
    st.session_state.eeg_history = st.session_state.eeg_history[-5:]

st.sidebar.markdown("---")
st.sidebar.caption("Tip: click **Simulate EEG Sample** before asking for an answer.")

# ---------- Main panel: show current EEG + state ----------
st.subheader("Current EEG Reading")
col1, col2 = st.columns(2, vertical_alignment="center")

with col1:
    st.write("**Raw bands (simulated):**")
    if st.session_state.last_eeg:
        st.json(st.session_state.last_eeg)
    else:
        st.info("No EEG sample yet. Click **Simulate EEG Sample** in the sidebar.")

with col2:
    st.write("**Detected state:**")
    if st.session_state.last_state:
        st.success(st.session_state.last_state)
    else:
        st.warning("No state yet.")

st.write("**Recent state history (last 5):**", ", ".join(st.session_state.eeg_history) or "—")

# ---------- Query + Answer ----------
st.subheader("Ask for Help")
query = st.text_area(
    "What do you need help with?",
    value="I have a big exam tomorrow and I'm feeling overwhelmed. How can I calm down and focus?",
    height=100,
)

go = st.button("💬 Get EEG-Aware Answer", type="primary")

if go:
    if not st.session_state.last_state:
        st.error("Please click **Simulate EEG Sample** in the sidebar first.")
    else:
        with st.spinner("Thinking…"):
            answer = generate_response(
                query=query,
                eeg_state=st.session_state.last_state,
                eeg_history=st.session_state.eeg_history
            )
        st.markdown("### Answer")
        st.markdown(answer)
        st.markdown("---")
        st.caption("This assistant is supportive and educational. It does **not** provide medical advice or diagnosis.")
