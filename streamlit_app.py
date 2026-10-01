"""
AI Accessibility Assistant - Streamlit UI
Uses the LangGraph in accessibility_graph.py (same folder), including the
human-in-the-loop review step.

Run from the backend folder:
    streamlit run streamlit_app.py
"""
import io
import json
import re
import uuid

import streamlit as st
from langgraph.types import Command
from pypdf import PdfReader

from accessibility_graph import MAX_CHARS, RULES, build_app

st.set_page_config(page_title="AI Accessibility Assistant", page_icon="♿", layout="wide")

PROFILE_LABELS = {
    "dyslexia": "Dyslexia / ADHD (short sentences, key points first)",
    "non_native": "Non-native English speaker (plain English)",
    "beginner": "Complete beginner (terms explained)",
}


@st.cache_resource
def get_graph():
    # One compiled graph for the whole server; MemorySaver keeps paused sessions by thread_id
    return build_app()


graph = get_graph()

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
DEFAULTS = {
    "stage": "input",        # input -> review -> done
    "thread_id": None,
    "review": None,          # data from the paused graph
    "final": None,           # data from the finished graph
    "grade_before": None,
    "original_shown": "",
    "uploader_key": 0,
    "pasted": "",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
ss = st.session_state


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def parse_glossary(raw: str) -> list:
    items = []
    for line in (raw or "").splitlines():
        if "::" in line:
            term, meaning = line.split("::", 1)
            items.append((term.strip(" -*•"), meaning.strip()))
    return items


def extract_text(uploaded) -> str:
    if uploaded.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded.getvalue()))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return uploaded.getvalue().decode("utf-8", errors="ignore")


def handle_output(out: dict) -> None:
    """The graph either paused for review (interrupt) or finished."""
    ss.grade_before = out.get("grade_before")
    if "__interrupt__" in out:
        ss.review = out["__interrupt__"][0].value
        ss.stage = "review"
    else:
        ss.final = {
            "simplified": out["simplified"],
            "glossary": parse_glossary(out.get("glossary", "")),
            "grade_before": out.get("grade_before"),
            "grade_after": out.get("grade_after"),
        }
        ss.stage = "done"


def run_start(source: str, profile: str) -> None:
    ss.thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": ss.thread_id}}
    handle_output(graph.invoke({"source": source, "profile": profile}, config))


def run_resume(answer: str) -> None:
    config = {"configurable": {"thread_id": ss.thread_id}}
    handle_output(graph.invoke(Command(resume=answer), config))


def reset() -> None:
    ss.stage = "input"
    ss.thread_id = None
    ss.review = None
    ss.final = None
    ss.original_shown = ""
    ss.pasted = ""
    ss.uploader_key += 1  # new key = empty file uploader


def inject_css(font_px: int, easy_font: bool, contrast: bool) -> None:
    family = "'Lexend', sans-serif" if easy_font else "inherit"
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;600&display=swap');
    .simplified-box {{
        font-family: {family}; font-size: {font_px}px; line-height: 1.7;
        letter-spacing: .02em; padding: 1rem 1.25rem;
        border: 2px solid #8888; border-radius: 10px;
    }}
    """
    if contrast:
        css += """
        .stApp { background: #000 !important; }
        .stApp, .stApp p, .stApp label, .stApp span, .stApp li,
        .stApp h1, .stApp h2, .stApp h3 { color: #fff !important; }
        .simplified-box { background: #000; color: #fff; border: 2px solid #ffeb3b; }
        """
    css += "</style>"
    st.markdown(css, unsafe_allow_html=True)


def show_box(text: str) -> None:
    safe = text.replace("<", "&lt;")  # model output is never treated as HTML
    st.markdown(f'<div class="simplified-box">\n\n{safe}\n\n</div>', unsafe_allow_html=True)


def listen_button(text: str) -> None:
    """Browser text-to-speech (no extra library). Click again to stop."""
    # JSON string literal; "<" is escaped so the text can never close the script tag
    payload = json.dumps(re.sub(r"[*#_`]", " ", text)).replace("<", "\\u003c")
    page = f"""
    <button id="b" style="font-size:16px;padding:8px 16px;border-radius:8px;cursor:pointer">
      🔊 Listen / Stop
    </button>
    <script>
      const t = {payload};
      document.getElementById('b').onclick = () => {{
        const s = window.speechSynthesis;
        if (!s) {{ alert('Your browser does not support audio.'); return; }}
        if (s.speaking) {{ s.cancel(); return; }}
        s.speak(new SpeechSynthesisUtterance(t));
      }};
    </script>
    """.strip()
    if hasattr(st, "iframe"):          # newer Streamlit
        st.iframe(page, height=60)
    else:                              # older Streamlit
        import streamlit.components.v1 as components

        components.html(page, height=60)


# ----------------------------------------------------------------------------
# Sidebar: reader profile + display settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Reader profile")
    profile = st.selectbox(
        "Who is this for?",
        list(RULES.keys()),
        format_func=lambda p: PROFILE_LABELS.get(p, p),
        disabled=ss.stage != "input",
    )
    st.header("Display")
    font_px = st.slider("Text size", 16, 32, 20)
    easy_font = st.checkbox("Easy-to-read font (Lexend)", value=True)
    contrast = st.checkbox("High contrast")

inject_css(font_px, easy_font, contrast)

st.title("♿ AI Accessibility Assistant")
st.caption("Any information, in a form that works for you. Powered by Gemma.")

# ----------------------------------------------------------------------------
# Stage 1: input
# ----------------------------------------------------------------------------
if ss.stage == "input":
    st.subheader("1. Your document")
    tab_paste, tab_file = st.tabs(["Paste text or link", "Upload PDF / TXT"])

    with tab_paste:
        st.text_area(
            "Paste a contract, medical report, or a link to a web page",
            key="pasted",
            height=220,
        )

    uploaded_text = ""
    with tab_file:
        uploaded = st.file_uploader(
            "Choose a file", type=["pdf", "txt"], key=f"uploader_{ss.uploader_key}"
        )
        if uploaded is not None:
            try:
                uploaded_text = extract_text(uploaded)
                if uploaded_text.strip():
                    st.success(f"Loaded {uploaded.name} ({len(uploaded_text):,} characters)")
                    if len(uploaded_text) > MAX_CHARS:
                        st.warning("The file is long, so only the first part will be processed.")
                else:
                    st.error("No text found. This may be a scanned PDF (images only).")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read this file: {e}")

    source = uploaded_text.strip() or ss.pasted.strip()

    if st.button("Make it accessible", type="primary", key="go"):
        if not source:
            st.error("Please paste some text or a link, or upload a file first.")
        elif re.fullmatch(r"\S+\.(pdf|txt)", source, re.IGNORECASE):
            st.error("To use a PDF or TXT file, upload it in the 'Upload' tab.")
        else:
            ss.original_shown = (
                "(Content loaded from the link)"
                if source.lower().startswith("http")
                else source[:MAX_CHARS]
            )
            with st.spinner("Simplifying and fact-checking... this can take a minute."):
                try:
                    run_start(source, profile)
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(
                        "Could not reach the model. Check GOOGLE_API_KEY and LLM_PROVIDER "
                        f"in your .env file.\n\nDetails: {e}"
                    )

# ----------------------------------------------------------------------------
# Stage 2: human review (the graph is paused here)
# ----------------------------------------------------------------------------
elif ss.stage == "review":
    info = ss.review
    st.subheader("2. Your review")
    st.caption("The AI paused so you can check the result before it is finalized.")

    if info["verdict"] == "PASS":
        st.success("Accuracy check passed: no facts, numbers or warnings were changed.")
    else:
        st.warning(
            "The accuracy check flagged possible issues. "
            f"Please read carefully. {info.get('verifier_notes', '')}"
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Original**")
        st.text_area("original", ss.original_shown, height=320, label_visibility="collapsed")
    with right:
        st.markdown("**Simplified**")
        show_box(info["simplified"])

    feedback = st.text_input(
        "Want a change? Say what to fix, then press Request changes",
        placeholder='e.g. "shorter sentences" or "explain the dosage more clearly"',
    )

    b1, b2, _ = st.columns([1, 1.3, 4])
    approve = b1.button("Approve", type="primary", key="approve")
    change = b2.button("Request changes", key="change")
    listen_button(info["simplified"])

    if approve:
        with st.spinner("Finishing up..."):
            try:
                run_resume("approve")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Model error: {e}")
    elif change:
        if not feedback.strip():
            st.error("Type what you want changed first, or press Approve.")
        else:
            with st.spinner("Rewriting with your feedback..."):
                try:
                    run_resume(feedback.strip())
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Model error: {e}")

# ----------------------------------------------------------------------------
# Stage 3: final result
# ----------------------------------------------------------------------------
elif ss.stage == "done":
    res = ss.final
    st.subheader("3. Final result")

    c1, c2 = st.columns(2)
    c1.metric("Reading grade (original)", res["grade_before"])
    c2.metric(
        "Reading grade (simplified)",
        res["grade_after"],
        delta=round(res["grade_after"] - res["grade_before"], 1),
        delta_color="inverse",
    )

    show_box(res["simplified"])
    listen_button(res["simplified"])

    st.markdown("**Glossary**")
    if res["glossary"]:
        for term, meaning in res["glossary"]:
            with st.expander(term):
                st.write(meaning)
    else:
        st.write("No difficult terms found.")

    st.button("Start over", on_click=reset, key="restart")
