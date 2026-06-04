"""VisaMate — Streamlit chat UI with conversational context capture."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

from config import LLM_PROVIDER, validate_environment  # noqa: E402
from conversation_router import generate_follow_up, route_question  # noqa: E402
from llm_factory import PROVIDER_MODEL, get_llm  # noqa: E402
from rag_pipeline import answer_question  # noqa: E402
from user_profile import UserProfile, extract_attributes, merge_into_profile  # noqa: E402


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VisaMate",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling — warm Australian palette (sand + eucalyptus)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg-deep:        #0D1411;
        --bg-surface:     #1A2620;
        --bg-elevated:    #21302A;
        --border-subtle:  #2B3D34;
        --border-strong:  #3D5447;
        --text-primary:   #E8E6E1;
        --text-secondary: #A8B0A4;
        --text-muted:     #6B7268;
        --eucalyptus:     #8FBC8F;
        --eucalyptus-dim: #6B9A6B;
        --eucalyptus-glow: rgba(143, 188, 143, 0.15);
    }

    /* Page background */
    .stApp {
        background-color: var(--bg-deep);
        color: var(--text-primary);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--eucalyptus) !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] .stCaption {
        color: var(--text-secondary) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border-subtle) !important;
        margin: 1.2rem 0;
    }

    /* Profile detail rows */
    .profile-row {
        padding: 5px 0;
        font-size: 0.93rem;
        color: var(--text-primary);
    }
    .profile-row b {
        color: var(--eucalyptus);
        font-weight: 600;
    }

    /* Main pane headings */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-family: 'Georgia', 'Times New Roman', serif;
        letter-spacing: -0.01em;
    }
    h1 {
        color: var(--eucalyptus) !important;
    }

    /* Chat message bubbles */
    [data-testid="stChatMessage"] {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: var(--bg-elevated);
        color: var(--text-primary);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        font-weight: 400;
        text-align: left;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--eucalyptus-glow);
        color: var(--eucalyptus);
        border-color: var(--eucalyptus-dim);
    }

/* Chat input container — kill all wrapper backgrounds AND borders */
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"] {
        background-color: var(--bg-deep) !important;
        # border-top: 1px solid var(--border-subtle);
    }

    /* Strip every nested wrapper Streamlit adds — backgrounds, borders, outlines */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] > div > div > div,
    [data-testid="stChatInput"] [data-baseweb] {
        background-color: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* The textarea itself */
    [data-testid="stChatInput"] textarea {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px;
        color: var(--text-primary) !important;
        padding: 12px 56px 12px 16px !important;
        min-height: 52px !important;
        font-family: inherit;
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--eucalyptus-dim) !important;
        box-shadow: 0 0 0 2px var(--eucalyptus-glow) !important;
        outline: none !important;
    }

    /* Send button — small eucalyptus-tinted, always visible background, sits inside textarea */
    [data-testid="stChatInput"] button {
        background-color: var(--eucalyptus-glow) !important;
        border: 1px solid var(--eucalyptus-dim) !important;
        color: var(--eucalyptus) !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        min-height: 32px !important;
        padding: 0 !important;
        border-radius: 8px !important;
        position: absolute !important;
        right: 20px !important;
        bottom: 10px !important;
    }
    [data-testid="stChatInput"] button:hover {
        background-color: var(--eucalyptus-dim) !important;
        color: var(--bg-deep) !important;
    }
    [data-testid="stChatInput"] button svg {
        width: 16px !important;
        height: 16px !important;
    }

    /* Position the input parent so the button anchors inside */
    [data-testid="stChatInput"] > div > div {
        position: relative !important;
    }

    /* Source expanders */
    [data-testid="stExpander"] {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-secondary) !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: var(--eucalyptus) !important;
    }

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
    }

    /* Links */
    a {
        color: var(--eucalyptus) !important;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }

    /* Selectbox in settings */
    [data-baseweb="select"] {
        background-color: var(--bg-elevated) !important;
    }
    [data-baseweb="select"] > div {
        background-color: var(--bg-elevated) !important;
        border-color: var(--border-subtle) !important;
        color: var(--text-primary) !important;
    }

    /* Spinner / status text */
    [data-testid="stSpinner"] {
        color: var(--eucalyptus) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = UserProfile()
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "greeted" not in st.session_state:
    st.session_state.greeted = False


# ---------------------------------------------------------------------------
# Cached LLM
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cached_llm(provider: str):
    return get_llm(provider)


# ---------------------------------------------------------------------------
# Sidebar — three sections only
# ---------------------------------------------------------------------------
with st.sidebar:
    # --- Section 1: Greeting ---
    name = st.session_state.profile.name
    if name:
        st.markdown(f"### Hey {name}! 👋")
    else:
        st.markdown("### G'day! 👋")
    st.caption("Your visa buddy in Australia")

    st.divider()

    # --- Section 2: Your details ---
    st.markdown("### Your details")
    profile_dict = st.session_state.profile.to_sidebar_dict()
    if not profile_dict:
        st.caption(
            "I'll pick up details as we chat — your course, English score, "
            "where you're at — so my answers actually fit your situation."
        )
    else:
        for label, value in profile_dict.items():
            st.markdown(
                f"<div class='profile-row'><b>{label}:</b> {value}</div>",
                unsafe_allow_html=True,
            )
        st.caption("")
        if st.button("Start over", use_container_width=True):
            st.session_state.profile = UserProfile()
            st.session_state.messages = []
            st.session_state.greeted = False
            st.rerun()

    st.divider()

    # --- Section 3: Footer / settings ---
    with st.expander("⚙️ Settings"):
        provider_choice = st.selectbox(
            "LLM Provider",
            options=["groq", "openai", "gemini"],
            index=["groq", "openai", "gemini"].index(LLM_PROVIDER),
            help="Groq is free and fast. OpenAI is paid. Gemini has a free tier.",
        )
        st.caption(f"Model: `{PROVIDER_MODEL[provider_choice]}`")
        if st.button("Clear chat (keep details)", use_container_width=True):
            st.session_state.messages = []
            st.session_state.greeted = False
            st.rerun()

    st.caption(
        "⚠️ Educational project. For real decisions, check "
        "[Home Affairs](https://immi.homeaffairs.gov.au/) or a "
        "MARA-registered migration agent."
    )


# ---------------------------------------------------------------------------
# Main pane — header
# ---------------------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("VisaMate")
    st.markdown(
        "**Your visa buddy in Australia** — for international students "
        "figuring out the path from student visa to permanent residency."
    )

# Validate env once
try:
    validate_environment()
except RuntimeError as e:
    st.error(f"⚠️ Configuration issue: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Auto-greeting on a fresh chat
# ---------------------------------------------------------------------------
if not st.session_state.greeted and not st.session_state.messages:
    greeting = (
        "Hey! 👋 I'm VisaMate — here to help you sort out the Australian visa "
        "maze, especially the international student journey. "
        "Before we dive in, **what's your name** and **what can I help you "
        "sort out today**? (Could be anything — student visa costs, PR "
        "pathways after graduating, post-study work options, whatever's on "
        "your mind.)"
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": greeting, "sources": []}
    )
    st.session_state.greeted = True


# ---------------------------------------------------------------------------
# Replay chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f"**{i}. {src['title']}** "
                        f"(chunk {src['chunk_id']})  \n"
                        f"🔗 [{src['source']}]({src['source']})"
                    )
                    st.caption(src["preview"])


# ---------------------------------------------------------------------------
# Pull provider_choice from settings expander (it lives in `with st.sidebar`)
# Default to LLM_PROVIDER if settings expander never opened
# ---------------------------------------------------------------------------
provider_choice = st.session_state.get("provider_choice_override", LLM_PROVIDER)


# ---------------------------------------------------------------------------
# Handle new input
# ---------------------------------------------------------------------------
user_input = st.chat_input("Type your question here…")

if st.session_state.pending_question and not user_input:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None


if user_input:
    llm = cached_llm(provider_choice)

    # 1. Append + render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Extract profile attributes silently
    extracted = extract_attributes(user_input, llm)
    updated_fields = merge_into_profile(st.session_state.profile, extracted)

    # 3. Route: answer or ask?
    decision = route_question(user_input, st.session_state.profile, llm)

    with st.chat_message("assistant"):
        if decision.route == "ask":
            with st.spinner("One sec..."):
                follow_up = generate_follow_up(
                    user_input,
                    st.session_state.profile,
                    decision.missing_fields,
                    llm,
                )
            st.markdown(follow_up)
            st.session_state.messages.append(
                {"role": "assistant", "content": follow_up, "sources": []}
            )
        else:
            with st.spinner("Looking it up..."):
                try:
                    result = answer_question(
                        user_input,
                        llm=llm,
                        profile=st.session_state.profile,
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    st.stop()

            st.markdown(result.answer)

            sources = [
                {
                    "title": d.metadata.get("title", "Untitled"),
                    "source": d.metadata.get("source", ""),
                    "chunk_id": d.metadata.get("chunk_id", "N/A"),
                    "preview": d.page_content[:300] + (
                        "…" if len(d.page_content) > 300 else ""
                    ),
                }
                for d in result.sources
            ]

            with st.expander(f"📚 Sources ({len(sources)})"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(
                        f"**{i}. {src['title']}** "
                        f"(chunk {src['chunk_id']})  \n"
                        f"🔗 [{src['source']}]({src['source']})"
                    )
                    st.caption(src["preview"])

            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "sources": sources}
            )

    # Sidebar refresh if profile changed
    if updated_fields:
        st.rerun()