"""
Production RAG chatbot for construction rules and regulations.
Streamlit app: scenario in, applicable rules out (strictly from PDF context).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone as PineconeClient


# ---------------------------------------------------------------------------
# Environment and config
# ---------------------------------------------------------------------------
def get_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env: {name}")
    return val.strip()


def log_feedback(kind: str, question: str, answer: str) -> None:
    """
    Append simple feedback to a local log file for later analysis.
    kind: 'helpful' or 'needs_improvement'
    """
    try:
        root = Path(__file__).resolve().parent
        log_path = root / "feedback.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{kind.upper()} | Q: {question.replace(os.linesep, ' ')} | A: {answer.replace(os.linesep, ' ') }\n")
    except Exception:
        # Feedback should never break the app; fail silently.
        pass


@st.cache_resource
def get_vectorstore():
    """Cached Pinecone vector store and embeddings."""
    embed_model = get_env("EMBED_MODEL")
    api_key = get_env("OPENAI_API_KEY")
    index_name = get_env("PINECONE_INDEX")
    pinecone_key = get_env("PINECONE_API_KEY")
    pc = PineconeClient(api_key=pinecone_key)
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model=embed_model, openai_api_key=api_key)
    return PineconeVectorStore(index=index, embedding=embeddings)


def get_llm():
    return ChatOpenAI(
        model=get_env("CHAT_MODEL"),
        openai_api_key=get_env("OPENAI_API_KEY"),
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# System prompt: strict grounding, no hallucination, English only
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a construction rules assistant. You answer ONLY using the provided PDF context.

RULES (CRITICAL):
- Base every answer STRICTLY on the \"Relevant PDF context\" below. Do not use any other knowledge.
- If the context does not contain a rule that applies to the user's scenario, you MUST respond with exactly: \"This scenario’s relevant rule is not present in this PDF.\"
- Do NOT guess, approximate, or invent any rule. If in doubt, say the rule is not found.
- Prefer precision over length. Be clear and actionable.
- All answers must be in clear, professional English suitable for construction engineers, site supervisors, and workers.

OUTPUT FORMAT (follow exactly):

1. Understanding the Scenario
   - What the user is planning/building
   - Location context (road-facing, residential, commercial)

2. Applicable Rules
   - Bullet points
   - Exact rule explanation from the context only
   - Only what applies to the scenario

3. Instructions for Workers
   - Simple actionable steps
   - On-site friendly language

4. Source Reference
   - Page number(s) from PDF

Relevant PDF context:
{context}
"""

USER_PROMPT = """User scenario / question:
{question}
"""


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs
    )


def build_chain(vectorstore, llm, k: int = 5):
    """Retriever + LLM chain; answer only from retrieved context."""
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Construction Rules RAG Bot",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Global styles (premium, 3D-style construction dashboard, similar to SaaS OS layout)
    st.markdown(
        """
        <style>
          /* Full app background: soft multi-color gradient, independent of theme */
          body,
          [data-testid="stAppViewContainer"] {
            background:
              radial-gradient(circle at top left, #1d1258 0, #0f172a 38%, #020617 60%),
              radial-gradient(circle at bottom right, #172554 0, #020617 55%, #020617 100%) !important;
            color: #e5e7eb;
          }
          [data-testid="stHeader"] {
            background: transparent !important;
          }

          .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1200px;
          }

          /* Hero: 3D, glass + neon edge, theme-agnostic */
          .crb-hero {
            position: relative;
            border-radius: 20px;
            padding: 1.1rem 1.4rem 0.9rem 1.4rem;
            margin-bottom: 1.25rem;
            background:
              radial-gradient(140% 160% at 0% 0%, rgba(251, 191, 36, 0.35), transparent 55%),
              radial-gradient(120% 160% at 100% 0%, rgba(56, 189, 248, 0.32), transparent 60%),
              linear-gradient(135deg, #020617, #020617 20%, #020617 45%, #020617 80%, #020617);
            box-shadow:
              0 28px 80px rgba(15, 23, 42, 0.85),
              0 0 0 1px rgba(148, 163, 184, 0.32),
              inset 0 1px 0 rgba(248, 250, 252, 0.16);
            overflow: hidden;
            color: #e5e7eb;
          }
          .crb-hero::before {
            content: "";
            position: absolute;
            inset: -40%;
            background:
              radial-gradient(circle at 0% 0%, rgba(251, 191, 36, 0.22), transparent 55%),
              radial-gradient(circle at 100% 0%, rgba(56, 189, 248, 0.20), transparent 55%);
            opacity: 0.9;
            mix-blend-mode: screen;
            pointer-events: none;
          }
          .crb-hero-inner {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            gap: 0.7rem;
          }
          .crb-top-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
          }
          .crb-logo-wrap {
            display: flex;
            align-items: center;
            gap: 0.65rem;
          }
          .crb-logo-circle {
            width: 34px;
            height: 34px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: radial-gradient(circle at 30% 0%, #facc15, #f97316);
            box-shadow:
              0 10px 25px rgba(15, 23, 42, 0.85),
              0 0 0 1px rgba(248, 250, 252, 0.4);
          }
          .crb-app-name {
            font-weight: 750;
            letter-spacing: -0.03em;
            font-size: 1.15rem;
            color: #f9fafb;
          }
          .crb-app-sub {
            font-size: 0.8rem;
            color: rgba(226, 232, 240, 0.78);
          }
          .crb-user-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.7);
            font-size: 0.75rem;
            color: rgba(226, 232, 240, 0.9);
          }
          .crb-user-role {
            font-size: 0.7rem;
            padding: 0.12rem 0.45rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.16);
            color: rgba(190, 242, 100, 0.95);
            border: 1px solid rgba(74, 222, 128, 0.5);
          }


          /* Cards: floating, subtle 3D */
          .crb-card {
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            background:
              linear-gradient(145deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.88));
            box-shadow:
              0 18px 45px rgba(15, 23, 42, 0.85),
              0 0 0 1px rgba(148, 163, 184, 0.35);
            color: #e5e7eb;
          }
          .crb-card h4, .crb-card h3, .crb-card h2 {
            margin-top: 0;
          }
          .crb-muted {
            color: rgba(148, 163, 184, 0.95);
          }

          /* Make Streamlit text area + button blend with the 3D card */
          .crb-card textarea {
            border-radius: 14px !important;
            border: 1px solid rgba(148, 163, 184, 0.5) !important;
            background: radial-gradient(circle at 0 0, rgba(51, 65, 85, 0.4), rgba(15, 23, 42, 0.95)) !important;
            color: #e5e7eb !important;
            box-shadow:
              inset 0 1px 0 rgba(248, 250, 252, 0.08),
              0 14px 30px rgba(15, 23, 42, 0.75);
          }
          .crb-card textarea:focus {
            border-color: rgba(250, 204, 21, 0.9) !important;
            box-shadow:
              0 0 0 1px rgba(250, 204, 21, 0.9),
              0 18px 40px rgba(15, 23, 42, 0.9),
              inset 0 1px 0 rgba(248, 250, 252, 0.12);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # App shell header (inspired by OS-style dashboards)
    st.markdown(
        """
        <div class="crb-hero">
          <div class="crb-hero-inner">
            <div class="crb-top-row">
              <div class="crb-logo-wrap">
                <div class="crb-logo-circle">🏗️</div>
                <div>
                  <div class="crb-app-name">Construction Rules OS</div>
                  <div class="crb-app-sub">Daily co-pilot for code-compliant site decisions</div>
                </div>
              </div>
              <div class="crb-user-pill">
                <span>Supervisor Mode</span>
                <span class="crb-user-role">Live</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load vector store once
    try:
        vectorstore = get_vectorstore()
    except Exception as e:
        st.error(f"Vector store could not be loaded. Please run `python ingest.py` first. Error: {e}")
        st.stop()

    llm = get_llm()

    # Main layout: chat on the left, \"today\" panel on the right
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown("<div class='crb-card'>", unsafe_allow_html=True)
        st.markdown("#### Chat with your rules assistant")
        st.markdown(
            "<p class='crb-muted'>Describe the construction scenario and location. The assistant will respond with only the rules that apply.</p>",
            unsafe_allow_html=True,
        )
        scenario = st.text_area(
            "🧱 Scenario / Question",
            height=140,
            placeholder="Example: I am constructing a residential building on a main road-facing plot. What are the applicable setback and signage regulations?",
            help="Write what you are building, where it is located, and what decision you need on-site.",
        )
        submit = st.button("📌 Get Applicable Rules", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='crb-card'>", unsafe_allow_html=True)
        st.markdown("#### 📅 Today's checks")
        st.markdown(
            """
            - Capture one scenario per query for precise rules.
            - Mention road type, land use (residential/commercial), and any special conditions.
            - If the PDF has no matching rule, the assistant will say so instead of guessing.
            """,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        if not (scenario and scenario.strip()):
            st.warning("Please write your scenario first.")
            return

        chain = build_chain(vectorstore, llm)

        with st.spinner("Searching rules from PDF context..."):
            try:
                answer = chain.invoke(scenario.strip())
            except Exception as e:
                st.error(f"Error: {e}")
                return
        st.success("Done. Retrieved from PDF context.")
        title = "### 🧾 Answer"
        # Formatted sections
        st.markdown("---")
        st.markdown(title)
        st.markdown(answer)
        st.markdown("---")

        # Lightweight feedback buttons for continuous improvement
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            if st.button("✅ This answer was helpful", use_container_width=True):
                log_feedback("helpful", scenario.strip(), answer)
                st.toast("Thanks, marked as helpful.", icon="✅")
        with fb_col2:
            if st.button("⚠️ Needs improvement", use_container_width=True):
                log_feedback("needs_improvement", scenario.strip(), answer)
                st.toast("Noted. We will review this scenario.", icon="⚠️")

    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.85rem;'>Answers are based only on uploaded PDF rules. Safety first.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
