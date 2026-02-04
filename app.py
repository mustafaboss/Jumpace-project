"""
Production RAG chatbot for construction rules and regulations.
Streamlit app: scenario in, applicable rules out (strictly from PDF context).
"""
import os
import re
from operator import itemgetter

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
# System prompt: strict grounding, no hallucination, Roman Urdu + English
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a construction rules assistant. You answer ONLY using the provided PDF context.

RULES (CRITICAL):
- Base every answer STRICTLY on the "Relevant PDF context" below. Do not use any other knowledge.
- If the context does not contain a rule that applies to the user's scenario, you MUST respond with exactly:
  - If response_language == "English": "This scenario’s relevant rule is not present in this PDF."
  - Otherwise (Roman Urdu): "Is scenario ka relevant rule is PDF main mojood nahi."
- Do NOT guess, approximate, or invent any rule. If in doubt, say the rule is not found.
- Prefer precision over length. Be clear and actionable.
- Accept input in Roman Urdu, English, or mix. Understand the scenario (e.g. road-facing plot, residential/commercial, location) before answering.

LANGUAGE (MANDATORY):
- You MUST write the full response in {response_language}.
- Keep technical terms as-is when needed (e.g., \"setback\", \"signage\", \"right-of-way\").

OUTPUT FORMAT (follow exactly, headings in the same language):
- If response_language == "English":
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

- Otherwise (Roman Urdu):
  1. Scenario Samajhna
     - User ne kya banana / karna hai
     - Location context (road-facing, residential, commercial)

  2. Laagu Honay Wale Rules
     - Bullet points
     - Exact rule explanation from the context only
     - Only what applies to the scenario

  3. Workers ke liye Instructions
     - Simple actionable steps
     - On-site friendly language

  4. Source Reference
     - Page number(s) from PDF

Relevant PDF context:
{context}
"""

USER_PROMPT = """response_language: {response_language}

User scenario / question:
{question}
"""


def detect_response_language(text: str) -> str:
    """
    Heuristic language routing:
    - Roman Urdu and English both use Latin script, so we use a small keyword scoring.
    - Returns: "English" or "Roman Urdu"
    """
    t = re.sub(r"[^a-zA-Z\s]", " ", (text or "").lower())
    words = [w for w in t.split() if w]
    if not words:
        return "Roman Urdu"

    english_markers = {
        "what",
        "which",
        "when",
        "where",
        "why",
        "how",
        "setback",
        "signage",
        "regulation",
        "regulations",
        "applicable",
        "required",
        "permit",
        "residential",
        "commercial",
        "building",
        "constructing",
        "road",
        "plot",
        "main",
        "please",
        "explain",
    }
    roman_urdu_markers = {
        "kya",
        "ka",
        "ki",
        "ke",
        "mein",
        "main",
        "par",
        "hain",
        "hai",
        "bana",
        "banae",
        "banana",
        "raha",
        "rahi",
        "hoon",
        "hoga",
        "honay",
        "laagu",
        "workers",
    }

    eng_score = sum(1 for w in words if w in english_markers)
    ur_score = sum(1 for w in words if w in roman_urdu_markers)

    # If the question is clearly English, prefer English.
    if eng_score > ur_score + 1:
        return "English"
    return "Roman Urdu"


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
    # IMPORTANT: the retriever must receive only the question string, not the full input dict.
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "response_language": itemgetter("response_language"),
        }
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

    # Global styles (clean, professional, construction-inspired)
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1200px; }
          .crb-hero {
            border: 1px solid rgba(255,255,255,0.08);
            background: radial-gradient(1200px 500px at 10% 10%, rgba(255,180,0,0.20), rgba(0,0,0,0.0)),
                        linear-gradient(135deg, rgba(20,24,40,0.95), rgba(12,12,16,0.95));
            border-radius: 18px;
            padding: 1.25rem 1.25rem;
            margin-bottom: 1rem;
          }
          .crb-title { margin: 0; font-weight: 750; letter-spacing: -0.02em; }
          .crb-subtitle { margin: 0.35rem 0 0 0; color: rgba(255,255,255,0.72); }
          .crb-badge {
            display: inline-flex; gap: .5rem; align-items: center;
            padding: .25rem .6rem; border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            font-size: .85rem; color: rgba(255,255,255,0.80);
          }
          .crb-card {
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            border-radius: 14px;
            padding: 1rem 1rem;
          }
          .crb-muted { color: rgba(255,255,255,0.65); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Construction-themed header
    st.markdown(
        """
        <div class="crb-hero">
          <div class="crb-badge">🏗️ <span>RAG • PDF-grounded • Pinecone</span></div>
          <h1 class="crb-title">Construction Rules Assistant</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load vector store once
    try:
        vectorstore = get_vectorstore()
    except Exception as e:
        st.error(f"Vector store load nahi ho saka. Pehle `python ingest.py` chalaen. Error: {e}")
        st.stop()

    llm = get_llm()

    # Input area + tips
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown("<div class='crb-card'>", unsafe_allow_html=True)
        scenario = st.text_area(
            "🧱 Scenario / Question (Roman Urdu / English)",
            height=140,
            placeholder="Example (English): I am constructing a residential building on a main road-facing plot. What are the applicable setback and signage regulations?\nExample (Roman Urdu): Main road-facing plot par residential building bana raha hoon. Setback aur signage ke kya rules hain?",
            help="Write what you are building, where it is located, and what decision you need on-site.",
        )
        submit = st.button("📌 Get Applicable Rules", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='crb-card'>", unsafe_allow_html=True)
        st.markdown("#### 🦺 On-site Tips")
        st.markdown(
            """
            - Be specific: road type, area (residential/commercial), work type.
            - Ask one scenario at a time for best retrieval.
            - If rule is not in PDF, the bot will say so (no hallucination).
            """,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        if not (scenario and scenario.strip()):
            st.warning("Please write your scenario / sawal first.")
            return

        response_language = detect_response_language(scenario.strip())
        chain = build_chain(vectorstore, llm)

        with st.spinner("Searching rules from PDF context..."):
            try:
                answer = chain.invoke({"question": scenario.strip(), "response_language": response_language})
            except Exception as e:
                st.error(f"Error: {e}")
                return
        if response_language == "English":
            st.success("Done. Retrieved from PDF context.")
            title = "### 🧾 Answer"
        else:
            st.success("Rules mil gaye.")
            title = "### 🧾 Jawab"
        # Formatted sections
        st.markdown("---")
        st.markdown(title)
        st.markdown(answer)
        st.markdown("---")

    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.85rem;'>Answers are based only on uploaded PDF rules. Safety first.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
