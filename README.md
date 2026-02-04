# Construction Rules Assistant

A production-ready **Retrieval Augmented Generation (RAG)** chatbot that answers construction rules and regulations strictly from your uploaded PDF documents. Built for workers, supervisors, and engineers who need accurate, source-grounded answers without guesswork.

---

## Features

- **PDF-grounded answers** — Responses are based only on retrieved PDF content; no hallucination or external knowledge.
- **Bilingual support** — Accepts queries in **English** or **Roman Urdu**; responses follow the same language.
- **Structured output** — Scenario understanding, applicable rules, worker instructions, and PDF page references.
- **Streamlit UI** — Clean, professional interface with construction-themed layout and on-site tips.
- **Pinecone vector store** — Scalable semantic search over large rulebooks (e.g. MUTCD).

---

## Tech Stack

| Component        | Technology                    |
|-----------------|-------------------------------|
| Language        | Python 3.10+                  |
| Framework       | LangChain                     |
| LLM / Embeddings| OpenAI (configurable models)  |
| Vector DB       | Pinecone                      |
| UI              | Streamlit                     |
| Config          | python-dotenv                 |

---

## Prerequisites

- Python 3.10 or higher  
- OpenAI API key  
- Pinecone API key and index name  
- PDF rulebook(s) in the `data/` folder  

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/mustafaboss/Jumpace-project.git
   cd Jumpace-project
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   # source venv/bin/activate     # macOS / Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root (never commit this file):

   ```env
   OPENAI_API_KEY=your_openai_api_key
   CHAT_MODEL=gpt-4o-mini
   EMBED_MODEL=text-embedding-3-small
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX=rag-index
   ```

   Obtain keys from [OpenAI](https://platform.openai.com/api-keys) and [Pinecone](https://app.pinecone.io/).

5. **Add your PDF(s)**

   Place one or more rulebook PDFs in the `data/` folder:

   ```
   data/
     your_rules.pdf
   ```

---

## Usage

### 1. Ingest PDFs (run once)

Index all PDFs in `data/` into Pinecone. For large documents this may take several minutes.

```bash
python ingest.py
```

- Creates the Pinecone index if it does not exist.  
- Chunks text (≈800–1200 tokens per chunk, 150–250 token overlap).  
- Stores embeddings and metadata (page number, source file).

### 2. Run the web app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (e.g. `http://localhost:8501`). Enter your scenario or question in the text area and click **Get Applicable Rules**.

---

## Project Structure

```
.
├── app.py              # Streamlit RAG application
├── ingest.py            # PDF ingestion and Pinecone indexing
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create locally, do not commit)
├── .gitignore
├── README.md
└── data/               # Place PDF rulebooks here
    └── *.pdf
```

---

## Response Format

Answers are structured as:

1. **Understanding the Scenario** — What is being built and where (e.g. road-facing, residential/commercial).
2. **Applicable Rules** — Bullet-point rules drawn only from the PDF context.
3. **Instructions for Workers** — Simple, on-site friendly steps.
4. **Source Reference** — Page number(s) from the PDF.

If no relevant rule exists in the PDF, the assistant states that clearly instead of guessing.

---

## Safety and Correctness

- Answers are **strictly grounded** in the provided PDFs.  
- The system is designed for real construction use; correctness and safety take precedence over creativity.  
- Always verify critical rules against the original document when in doubt.

---

## License

Use and modify as needed for your organization. Ensure API keys and `.env` are never committed or shared.
