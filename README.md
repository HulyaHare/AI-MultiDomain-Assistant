# AI Multi-Domain Recommendation Assistant

A production-style, AI-powered smart assistant that provides intelligent recommendations across **Healthcare**, **Movies**, and **E-Commerce** — all from a single chat interface.

Built with **FastAPI**, **scikit-learn**, and **LLM orchestration** (Groq / Llama 3.3 70B), the system combines classical ML pipelines with modern large language model capabilities.

---

## Features

- **Healthcare Engine** — Symptom-based disease prediction using TF-IDF + LinearSVC (40 diseases, 217 unique symptoms)
- **Movie Engine** — Content-based movie recommendations using TF-IDF + Cosine Similarity (4,800+ films from TMDB)
- **E-Commerce Engine** — Product search with price filtering using TF-IDF + Cosine Similarity (20,000+ Flipkart products)
- **LLM Orchestration** — Two-stage pipeline: intent detection + natural language response generation via Groq API
- **Session Memory** — Context-aware follow-up conversations with per-session history
- **Mobile Chat UI** — Responsive chat interface with domain-specific card rendering (Tailwind CSS, no build step)

---

## Architecture

```
User Message
    │
    ▼
┌─────────────────────┐
│   LLM: Intent &     │  ← Groq / Llama 3.3 70B
│   Entity Extraction  │
└────────┬────────────┘
         │
    ┌────┴────┬──────────┐
    ▼         ▼          ▼
┌────────┐ ┌────────┐ ┌──────────┐
│ Health │ │ Movie  │ │E-Commerce│  ← Local ML (scikit-learn)
│ Engine │ │ Engine │ │  Engine  │
└───┬────┘ └───┬────┘ └────┬─────┘
    └──────┬───┴───────────┘
           ▼
┌─────────────────────┐
│  LLM: Natural Lang  │  ← Groq / Llama 3.3 70B
│  Response Generation │
└────────┬────────────┘
         ▼
    Chat Response
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| ML Models | scikit-learn (TF-IDF, LinearSVC, Cosine Similarity) |
| Data | pandas, numpy |
| LLM | LangChain, Groq API (Llama 3.3 70B Versatile) |
| Frontend | HTML, JavaScript, Tailwind CSS (CDN) |
| Validation | Pydantic, pydantic-settings |

---

## Project Structure

```
├── main.py                     # FastAPI entry point
├── app/
│   ├── config.py               # Environment settings (Groq API key)
│   ├── schemas.py              # Pydantic request/response models
│   ├── lifespan.py             # Startup: load CSVs & fit ML models once
│   ├── data/
│   │   └── loader.py           # DataStore + CSV preprocessing + model fitting
│   ├── engines/
│   │   ├── healthcare.py       # TF-IDF + SVM disease prediction
│   │   ├── movie.py            # Cosine similarity movie recommendations
│   │   └── ecommerce.py        # Cosine similarity product search + price filter
│   ├── services/
│   │   ├── llm.py              # Groq LLM: intent detection + response generation
│   │   └── memory.py           # Session-based conversation memory
│   └── routers/
│       └── assistant.py        # POST /api/query endpoint
├── static/
│   └── index.html              # Mobile-style chat UI
├── Healthcare.csv              # 14,086 records, 40 diseases
├── tmdb_5000_movies.csv        # 4,803 movies (TMDB)
├── tmdb_5000_credits.csv       # Cast & crew data
├── flipkart_com-ecommerce_sample.csv  # 20,000+ products
├── requirements.txt
└── .env                        # GROQ_API_KEY (not committed)
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/HulyaHare/AI-MultiDomain-Assistant.git
cd AI-MultiDomain-Assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com)

### 4. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Open the chat UI

Navigate to **http://localhost:8000** in your browser.

---

## Usage Examples

| Domain | Input | Expected Output |
|--------|-------|----------------|
| Healthcare | "I have a runny nose, sneezing and sore throat" | Common Cold |
| Healthcare | "severe chest pain radiating to left arm" | Heart Attack |
| Healthcare | "wheezing and difficulty breathing" | Asthma |
| Movie | "Recommend movies similar to Inception" | Similar sci-fi/thriller films |
| Movie | "I want a romantic comedy" | Romantic comedy recommendations |
| E-Commerce | "headphones under 2000 rupees" | Budget headphone options (INR) |
| E-Commerce | "Samsung mobile phones" | Samsung phone listings |

---

## Key Design Decisions

- **Lifespan Pattern**: All CSVs are loaded and ML models are fitted once at startup — zero disk I/O per request
- **NaN-Safe Pipeline**: `fillna("")` on all text columns, `pd.to_numeric(errors="coerce")` on numerics, `_sanitize()` on JSON output
- **Two-Stage LLM**: Intent extraction (structured JSON) and response generation (natural language) are separate LLM calls for reliability
- **No External APIs for Data**: Movie and product data served from local CSVs — no rate limits, no costs, no internet dependency
- **ngram_range=(1,2)**: Captures multi-word symptoms like "chest pain" and "shortness of breath" as single features

---

## API Reference

### POST `/api/query`

**Request:**
```json
{
  "message": "I have a headache and fever",
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "message": "Based on your symptoms, this could be related to...",
  "domain": "healthcare",
  "data": {
    "predicted_disease": "Common Cold",
    "confidence": 2.341,
    "common_symptoms": ["runny nose", "sneezing", "sore throat"],
    "total_cases_in_data": 381
  },
  "session_id": "uuid-v4"
}
```

---

## Datasets

| File | Records | Source |
|------|---------|--------|
| Healthcare.csv | 14,086 | Custom generated — 40 diseases with medically accurate symptom mappings |
| tmdb_5000_movies.csv | 4,803 | [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |
| tmdb_5000_credits.csv | 4,803 | TMDB (cast & crew data) |
| flipkart_com-ecommerce_sample.csv | 20,000+ | [Flipkart Products](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products) |

---

## License

This project is for educational and portfolio purposes.
