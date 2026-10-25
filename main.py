import asyncio
import os
import re
import uuid
import textwrap
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field

import pymupdf as fitz
import numpy as np
from groq import Groq
from langdetect import detect_langs

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- Common Stopwords for TF-IDF Search Optimization ---
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    "paper", "this", "these"
}

# --- Load Environment Variables ---
def load_env_vars():
    """Safely load .env file from script directory or current working directory."""
    for base in [Path(__file__).parent, Path.cwd()]:
        env_path = base / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v = parts[1].strip().strip("\"'")
                        if k and k not in os.environ:
                            os.environ[k] = v

load_env_vars()

# --- Initialize FastAPI App ---
app = FastAPI(title="Research Paper Assistant Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global In-Memory Stores with LRU Eviction ---
MAX_PAPERS = 20
ACTIVE_PAPERS: OrderedDict[str, Dict[str, Any]] = OrderedDict()

LANGUAGE_NAMES = {
    "en": "English", "ml": "Malayalam", "hi": "Hindi",
    "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "fr": "French", "de": "German", "es": "Spanish",
    "zh-cn": "Chinese", "ar": "Arabic", "ja": "Japanese",
    "ru": "Russian", "pt": "Portuguese", "it": "Italian",
    "ko": "Korean", "nl": "Dutch", "tr": "Turkish"
}

# --- PDF Processing & Vector Database ---
def extract_pdf_text(path: Path) -> List[Dict[str, Any]]:
    """Extract and clean text from every page of the PDF using PyMuPDF."""
    pages = []
    try:
        with fitz.open(str(path)) as doc:
            if doc.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="The PDF is password protected. Please upload an unprotected PDF."
                )
            for i, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    # Clean up hyphenated line breaks (e.g. "meth-\nodology" -> "methodology")
                    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                    # Normalize spaces
                    text = re.sub(r"[ \t]+", " ", text)
                    # Normalize newlines
                    text = re.sub(r"\n{3,}", "\n\n", text).strip()
                    pages.append({"page": i + 1, "text": text})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read PDF file: {str(e)}"
        )
    return pages

def chunk_pages(pages: List[Dict[str, Any]], size: int = 350, step: int = 150) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Sliding-window word chunker. Preserves short pages and avoids trailing slivers."""
    chunks, meta = [], []
    for page_info in pages:
        words = page_info["text"].split()
        page_num = page_info["page"]
        if not words:
            continue

        # If page word count is within chunk size, keep as single chunk
        if len(words) <= size:
            chunk = " ".join(words)
            if chunk.strip():
                chunks.append(chunk)
                meta.append({"page": page_num, "offset": 0})
            continue

        # Sliding window for long pages
        for i in range(0, len(words), step):
            chunk_words = words[i : i + size]
            # Avoid redundant trailing snippet if already mostly covered
            if len(chunk_words) < 30 and i > 0:
                continue
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            meta.append({"page": page_num, "offset": i})
            if i + size >= len(words):
                break
    return chunks, meta

class TfidfStore:
    """A compact NumPy TF-IDF index with stopword weighting and cosine similarity."""
    def __init__(self, chunks: List[str], meta: List[Dict[str, Any]]):
        self.chunks = chunks or []
        self.meta = meta or []
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        if not self.chunks:
            self.vocabulary: Dict[str, int] = {}
            self.idf = np.array([], dtype=np.float32)
            self.index = np.zeros((0, 0), dtype=np.float32)
            return

        tokenized = [self._tokens(chunk) for chunk in self.chunks]
        document_frequency = Counter(
            token for tokens in tokenized for token in set(tokens)
        )
        vocabulary = [
            token for token, count in document_frequency.most_common(80000)
            if count >= 1
        ]
        self.vocabulary = {token: i for i, token in enumerate(vocabulary)}
        self.idf = np.log(
            (1 + len(self.chunks))
            / (1 + np.array(
                [document_frequency[token] for token in vocabulary],
                dtype=np.float32,
            ))
        ) + 1.0
        self.index = self._embed_documents(tokenized)

    def _tokens(self, text: str) -> List[str]:
        return self.token_pattern.findall(text.lower())

    def _embed_documents(self, documents: List[List[str]]) -> np.ndarray:
        if not self.vocabulary or not documents:
            return np.zeros((len(documents), len(self.vocabulary)), dtype=np.float32)
        vectors = np.zeros((len(documents), len(self.vocabulary)), dtype=np.float32)
        for row, tokens in enumerate(documents):
            counts = Counter(tokens)
            for token, count in counts.items():
                column = self.vocabulary.get(token)
                if column is not None:
                    # Give lower weight to common stopwords
                    weight = 0.2 if token in STOPWORDS else 1.0
                    vectors[row, column] = (1.0 + np.log(count)) * self.idf[column] * weight
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def _embed(self, text: str) -> np.ndarray:
        tokens = self._tokens(text)
        # If query contains non-stopwords, focus on them for higher precision
        non_stops = [t for t in tokens if t not in STOPWORDS]
        active_tokens = non_stops if non_stops else tokens
        return self._embed_documents([active_tokens])[0]

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks or not self.vocabulary:
            return []
        embed = self._embed(query)
        if np.all(embed == 0):
            return []
        scores = self.index @ embed
        k = min(k, len(self.chunks))
        idxs = np.argsort(scores)[::-1][:k]
        return [
            {
                "text": self.chunks[i],
                "score": float(scores[i]),
                "page": self.meta[i]["page"],
            }
            for i in idxs
            if scores[i] > 0.001
        ]

# --- Groq LLM Wrapper ---
class GroqLLM:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key must be provided.")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def chat(self, system: str, user: str, max_tokens: int = 700) -> str:
        return self.chat_messages(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )

    def chat_messages(self, messages: List[Dict[str, str]], max_tokens: int = 700) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()

# --- QA Pipeline State & Prompts ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatState(BaseModel):
    query: str = Field(...)
    paper_id: str = Field(...)
    api_key: str = Field(...)
    lang_code: str = "en"
    lang_name: str = "English"
    action: Optional[str] = None
    search_query: Optional[str] = None
    docs: Optional[List[Dict[str, Any]]] = None
    answer: Optional[str] = None
    history: Optional[List[ChatMessage]] = None

PLANNER_SYS = textwrap.dedent("""
    You are a routing assistant for a Research Paper QA system.
    The user may write in any language.
    Decide whether the question needs searching or referencing the uploaded research paper
    (e.g., findings, methodology, overview, definitions, authors, data, results) -> action = "retrieve"
    or is purely general conversational banter / off-topic greeting -> action = "answer_direct".

    Rules:
    - Questions about the paper, authors, summary, methods, data, topic, or equations -> retrieve
    - Pure greetings (hello, hi), off-topic small talk, or jokes -> answer_direct

    Reply with ONLY one word: retrieve   OR   answer_direct
""").strip()

ANSWER_SYS = textwrap.dedent("""
    You are an expert research assistant. You answer questions about the uploaded
    research paper based on the provided context excerpts and paper overview.

    CRITICAL RULE: You MUST reply in {lang_name} ({lang_code}).
    Match the user's language exactly. If the query is in Spanish, reply in Spanish; if in Malayalam, reply in Malayalam; if in German, reply in German, etc.

    Guidelines:
    - Format your response cleanly using Markdown (use bolding, bullet points, or numbered lists where appropriate).
    - Quote page numbers when referencing specific facts or findings (e.g., "On page 4, the authors...").
    - If the provided excerpts do not contain enough information to answer a specific factual claim, clearly state that in {lang_name}.
    - Do NOT hallucinate statistics or claims not supported by the paper.
""").strip()

FALLBACK_SYS = textwrap.dedent("""
    You are a helpful research assistant.
    Reply in {lang_name} ({lang_code}).
    Answer the user politely and redirect them to explore the research paper.
""").strip()

def detect_language(text: str) -> Tuple[str, str]:
    """Accurately detect language with safety fallbacks for short conversational queries."""
    cleaned = text.strip()
    if not cleaned:
        return "en", "English"

    # Common English words/greetings
    english_words = {
        "hi", "hello", "hey", "what", "is", "the", "paper", "about",
        "who", "are", "authors", "summary", "summarize", "how", "why",
        "does", "can", "you", "explain", "findings", "results", "method",
        "methods", "model", "dataset", "conclusion", "ok", "thanks"
    }
    tokens = re.findall(r"\b\w+\b", cleaned.lower())
    if tokens and (len(tokens) <= 3 and any(t in english_words for t in tokens)):
        return "en", "English"

    if len(cleaned) < 15 and all(ord(c) < 128 for c in cleaned):
        return "en", "English"

    try:
        langs = detect_langs(cleaned)
        if langs and langs[0].prob >= 0.7:
            code = langs[0].lang
            name = LANGUAGE_NAMES.get(code, code.upper())
            return code, name
    except Exception:
        pass

    return "en", "English"

def node_detect_language(state: ChatState) -> ChatState:
    code, name = detect_language(state.query)
    state.lang_code = code
    state.lang_name = name
    return state

def node_planner(state: ChatState) -> ChatState:
    llm = GroqLLM(api_key=state.api_key)
    routing_prompt = (
        f"User question (in {state.lang_name}): {state.query}\n\n"
        "Route this: reply ONLY with 'retrieve' or 'answer_direct'."
    )
    decision = llm.chat(PLANNER_SYS, routing_prompt, max_tokens=8)
    state.action = "retrieve" if "retrieve" in decision.lower() else "answer_direct"
    return state

def node_retrieve(state: ChatState) -> ChatState:
    paper = ACTIVE_PAPERS.get(state.paper_id)
    if not paper:
        state.docs = []
        return state

    vector_store: TfidfStore = paper["vector_store"]
    search_query = state.query

    # If the user asked in a non-English language, translate user intent to English keywords
    # so TF-IDF can match against the English research paper chunks
    if state.lang_code != "en":
        try:
            llm = GroqLLM(api_key=state.api_key)
            trans_prompt = (
                f"Extract 3 to 6 English search keywords for an academic paper search based on this question in {state.lang_name}:\n"
                f"'{state.query}'\n"
                "Reply with ONLY the English keywords separated by spaces."
            )
            english_keywords = llm.chat("You are an academic translation assistant.", trans_prompt, max_tokens=30)
            if english_keywords.strip():
                search_query = english_keywords.strip()
        except Exception:
            search_query = state.query

    state.search_query = search_query
    results = vector_store.search(search_query, k=5)

    # Check if query is broad / meta (authors, summary, paper overview) or if search yielded no hits
    query_lower = state.query.lower()
    is_meta_query = any(k in query_lower for k in [
        "author", "who wrote", "summary", "summarize", "about", "overview",
        "title", "topic", "purpose", "aim", "goal"
    ])

    if not results or is_meta_query:
        overview_parts = []
        if paper.get("title"):
            overview_parts.append(f"Title: {paper['title']}")
        if paper.get("summary"):
            overview_parts.append(f"Summary:\n{paper['summary']}")
        if paper.get("contributions"):
            overview_parts.append(f"Key Contributions:\n{paper['contributions']}")
        if paper.get("first_page_text"):
            overview_parts.append(f"First Page (Authors & Abstract):\n{paper['first_page_text'][:1500]}")

        if overview_parts:
            overview_chunk = {
                "text": "\n\n".join(overview_parts),
                "score": 1.0,
                "page": 1,
                "is_overview": True
            }
            if not results:
                results = [overview_chunk]
            else:
                results.insert(0, overview_chunk)

    state.docs = results
    return state

def node_answer(state: ChatState) -> ChatState:
    llm = GroqLLM(api_key=state.api_key)
    has_docs = bool(state.docs)

    sys_prompt = ANSWER_SYS.format(
        lang_name=state.lang_name, lang_code=state.lang_code
    ) if has_docs else FALLBACK_SYS.format(
        lang_name=state.lang_name, lang_code=state.lang_code
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]

    # Include recent multi-turn conversation history
    if state.history:
        for msg in state.history[-6:]:
            if msg.role in ("user", "assistant") and msg.content.strip():
                messages.append({"role": msg.role, "content": msg.content})

    if has_docs:
        passages = "\n\n".join(
            f"[Page {d['page']}{' | Paper Overview' if d.get('is_overview') else f' | relevance {d['score']:.2f}'}]\n{d['text']}"
            for d in state.docs
        )
        user_msg = (
            f"Context from the uploaded research paper:\n{passages}\n\n"
            f"Question ({state.lang_name}): {state.query}"
        )
    else:
        user_msg = state.query

    messages.append({"role": "user", "content": user_msg})
    state.answer = llm.chat_messages(messages, max_tokens=700)
    return state

# --- Helpers to Get API Key ---
def get_groq_api_key(key_input: Optional[str] = None) -> str:
    key = (key_input or "").strip() or os.environ.get("GROQ_API_KEY", "").strip()
    if not key or key == "YOUR_GROQ_API_KEY_HERE":
        raise HTTPException(
            status_code=400,
            detail="Groq API key not set. Set it in .env or supply it in the request."
        )
    return key

# --- API Endpoints ---
@app.post("/upload")
async def upload_paper(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None)
):
    temp_path: Optional[Path] = None
    try:
        # Validate filename
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

        # Resolve API key
        groq_key = get_groq_api_key(api_key)

        # Save temp file
        paper_id = str(uuid.uuid4())
        upload_dir = Path(__file__).parent / "temp_uploads"
        upload_dir.mkdir(exist_ok=True)
        temp_path = upload_dir / f"{paper_id}.pdf"

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # Extract text & chunk
        pages = extract_pdf_text(temp_path)
        if not pages:
            raise HTTPException(
                status_code=400,
                detail="The PDF contains no readable text. It may be scanned or image-only."
            )

        chunks, meta = chunk_pages(pages)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="The PDF text could not be extracted into chunks."
            )

        # Build vector store
        vector_store = TfidfStore(chunks, meta)
        llm = GroqLLM(api_key=groq_key)

        # Gather excerpts for initial analysis
        first_page = pages[0]["text"] if pages else ""

        # 1. Summary context
        sum_docs = vector_store.search("abstract introduction overview summary conclusion", k=6)
        sum_context = "\n\n".join(d["text"] for d in sum_docs)
        if first_page and first_page not in sum_context:
            sum_context = f"{first_page[:1000]}\n\n{sum_context}"

        summary_prompt = (
            "Based on the following excerpts, write a clear, structured summary of this paper "
            "including its main goals, methodology, and key findings:\n\n"
            f"{sum_context}"
        )

        # 2. Contributions context
        contrib_docs = vector_store.search("contributions results findings novelty value", k=5)
        contrib_context = "\n\n".join(d["text"] for d in contrib_docs)
        contrib_prompt = (
            "Based on the following excerpts, list the main scientific and technical contributions "
            "of this paper. Present them as bullet points with bold titles:\n\n"
            f"{contrib_context}"
        )

        # 3. Technical Terms context
        terms_docs = vector_store.search("methodology technical equations definition framework acronyms", k=5)
        terms_context = "\n\n".join(d["text"] for d in terms_docs)
        terms_prompt = (
            "Based on the following excerpts, identify 5-8 advanced technical terms, algorithms, "
            "frameworks, or acronyms used in the paper and explain them briefly in bullet points:\n\n"
            f"{terms_context}"
        )

        # Execute all 3 initial analysis prompts concurrently
        loop = asyncio.get_running_loop()
        summary_task = loop.run_in_executor(
            None,
            lambda: llm.chat("You are an expert research assistant writing a paper summary.", summary_prompt, 600)
        )
        contrib_task = loop.run_in_executor(
            None,
            lambda: llm.chat("You are an expert research assistant listing paper contributions.", contrib_prompt, 400)
        )
        terms_task = loop.run_in_executor(
            None,
            lambda: llm.chat("You are an expert research assistant defining key technical terms.", terms_prompt, 500)
        )

        summary, contributions, technical_terms = await asyncio.gather(
            summary_task, contrib_task, terms_task
        )

        # Enforce LRU cache limit on ACTIVE_PAPERS
        if len(ACTIVE_PAPERS) >= MAX_PAPERS:
            ACTIVE_PAPERS.popitem(last=False)

        # Store in active papers
        ACTIVE_PAPERS[paper_id] = {
            "title": file.filename,
            "page_count": len(pages),
            "vector_store": vector_store,
            "first_page_text": first_page,
            "summary": summary,
            "contributions": contributions,
            "technical_terms": technical_terms,
        }

        return {
            "paper_id": paper_id,
            "title": file.filename,
            "page_count": len(pages),
            "summary": summary,
            "contributions": contributions,
            "technical_terms": technical_terms
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary upload file to prevent disk leaks
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

class QueryRequest(BaseModel):
    query: str
    paper_id: str
    api_key: Optional[str] = None
    history: Optional[List[ChatMessage]] = None

@app.post("/query")
async def query_paper(req: QueryRequest):
    try:
        groq_key = get_groq_api_key(req.api_key)

        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        if req.paper_id not in ACTIVE_PAPERS:
            raise HTTPException(
                status_code=404,
                detail="Paper session expired or not found. Please re-upload the paper."
            )

        # Run the QA pipeline
        state = ChatState(
            query=req.query,
            paper_id=req.paper_id,
            api_key=groq_key,
            history=req.history,
        )
        state = node_detect_language(state)
        state = node_planner(state)
        if state.action == "retrieve":
            state = node_retrieve(state)
        state = node_answer(state)

        pages = sorted(list(set(
            d["page"] for d in (state.docs or []) if not d.get("is_overview")
        )))

        has_overview = any(d.get("is_overview") for d in (state.docs or []))

        return {
            "answer": state.answer,
            "lang_code": state.lang_code,
            "lang_name": state.lang_name,
            "action": state.action,
            "pages": pages,
            "has_overview": has_overview,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "active_papers": len(ACTIVE_PAPERS)}

# Serve the frontend from the webui directory if present
webui_dir = Path(__file__).parent / "webui"
if webui_dir.exists():
    app.mount(
        "/",
        StaticFiles(directory=webui_dir, html=True),
        name="webui",
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)