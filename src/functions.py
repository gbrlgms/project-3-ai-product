"""Project settings and the LLM-track functions.

The data functions live in the other modules of this folder: first_look, data_quality,
explore_texts and prepare_documents.

The rule from the brief: Python logic goes in `.py` files, and the notebooks
hold the narrative. The app imports the same functions, so the text it receives
goes through exactly the same cleaning as your training data.

To use this module from a notebook in `notebooks/`:

    import sys
    sys.path.append("..")
    from src.functions import *

and from the app in `app/`, the same two lines with `sys.path.append(".")` when
you run `streamlit run app/streamlit_app.py` from the repo root.

To use the data modules as well, also add `src` itself to the path (they import
each other by bare name), and import each one: `from src.first_look import *`.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.eval_checks import (Answer, check_constraints, list_coverage, lookup, norm, score_run, source_hit,
                             summarize)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

RANDOM_STATE = 42
LLM_MODEL = "gpt-6-luna"                                    # LLM track: one place to change the model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
USD_PER_1M_TOKENS = {"input": 0.10, "output": 0.50}         # gpt-6-luna, standard tier (Nov 2026): update with the model
BASELINE_RUN = ROOT / "data" / "processed" / "baseline_llm_alone.json"


# Baseline: the LLM alone, no retrieval and no tools

BASELINE_INSTRUCTIONS = """You are a film guide for the Criterion Collection. Answer from what you know.

Fill in these fields:
- answer: a short answer in plain text. Write any number the question asks for in digits.
- picks: the film titles you recommend or list, as the Criterion Collection titles them, with no year or
  other notes. Give three for a recommendation unless the question asks for another number. Give every
  film that qualifies for a "which films" question. Leave it empty for a question about one fact.
- refused: true only if you cannot answer. The answer must then start with "I don't know".

Recommend only films that are in the Criterion Collection. If the question cannot be answered reliably,
for example because it needs live, undocumented or future information, say so instead of guessing."""

BASELINE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer", "picks", "refused"],
    "properties": {"answer": {"type": "string"}, "picks": {"type": "array", "items": {"type": "string"}},
                   "refused": {"type": "boolean"}},
}


def token_cost(input_tokens: int, output_tokens: int) -> float:
    """Cost in USD. Output tokens already include any reasoning tokens."""
    return (input_tokens * USD_PER_1M_TOKENS["input"] + output_tokens * USD_PER_1M_TOKENS["output"]) / 1_000_000


def ask_llm_alone(question: str, client: OpenAI) -> dict:
    """One question to the model, with no retrieval: its parsed answer plus tokens, cost and time."""
    start = time.time()
    response = client.responses.create(
        model=LLM_MODEL, instructions=BASELINE_INSTRUCTIONS, input=question,
        text={"format": {"type": "json_schema", "name": "film_answer", "strict": True, "schema": BASELINE_SCHEMA}},
    )
    usage = response.usage
    return {**json.loads(response.output_text), "model": response.model, "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens, "cost_usd": token_cost(usage.input_tokens, usage.output_tokens),
            "seconds": round(time.time() - start, 2)}


def run_baseline(eval_set: pd.DataFrame, path: Path = BASELINE_RUN, refresh: bool = False) -> tuple[dict, pd.DataFrame]:
    """({question id: Answer}, one row per question with tokens and cost). The run is saved to `path`
    and reused, so rerunning the notebook costs nothing; refresh=True calls the model again."""
    if path.exists() and not refresh:
        log = json.loads(path.read_text(encoding="utf-8"))
    else:
        load_dotenv(ROOT / ".env")
        client = OpenAI()
        log = [{"id": int(q["id"]), **ask_llm_alone(q["question"], client)} for _, q in eval_set.iterrows()]
        path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    answers = {row["id"]: Answer(text=row["answer"], picks=row["picks"], refused=row["refused"],
                                 cost_usd=row["cost_usd"]) for row in log}     # no retrieval, so no sources
    return answers, pd.DataFrame(log)


def pick_precision(question: pd.Series, answer: Answer, cat: pd.DataFrame) -> float | None:
    """Automatic part of precision@k for the pick questions: the share of the k picks that are in the
    catalogue and meet the hard constraints. A missing pick counts as wrong. None for other questions."""
    if question["auto_check"] not in ("constraints", "seed_excluded", "decades_differ"):
        return None
    params = json.loads(question["auto_params"])
    k = params.pop("k", 3)

    def valid(title: str) -> bool:
        if question["auto_check"] == "constraints":
            return check_constraints(Answer(picks=[title]), cat, k=1, **params)[0]
        if question["auto_check"] == "seed_excluded" and norm(title) == norm(params["seed"]):
            return False
        return lookup(cat, title) is not None

    return sum(valid(title) for title in answer.picks[:k]) / k


# LLM track: retrieval

CHROMA_DIR = MODELS / "chroma"
COLLECTION_NAME = "criterion_chunks"
GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy",
          "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"]
_METADATA_FIELDS = ("film_id", "title", "year", "decade", "director", "film_spine", "country", "genres", "runtime",
                    "section", "section_type", "spoilers", "source", "source_title", "revid", "url", "n_words")
RUNS_DIR = ROOT / "data" / "processed" / "pipeline_runs"
FETCH_FACTOR = 5                                            # candidates per final chunk when hybrid search reorders them
RRF_CONSTANT = 60                                           # standard damping value for reciprocal rank fusion
_QUERY_STOP_WORDS = ENGLISH_STOP_WORDS | {"film", "films", "movie", "movies", "collection", "criterion"}   # would outweigh a rare term
_embedder = None
_openai_client = None
_keyword_index = None


def _openai() -> OpenAI:
    """Shared client. The retries ride out rate limits."""
    global _openai_client
    if _openai_client is None:
        load_dotenv(ROOT / ".env")
        _openai_client = OpenAI(max_retries=8)
    return _openai_client


def _embed(texts: list[str]) -> list[list[float]]:
    """Normalized embeddings from the local model (loaded on first use, so importing stays fast)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=len(texts) > 1000).tolist()


def _flag(kind: str, value: str) -> str:
    """Metadata key for one genre, country or director, for example 'genre_science_fiction'."""
    return f"{kind}_{norm(value).replace(' ', '_')}"


def _chunk_metadata(chunk: dict) -> dict:
    """What Chroma stores for a chunk. Chroma has no empty values, and it cannot search inside a
    joined string such as 'Drama|Thriller', so each genre, country and director gets its own flag."""
    meta = {field: chunk[field] for field in _METADATA_FIELDS if chunk.get(field) is not None}
    for kind, field, separator in (("genre", "genres", "|"), ("country", "country", ", "), ("director", "director", ", ")):
        for value in (chunk.get(field) or "").split(separator):
            if value:
                meta[_flag(kind, value)] = True
    return meta


def build_index(documents: list[dict], path: Path = CHROMA_DIR, rebuild: bool = False):
    """Embed the chunks and store them, with their metadata, in a Chroma collection.
    An index that already holds every chunk is reused unless rebuild=True."""
    import chromadb
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() == len(documents) and not rebuild:
        return collection
    client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)
    embeddings = _embed([d["embed_text"] for d in documents])
    for start in range(0, len(documents), 2000):          # Chroma limits the size of one insert
        batch = documents[start:start + 2000]
        collection.add(ids=[d["chunk_id"] for d in batch], embeddings=embeddings[start:start + 2000],
                       documents=[d["embed_text"] for d in batch], metadatas=[_chunk_metadata(d) for d in batch])
    return collection


def _where(filters: dict | None, extra: list[dict] | None = None) -> dict | None:
    """Chroma filter for years, runtime, genre, country and director, plus any `extra` clauses. Empty values are ignored."""
    filters = filters or {}
    clauses = list(extra or [])
    for key, field, operator in (("year_min", "year", "$gte"), ("year_max", "year", "$lte"),
                                 ("runtime_min", "runtime", "$gte"), ("runtime_max", "runtime", "$lte")):
        if filters.get(key) is not None:
            clauses.append({field: {operator: filters[key]}})
    clauses += [{_flag(kind, filters[kind]): True} for kind in ("genre", "country", "director") if filters.get(kind)]
    return None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _keyword_search(question: str, collection, n: int, where: dict | None) -> list[dict]:
    """The n best BM25 matches that meet the filter. The index is built on first use and kept in memory."""
    global _keyword_index
    if _keyword_index is None:
        from rank_bm25 import BM25Okapi
        rows = collection.get(include=["documents"])
        _keyword_index = (BM25Okapi([norm(document).split() for document in rows["documents"]]), rows["ids"])
    bm25, ids = _keyword_index
    allowed = set(collection.get(where=where, include=[])["ids"]) if where else None
    scores = bm25.get_scores([term for term in norm(question).split() if term not in _QUERY_STOP_WORDS])
    best = [ids[i] for i in np.argsort(-scores) if scores[i] > 0 and (allowed is None or ids[i] in allowed)][:n]
    if not best:
        return []
    rows = collection.get(ids=best, include=["documents", "metadatas"])
    found = {i: {"id": i, "text": text, "metadata": meta, "distance": None}
             for i, text, meta in zip(rows["ids"], rows["documents"], rows["metadatas"])}
    return [found[i] for i in best]


def _fuse_rankings(*rankings: list[dict]) -> list[dict]:
    """Merge ranked chunk lists by reciprocal rank fusion: a chunk near the top of either list rises."""
    scores: Counter = Counter()
    chunks: dict = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk["id"]] += 1 / (RRF_CONSTANT + rank)
            chunks.setdefault(chunk["id"], chunk)
    return [chunks[i] for i, _ in scores.most_common()]


def retrieve(question: str, collection, k: int = 5, filters: dict | None = None, hybrid: bool = False) -> list[dict]:
    """The k best chunks that meet the filters: id, text (with its header), metadata and distance.
    `hybrid` merges keyword search (BM25) with the embeddings."""
    where = _where(filters)
    n_results = k * FETCH_FACTOR if hybrid else k
    result = collection.query(query_embeddings=_embed([question]), n_results=n_results, where=where)
    ranked = [{"id": i, "text": text, "metadata": meta, "distance": distance} for i, text, meta, distance
              in zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])]
    if hybrid:
        ranked = _fuse_rankings(ranked, _keyword_search(question, collection, n_results, where))
    return ranked[:k]


def catalog_lookup(filters: dict, collection) -> dict | None:
    """Read-only catalog tool: answer a list or count question from the catalog rows instead of from retrieved text.
    Returns a reply with the same fields as generate_answer, or None when the question should go to retrieval."""
    start = time.time()
    if filters.get("task") not in ("list", "count"):
        return None
    filters, facts = dict(filters), {"section_type": "catalog_facts"}
    spine_row = None
    if filters.get("director_of_spine") is not None:
        found = collection.get(where=_where(None, [facts, {"film_spine": filters["director_of_spine"]}]),
                               include=["documents", "metadatas"])
        if not found["ids"]:
            return None
        spine_row = {"id": found["ids"][0], "text": found["documents"][0], "metadata": found["metadatas"][0], "distance": None}
        filters["director"] = spine_row["metadata"]["director"].split(", ")[0]
    if _where(filters) is None:                             # nothing structured to look up
        return None
    rows = collection.get(where=_where(filters, [facts]), include=["documents", "metadatas"])
    matches = sorted(({"id": i, "text": text, "metadata": meta, "distance": None}
                      for i, text, meta in zip(rows["ids"], rows["documents"], rows["metadatas"])),
                     key=lambda chunk: (chunk["metadata"].get("year", 0), chunk["metadata"]["title"]))
    titles = [chunk["metadata"]["title"] for chunk in matches]
    if not matches:
        answer, picks = "I don't know: no film in the Collection meets those conditions.", []
    elif filters["task"] == "count":
        answer, picks = f"The Collection has {len(matches)} films that meet those conditions.", []
    else:
        answer, picks = f"{len(matches)} films in the Collection qualify: {'; '.join(titles)}.", titles
    retrieved = [spine_row, *matches] if spine_row else matches
    return {"answer": answer, "picks": picks, "refused": not matches, "cited_ids": [chunk["id"] for chunk in retrieved],
            "titles_removed": [], "input_tokens": 0, "output_tokens": 0, "seconds": round(time.time() - start, 2),
            "retrieved": retrieved, "route": "catalog"}


# LLM track: generation

FILTER_INSTRUCTIONS = """Extract the hard constraints from a question about films in the Criterion Collection.
Use null for anything the question does not state explicitly.
- year_min, year_max: the release years. "the 1970s" is 1970 to 1979.
- genre: only when the question asks for a kind of film, as in "a comedy". A theme is not a genre: "films that deal with war" has no genre.
- country: the country that produced the film, in English, as a full name such as "Japan" or "United States of America".
- director: the director's full name, only when the question names the person.
- runtime_min, runtime_max: the film's length in minutes, only when the question states a limit. "under 100 minutes" is a runtime_max of 99.
- director_of_spine: the spine number, only when the question identifies the director through a film's spine number.
- task: "count" if the question asks how many films meet some conditions. "list" if it asks for every film that meets some conditions and
  the fields above express all of those conditions. Otherwise "answer": a theme, mood, plot or anything else these fields cannot express makes it "answer"."""

_nullable = lambda kind: {"type": [kind, "null"]}
FILTER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["year_min", "year_max", "genre", "country", "director", "runtime_min", "runtime_max", "director_of_spine", "task"],
    "properties": {"year_min": _nullable("integer"), "year_max": _nullable("integer"),
                   "genre": {"type": ["string", "null"], "enum": [*GENRES, None]},
                   "country": _nullable("string"), "director": _nullable("string"),
                   "runtime_min": _nullable("integer"), "runtime_max": _nullable("integer"),
                   "director_of_spine": _nullable("integer"), "task": {"type": "string", "enum": ["answer", "list", "count"]}},
}
_EXTENDED_FILTERS = ("runtime_min", "runtime_max", "director_of_spine", "task")

ANSWER_INSTRUCTIONS = """You are a film guide for the Criterion Collection. Answer only from the numbered sources in the message.
Do not use anything else you know, even if you are sure of it.

Fill in these fields:
- answer: a short answer in plain text. Write any number the question asks for in digits. Never reveal how a film ends.
- picks: the film titles you recommend, list or identify, exactly as the sources title them, with no year. Give three for a
  recommendation unless the question asks for another number. Give every film in the sources that qualifies for a "which films"
  question. Give the one film for a question that describes a film. Leave it empty for a question about a single fact.
- sources: the ids of the sources that support the answer, such as "S2".
- refused: true if the sources do not answer the question. The answer must then start with "I don't know"."""

FIRM_PICKS_RULE = """

For a recommendation, give the full number of picks whenever the sources hold enough films that fit, even if the fit is not perfect."""

ANSWER_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer", "picks", "sources", "refused"],
    "properties": {"answer": {"type": "string"}, "picks": {"type": "array", "items": {"type": "string"}},
                   "sources": {"type": "array", "items": {"type": "string"}}, "refused": {"type": "boolean"}},
}


def _ask_json(client: OpenAI, instructions: str, message: str, name: str, schema: dict) -> tuple[dict, int, int]:
    """(parsed JSON reply, input tokens, output tokens)."""
    response = client.responses.create(
        model=LLM_MODEL, instructions=instructions, input=message,
        text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
    )
    return json.loads(response.output_text), response.usage.input_tokens, response.usage.output_tokens


def extract_filters(question: str, client: OpenAI | None = None) -> tuple[dict, int, int]:
    """(the question's hard constraints, input tokens, output tokens)."""
    filters, input_tokens, output_tokens = _ask_json(client or _openai(), FILTER_INSTRUCTIONS, question, "filters", FILTER_SCHEMA)
    return {key: value for key, value in filters.items() if value is not None}, input_tokens, output_tokens


def catalog_titles(collection) -> set[str]:
    """The normalized title of every film in the Collection, from the index's catalog-facts chunks."""
    rows = collection.get(where={"section_type": "catalog_facts"}, include=["metadatas"])
    return {norm(metadata["title"]) for metadata in rows["metadatas"]}


def generate_answer(question: str, retrieved: list[dict], client: OpenAI | None = None, catalog: set[str] | None = None,
                    runtime_header: bool = False, firm_picks: bool = False) -> dict:
    """The grounded answer to a question from its retrieved chunks, with the cited ids, tokens and time.
    With `catalog`, a named film outside the Collection triggers one retry, and any that remains is removed.
    `runtime_header` shows each source's runtime to the model, and `firm_picks` adds FIRM_PICKS_RULE to the prompt."""
    start = time.time()
    if not retrieved:
        return {"answer": "I don't know: no film in the Collection meets those conditions.", "picks": [], "refused": True,
                "cited_ids": [], "titles_removed": [], "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
    client = client or _openai()
    labels = {f"S{n}": chunk for n, chunk in enumerate(retrieved, 1)}
    def header(label: str, chunk: dict) -> str:
        runtime = chunk["metadata"].get("runtime") if runtime_header else None
        return f"[{label}] Runtime: {runtime} minutes" if runtime else f"[{label}]"

    context = "\n\n".join(f"{header(label, chunk)}\n{chunk['text']}" for label, chunk in labels.items())
    instructions = ANSWER_INSTRUCTIONS + (FIRM_PICKS_RULE if firm_picks else "")
    message = f"Question: {question}\n\nSources:\n\n{context}"
    reply, input_tokens, output_tokens = _ask_json(client, instructions, message, "film_answer", ANSWER_SCHEMA)
    removed: list[str] = []
    if catalog:
        outside = [p for p in reply["picks"] if norm(p) not in catalog]
        if outside:
            reminder = f"\n\nThese films are not in the Collection, so do not name them anywhere: {', '.join(outside)}."
            reply, more_input, more_output = _ask_json(client, instructions, message + reminder, "film_answer", ANSWER_SCHEMA)
            input_tokens, output_tokens = input_tokens + more_input, output_tokens + more_output
            removed = list(dict.fromkeys([*outside, *(p for p in reply["picks"] if norm(p) not in catalog)]))
            reply["picks"] = [p for p in reply["picks"] if norm(p) in catalog]
            if any(norm(title) in norm(reply["answer"]) for title in removed):      # still named in the prose
                reply["answer"] = (f"From the Collection: {', '.join(reply['picks'])}." if reply["picks"]
                                   else "I don't know: the sources name no film in the Collection that fits.")
                reply["refused"] = not reply["picks"]
    return {"answer": reply["answer"], "picks": reply["picks"], "refused": reply["refused"],
            "cited_ids": [labels[s]["id"] for s in reply["sources"] if s in labels], "titles_removed": removed,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "seconds": round(time.time() - start, 2)}


def answer_question(question: str, collection, k: int = 5, client: OpenAI | None = None, catalog: set[str] | None = None,
                    hybrid: bool = False, runtime_header: bool = False, firm_picks: bool = False, tool: bool = False) -> dict:
    """Retrieve, prompt and answer; return the answer, the cited ids, the retrieved ids and the tokens.
    With `tool`, list and count questions on catalog fields are answered by catalog_lookup instead."""
    client = client or _openai()
    start = time.time()
    filters, input_tokens, output_tokens = extract_filters(question, client)
    reply = catalog_lookup(filters, collection) if tool else None
    retrieved = reply["retrieved"] if reply else retrieve(question, collection, k, filters, hybrid)
    reply = reply or generate_answer(question, retrieved, client, catalog, runtime_header, firm_picks)
    input_tokens, output_tokens = input_tokens + reply["input_tokens"], output_tokens + reply["output_tokens"]
    return {**reply, "question": question, "filters": filters, "retrieved_ids": [chunk["id"] for chunk in retrieved],
            "retrieved": retrieved, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cost_usd": token_cost(input_tokens, output_tokens), "seconds": round(time.time() - start, 2)}


# LLM track: comparing versions on the evaluation set

QUESTION_TYPES = {"recommendation": "Recommendations (1–6)", "similarity": "Recommendations (1–6)", "programming": "Recommendations (1–6)",
                  "catalog set": "Catalog sets and facts (7–11)", "catalog fact": "Catalog sets and facts (7–11)",
                  "film detail": "Film details (12–13)", "aggregation": "Counts and two-step lists (14–15)",
                  "multi-hop": "Counts and two-step lists (14–15)", "mood recommendation": "Mood recommendations (16–20)",
                  "plot identification": "Plot identification (21–23)", "plot keyword list": "Plot keyword lists (24–25)"}
UNANSWERABLE_TYPE = "Data cannot answer (26–30)"
CHART_COLORS = {"baseline": "#eb6834", "final": "#2a78d6", "surface": "#fcfcfb", "ink": "#0b0b0b", "ink_secondary": "#52514e",
                "grid": "#e1e0d9", "axis": "#c3c2b7"}

def freeze_filters(eval_set: pd.DataFrame, path: Path = RUNS_DIR / "filters.json") -> dict:
    """{question id: filters and tokens}, extracted once, so that every version retrieves under the same filters."""
    if path.exists():
        return {int(key): value for key, value in json.loads(path.read_text(encoding="utf-8")).items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(extract_filters, eval_set["question"]))
    frozen = {int(i): {"filters": f, "input_tokens": a, "output_tokens": b} for i, (f, a, b) in zip(eval_set["id"], replies)}
    path.write_text(json.dumps(frozen, ensure_ascii=False, indent=1), encoding="utf-8")
    return frozen


def extend_filters(frozen: dict, eval_set: pd.DataFrame, path: Path = RUNS_DIR / "filters_extended.json") -> dict:
    """Add the runtime limits, the spine lookup and the task to frozen filters. The filters already frozen
    are kept, so that the later versions differ from the earlier ones only in what they add."""
    if path.exists():
        return {int(key): value for key, value in json.loads(path.read_text(encoding="utf-8")).items()}
    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(extract_filters, eval_set["question"]))
    extended = {int(i): {"filters": {**frozen[int(i)]["filters"], **{key: f[key] for key in _EXTENDED_FILTERS if key in f}},
                         "input_tokens": a, "output_tokens": b} for i, (f, a, b) in zip(eval_set["id"], replies)}
    path.write_text(json.dumps(extended, ensure_ascii=False, indent=1), encoding="utf-8")
    return extended


def run_version(name: str, eval_set: pd.DataFrame, collection, k: int, frozen: dict, refresh: bool = False,
                catalog: set[str] | None = None, hybrid: bool = False, runtime_header: bool = False,
                firm_picks: bool = False, tool: bool = False) -> list[dict]:
    """Answer the whole evaluation set with one version. The run is saved under RUNS_DIR and reused,
    so rerunning the notebook costs nothing; refresh=True calls the models again.
    With `catalog` (see catalog_titles), films outside the Collection are kept out of the answers.
    The other options are those of retrieve, generate_answer and answer_question."""
    path = RUNS_DIR / f"{name}.json"
    if path.exists() and not refresh:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if set(eval_set["id"]) <= {r["id"] for r in saved}:     # a run of fewer questions cannot stand in for this one
            return saved
    prepared = []
    for _, q in eval_set.iterrows():                       # retrieval stays sequential: the local model is not thread-safe
        start, filters = time.time(), frozen[int(q["id"])]["filters"]
        lookup = catalog_lookup(filters, collection) if tool else None
        retrieved = lookup["retrieved"] if lookup else retrieve(q["question"], collection, k, filters, hybrid)
        prepared.append((int(q["id"]), q["question"], retrieved, lookup, time.time() - start))
    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda p: p[3] or generate_answer(p[1], p[2], catalog=catalog, runtime_header=runtime_header,
                                                                  firm_picks=firm_picks), prepared))
    results = []
    for (qid, question, retrieved, _, retrieve_seconds), reply in zip(prepared, replies):
        input_tokens = reply["input_tokens"] + frozen[qid]["input_tokens"]
        output_tokens = reply["output_tokens"] + frozen[qid]["output_tokens"]
        results.append({**reply, "id": qid, "question": question, "k": k, "embedding_model": EMBEDDING_MODEL,
                        "filters": frozen[qid]["filters"], "retrieved": [{"id": c["id"], "metadata": c["metadata"]} for c in retrieved],
                        "input_tokens": input_tokens, "output_tokens": output_tokens,
                        "cost_usd": token_cost(input_tokens, output_tokens),
                        "seconds": round(reply["seconds"] + retrieve_seconds, 2)})
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


def _source_labels(metadata: dict) -> list[str]:
    """Labels of a chunk in the form the evaluation set uses, at the level of the film or director."""
    return [f"{kind}:{metadata['title']}" for kind in "ATL"] + [f"L:{d}" for d in (metadata.get("director") or "").split(", ") if d]


def evaluate(results: list[dict], eval_set: pd.DataFrame, cat: pd.DataFrame, name: str) -> pd.DataFrame:
    """One row per question: the automatic checks, retrieval hit, list coverage, precision, cost and time."""
    by_id, rows, answers = {r["id"]: r for r in results}, {}, {}
    for _, q in eval_set.iterrows():
        r = by_id[q["id"]]
        metadata = {chunk["id"]: chunk["metadata"] for chunk in r["retrieved"]}
        answers[q["id"]] = Answer(text=r["answer"], picks=r["picks"], refused=r["refused"], cost_usd=r["cost_usd"],
                                  sources=[label for chunk_id in r["cited_ids"] for label in _source_labels(metadata[chunk_id])])
        labels = [label for m in metadata.values() for label in _source_labels(m)]
        rows[q["id"]] = {"retrieval_hit": source_hit(Answer(sources=labels), q["expected_sources"]) if q["answerable"] == 1 else None,
                         "list_coverage": list_coverage(q, [m["title"] for m in metadata.values()], cat),
                         "pick_precision": pick_precision(q, answers[q["id"]], cat),
                         "titles_named": len(r["picks"]), "titles_outside": sum(lookup(cat, p) is None for p in r["picks"]),
                         "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"], "seconds": r["seconds"]}
    return score_run(eval_set, answers, cat, name).merge(pd.DataFrame.from_dict(rows, orient="index"), left_on="id", right_index=True)


def load_pipeline(config_path: Path = MODELS / "config.json") -> tuple:
    """(index, catalog titles, settings for answer_question), from what notebook 02 saved. The app starts here."""
    import chromadb
    config = json.loads(config_path.read_text(encoding="utf-8"))
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)
    return collection, catalog_titles(collection), {key: config[key] for key in ("k", "runtime_header", "firm_picks", "hybrid", "tool")}


def baseline_as_run(log: pd.DataFrame) -> list[dict]:
    """The baseline's log in the shape `evaluate` expects. It has no retrieval, so nothing is retrieved or cited."""
    return [{**row, "cited_ids": [], "retrieved": []} for row in log.to_dict("records")]


def results_by_type(scored: dict, eval_set: pd.DataFrame) -> pd.DataFrame:
    """Questions passed per question type, one column for each version in `scored`, plus the number of questions."""
    types = eval_set.set_index("id")["group"].map(lambda group: QUESTION_TYPES.get(group, UNANSWERABLE_TYPE))
    table = pd.DataFrame({"questions": types.value_counts()})
    for name, df in scored.items():
        table[name] = df.set_index("id")["auto_pass"].astype(bool).groupby(types).sum()
    return table.loc[[*dict.fromkeys(QUESTION_TYPES.values()), UNANSWERABLE_TYPE]]


def _failure_kind(row: pd.Series, result: dict, cat: pd.DataFrame, collection) -> str:
    """'tool', 'retrieval' (the evidence never reached the model) or 'generation' (it did, and the answer still failed)."""
    if result.get("route") == "catalog":
        return "tool"
    if not result["retrieved"] or (pd.notna(row["retrieval_hit"]) and not row["retrieval_hit"]):
        return "retrieval"
    if row["auto_check"] == "contains_groups":
        read = norm(" ".join(collection.get(ids=[c["id"] for c in result["retrieved"]], include=["documents"])["documents"]))
        answer = norm(row["answer"])
        missing = [g for g in json.loads(row["auto_params"])["groups"] if not any(norm(term) in answer for term in g)]
        return "generation" if all(any(norm(term) in read for term in g) for g in missing) else "retrieval"
    if row["auto_check"] == "set_from_catalog" and row["list_coverage"] <= list_coverage(row, result["picks"], cat):
        return "retrieval"
    return "generation"


def classify_failures(scored: pd.DataFrame, results: list[dict], eval_set: pd.DataFrame, cat: pd.DataFrame, collection) -> pd.DataFrame:
    """One row per failed automatic check, with the kind of failure."""
    by_id = {r["id"]: r for r in results}
    merged = scored.merge(eval_set[["id", "question", "auto_check", "auto_params"]], on="id")
    failed = merged[~merged["auto_pass"].astype(bool)]
    kinds = [_failure_kind(row, by_id[row["id"]], cat, collection) for _, row in failed.iterrows()]
    return failed.assign(failure=kinds)[["id", "question", "answer", "picks", "auto_detail", "failure"]]


def _style_chart(ax, title: str) -> None:
    """Recessive grid and axes, so the marks carry the chart."""
    ax.set_facecolor(CHART_COLORS["surface"])
    ax.figure.set_facecolor(CHART_COLORS["surface"])
    ax.set_title(title, loc="left", color=CHART_COLORS["ink"], fontsize=12, pad=26)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(CHART_COLORS["axis"])
    ax.tick_params(colors=CHART_COLORS["ink_secondary"], length=0)


def plot_two_versions(labels: list[str], baseline: list[float], final: list[float], title: str, xlabel: str) -> None:
    """Paired horizontal bars of two versions on one share scale, with the final version labeled."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(labels) + 1.5))
    positions = np.arange(len(labels))[::-1]
    ax.barh(positions + 0.19, baseline, height=0.32, color=CHART_COLORS["baseline"], label="LLM alone (baseline)")
    bars = ax.barh(positions - 0.19, final, height=0.32, color=CHART_COLORS["final"], label="Final version")
    ax.bar_label(bars, fmt="%.0f%%", padding=4, color=CHART_COLORS["ink_secondary"], fontsize=9)
    ax.set_yticks(positions, labels, color=CHART_COLORS["ink"])
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xlabel(xlabel, color=CHART_COLORS["ink_secondary"])
    ax.grid(axis="x", color=CHART_COLORS["grid"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    _style_chart(ax, title)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, frameon=False, labelcolor=CHART_COLORS["ink_secondary"], fontsize=9, borderaxespad=0.1)
    fig.tight_layout()
    plt.show()


def plot_version_passes(labels: list[str], passes: list[int], baseline: int, total: int, title: str) -> None:
    """One bar per version in build order, with the baseline as a reference line."""
    import textwrap

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar([textwrap.fill(label, 10) for label in labels], passes, width=0.5, color=CHART_COLORS["final"])
    ax.bar_label(bars, labels=[str(passes[0]), *[""] * (len(passes) - 2), str(passes[-1])], padding=3, color=CHART_COLORS["ink_secondary"], fontsize=9)
    ax.axhline(baseline, color=CHART_COLORS["baseline"], linewidth=2)
    ax.annotate(f"LLM alone: {baseline}", (0, baseline), xytext=(-14, 5), textcoords="offset points", color=CHART_COLORS["ink_secondary"], fontsize=9)
    ax.set_ylim(0, total)
    ax.set_ylabel(f"Automatic checks passed, of {total}", color=CHART_COLORS["ink_secondary"])
    ax.grid(axis="y", color=CHART_COLORS["grid"], linewidth=0.8)
    ax.grid(axis="x", visible=False)
    _style_chart(ax, title)
    ax.tick_params(axis="x", colors=CHART_COLORS["ink"])
    fig.tight_layout()
    plt.show()


def tool_use(results: list[dict], eval_set: pd.DataFrame) -> pd.DataFrame:
    """Which route each question took against the route it should take: the catalog for lists and counts on catalog fields."""
    routes = {r["id"]: r.get("route", "retrieval") for r in results}
    expected = eval_set["auto_check"].isin(["set_from_catalog", "count_from_catalog"]).map({True: "catalog", False: "retrieval"})
    table = eval_set[["id", "question"]].assign(expected_route=expected, route=eval_set["id"].map(routes))
    return table.assign(correct=table["expected_route"] == table["route"])


def version_summary(scored: pd.DataFrame, eval_set: pd.DataFrame) -> pd.Series:
    """The headline numbers of one version, next to those of the baseline."""
    s = summarize(scored, eval_set)
    answerable = scored[scored["id"].isin(eval_set.loc[eval_set["answerable"] == 1, "id"])]
    return pd.Series({
        "retrieval_hit": answerable["retrieval_hit"].dropna().astype(bool).mean(),
        "list_coverage": scored["list_coverage"].dropna().mean(),
        "citation_hit": s["retrieval_hit_rate"],
        "auto_pass": s["auto_pass_rate"],
        "pick_precision": scored["pick_precision"].dropna().mean(),
        "titles_named": scored["titles_named"].sum(),
        "titles_outside": scored["titles_outside"].sum(),
        "false_refusal": s["false_refusal_rate"],
        "refused_unanswerable": s["refusal_rate_unanswerable"],
        "total_cost_usd": scored["cost_usd"].sum(),
        "mean_seconds": scored["seconds"].mean(),
        "input_tokens": scored["input_tokens"].sum(),
    })


# App: the film pages, the catalog and search

EXCLUDED_ARTICLES = ROOT / "data" / "processed" / "chunks_excluded.csv"


def wikipedia_page(title: str, section: str | None = None) -> str:
    """Link to the current version of a Wikipedia article, or of one of its sections."""
    from urllib.parse import quote
    url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    return url + (f"#{quote(section.replace(' ', '_'))}" if section else "")


def latest_version(url: str) -> str:
    """A Wikipedia permalink to one revision (what the chunks store, so the evaluation can check the exact text)
    turned into a link to the current version of the same page and section. Other links are returned unchanged."""
    from urllib.parse import parse_qs, unquote, urlsplit
    parts = urlsplit(url)
    if parts.netloc != "en.wikipedia.org" or not parts.path.endswith("index.php"):
        return url
    title = parse_qs(parts.query).get("title", [""])[0]
    return wikipedia_page(title, unquote(parts.fragment).replace("_", " ") or None) if title else url


def film_key(title: str, year) -> str:
    """The id of one film in the app's links, from its title and year, for example 'seven-samurai-1954'."""
    year = "" if year is None or pd.isna(year) or year == "" else str(int(year))
    return f"{norm(title)} {year}".strip().replace(" ", "-")


def load_catalog() -> pd.DataFrame:
    """One row per film for the app's pages, indexed by film_key: the same cleaned table the chunks were built
    from, so low-confidence TMDB matches have no overview or poster. Sorted by title, ignoring A, An and The."""
    import sys
    if str(ROOT / "src") not in sys.path:                     # the data modules import each other by bare name
        sys.path.append(str(ROOT / "src"))
    from first_look import load_tables
    from prepare_documents import prepare_films

    films, manifest = load_tables()
    df = prepare_films(films, manifest)
    excluded = set(pd.read_csv(EXCLUDED_ARTICLES)["article"]) if EXCLUDED_ARTICLES.exists() else set()
    has_article = df["article_status"].eq("ok") & ~df["wiki_title"].isin(excluded)
    df["wikipedia_url"] = [wikipedia_page(title) if ok else None for title, ok in zip(df["article_title"], has_article)]
    for column in ("genres", "countries"):
        df[column] = df[column].map(lambda value: list(value) if isinstance(value, (list, np.ndarray)) else [])
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")
    df["key"] = [film_key(title, year) for title, year in zip(df["title"], df["year"])]
    df["sort_title"] = df["title"].map(norm)
    df["search_text"] = [norm(" ".join(str(v) for v in values if isinstance(v, str)))
                         for values in zip(df["title"], df["tmdb_original_title"], df["director"])]
    return df.sort_values(["sort_title", "year"]).set_index("key", drop=False)


def search_films(catalog: pd.DataFrame, query: str) -> pd.DataFrame:
    """Films whose title, original title or director contains every word of the query (accents and case ignored)."""
    words = norm(query).split()
    if not words:
        return catalog.iloc[0:0]
    return catalog[catalog["search_text"].map(lambda text: all(word in text for word in words))]


def filter_catalog(catalog: pd.DataFrame, countries: list[str] = (), decades: list[int] = (),
                   genres: list[str] = ()) -> pd.DataFrame:
    """Films that match any chosen country, any chosen decade and any chosen genre. An empty choice keeps everything."""
    keep = pd.Series(True, index=catalog.index)
    if countries:
        keep &= catalog["countries"].map(lambda values: bool(set(values) & set(countries)))
    if decades:
        keep &= catalog["decade"].isin(decades)
    if genres:
        keep &= catalog["genres"].map(lambda values: bool(set(values) & set(genres)))
    return catalog[keep]


def films_in_reply(reply: dict, catalog: pd.DataFrame) -> list[str]:
    """The film_key of each pick in an answer. A pick is matched to the retrieved chunk of the same title first,
    so that two films with one title (a remake, for example) resolve to the one the answer drew on."""
    by_title = {norm(chunk["metadata"].get("title", "")): chunk["metadata"] for chunk in reply.get("retrieved", [])
                if chunk["metadata"].get("film_id")}
    keys = []
    for pick in reply.get("picks", []):
        meta = by_title.get(norm(pick))
        key = film_key(meta["title"], meta.get("year")) if meta else None
        if key not in catalog.index:
            same_title = catalog.index[catalog["sort_title"] == norm(pick)]
            key = same_title[0] if len(same_title) else None
        if key and key not in keys:
            keys.append(key)
    return keys


def sources_in_reply(reply: dict) -> list[dict]:
    """The chunks an answer cites, one per film and section: film title, section, source name and URL."""
    cited, seen, sources = set(reply.get("cited_ids", [])), set(), []
    for chunk in reply.get("retrieved", []):
        meta = chunk["metadata"]
        label = (meta.get("title"), meta.get("section"), meta.get("url"))
        if chunk["id"] in cited and meta.get("url") and label not in seen:
            seen.add(label)
            sources.append({"title": meta.get("title", ""), "section": meta.get("section", ""),
                            "source": {"wikipedia": "Wikipedia", "tmdb": "TMDB", "catalog": "Criterion list"}.get(meta.get("source"), meta.get("source", "")),
                            "url": latest_version(meta["url"])})
    return sources


STANDALONE_INSTRUCTIONS = """You turn the latest message of a conversation with a Criterion Collection film guide into one
question that can be understood without the conversation. Keep every constraint still in force from earlier turns
(decade, country, genre, director, runtime, mood) unless the latest message changes it. When the user asks for other or
different films, name the films already suggested so they can be avoided. If the latest message already stands on its
own, return it unchanged. Write only the question, in the user's words where possible."""

STANDALONE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["question"],
                     "properties": {"question": {"type": "string"}}}


def standalone_question(question: str, history: list[dict], client: OpenAI | None = None,
                        turns: int = 6) -> tuple[str, int, int]:
    """(the question rewritten to stand on its own, input tokens, output tokens), so that a follow-up such as
    "something shorter?" reaches the pipeline with the earlier constraints. `history` holds dicts with a role
    ('user' or 'assistant') and the text; only the last `turns` messages are sent. No history, no call."""
    if not history:
        return question, 0, 0
    lines = [f"{'User' if m['role'] == 'user' else 'Guide'}: {m['text']}" for m in history[-turns:]]
    message = "Conversation:\n" + "\n".join(lines) + f"\n\nLatest message: {question}"
    reply, input_tokens, output_tokens = _ask_json(client or _openai(), STANDALONE_INSTRUCTIONS, message,
                                                   "standalone_question", STANDALONE_SCHEMA)
    return reply["question"].strip() or question, input_tokens, output_tokens

