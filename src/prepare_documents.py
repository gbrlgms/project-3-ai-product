"""Prepare the documents for retrieval: what is searched, how it is chunked, what each chunk keeps.

Put these functions in src/functions.py (the app needs format_citation and load_chunks).
Run this file directly to build data/processed/chunks.jsonl and print a report:
    python prepare_documents.py
    python prepare_documents.py --target-words 180 --overlap 0     # try another chunk size

WHAT IS SEARCHED (one "document" = one chunk of one source, six kinds):
    catalog_facts   built from the Criterion list + TMDB: title, year, decade, director, country,
                    genres, runtime, cast, spine number.       Answers "which films..." and filters.
    infobox         the Wikipedia infobox lines (cinematography, music, writers...).  "Who shot it?"
    tmdb_overview   TMDB overview, tagline and keywords.        Short, spoiler-light descriptions.
    lead            the first paragraph(s) of the Wikipedia article (genre, premise, context).
    plot, cast, production, release, reception, awards, themes_legacy, other
                    the article's sections; `plot` chunks carry spoilers=True so a recommendation
                    can leave them out and answer from lead, TMDB and reception instead.
    The Wikipedia page about the Criterion Collection itself is chunked the same way. It is not a
    film, so its film fields (film_id, year, director...) are empty.

HOW IT IS CHUNKED  sections first (never across headings), then paragraphs, merged up to about
    --target-words words (soft maximum --max-words), with --overlap sentences repeated at the start
    of the next chunk of the same section. A tiny tail is merged into the previous chunk, and
    sections under 15 words are dropped. Each chunk gets a one-line header (film, year, section)
    in `embed_text` so it still makes sense on its own; `text` stays clean for display and citing.

WHAT EACH CHUNK KEEPS (so answers can cite it): chunk_id, film_id, title, year, decade, director,
    film_spine, country, genres, runtime, tmdb_id, section, section_type, chunk_index, n_chunks,
    spoilers, source (wikipedia | tmdb | catalog), source_title, revid, url (permalink to the
    exact revision and section), license, poster_url, tmdb_url, quality_flags, n_words.

WHAT IS LEFT OUT (listed in the report, never silently): articles that are disambiguation pages or
    probably the wrong film (neither year nor director found on the page); and
    the TMDB fields of low-confidence matches (a wrong match would corrupt genre and runtime filters).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from data_quality import clean_catalog, clean_text
from explore_texts import split_sections
from first_look import ARTICLES, article_text, load_tables, safe_name

OUT = Path("../data/processed/chunks.jsonl")
LIST_TITLE = "List_of_films_in_the_Criterion_Collection"
COLLECTION_PAGE = "The Criterion Collection"
TMDB_FIELDS = ["genres", "languages", "runtime", "overview", "tagline", "keywords",
               "top_cast", "tmdb_directors", "poster_url", "tmdb_url", "original_language"]

# ------------------------------------------------------------------ sentences and chunking

_ABBR = re.compile(r"(?:^|\s)(?:mr|mrs|ms|dr|st|vs|jr|sr|no|inc|co|ltd|etc|u\.s|u\.k|[A-Z])\.$", re.I)


def split_sentences(text: str) -> list[str]:
    """Sentence split with a guard for abbreviations and initials ("Dr. Strangelove", "J. Smith")."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“‘(])", text.strip())
    out: list[str] = []
    for p in parts:
        if out and _ABBR.search(out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    return [s for s in out if s]


def _words(text: str) -> int:
    return len(text.split())


def _split_long(paragraph: str, target: int, hard_max: int) -> list[str]:
    """Break an over-long paragraph into groups of sentences."""
    groups, cur = [], []
    for s in split_sentences(paragraph):
        if cur and _words(" ".join(cur)) + _words(s) > target:
            groups.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        groups.append(" ".join(cur))
    return groups


def chunk_section(text: str, target_words: int = 220, max_words: int = 320,
                  min_words: int = 40, overlap_sentences: int = 1) -> list[str]:
    """Split one section into chunks of about target_words words. max_words is a soft limit."""
    units: list[str] = []
    for p in (p.strip() for p in text.split("\n") if p.strip()):
        units.extend([p] if _words(p) <= max_words else _split_long(p, target_words, max_words))
    chunks: list[str] = []
    cur: list[str] = []
    n_overlap = 0
    for u in units:
        cur_words = sum(_words(x) for x in cur)
        if cur and cur_words >= min_words and cur_words + _words(u) > target_words:
            chunks.append("\n".join(cur))
            tail = split_sentences(" ".join(cur))[-overlap_sentences:] if overlap_sentences else []
            cur = ([" ".join(tail)] if tail else []) + [u]
            n_overlap = 1 if tail else 0
        else:
            cur.append(u)
    if cur:
        last = "\n".join(cur)
        if chunks and _words("\n".join(cur[n_overlap:])) < min_words:
            chunks[-1] += "\n" + "\n".join(cur[n_overlap:])        # merge a tiny tail, without the overlap
        else:
            chunks.append(last)
    return chunks


# ------------------------------------------------------------------ section types

SECTION_TYPES = [   # first match wins
    ("plot", r"^(plot|synopsis|premise|story|summary)"),
    ("cast", r"^(cast|casting|characters)\b"),
    ("awards", r"(awards?|accolades|nominations|honou?rs)"),
    ("reception", r"(reception|critical|reviews?|response|box office)"),
    ("themes_legacy", r"(theme|analysis|style|interpretation|symbolism|legacy|influence|reputation|remake)"),
    ("production", r"(production|development|filming|writing|photography|post-production|music|score|soundtrack)"),
    ("release", r"(release|distribution|home media|restoration|theatrical|marketing|premiere|screening)"),
]


def section_type(heading: str) -> str:
    for name, pattern in SECTION_TYPES:
        if re.search(pattern, heading, re.I):
            return name
    return "other"


# ------------------------------------------------------------------ texts of the non-article chunks

def _py(v):
    """JSON-safe Python value (numpy numbers, NA and NaN become plain numbers or None)."""
    if v is None or (not isinstance(v, (list, dict, str)) and pd.isna(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _commas(pipe_value) -> str:
    """'Drama|Thriller' or ['Drama', 'Thriller'] -> 'Drama, Thriller'; missing values (None, NaN, NA) -> ''."""
    if isinstance(pipe_value, list):
        return ", ".join(pipe_value)
    if pipe_value is None or (not isinstance(pipe_value, str) and pd.isna(pipe_value)):
        return ""
    return ", ".join(p for p in str(pipe_value).split("|") if p)


def _pipe_text(v):
    """A list as one '|'-separated string, the form chunk metadata stores; missing or empty -> None."""
    return ("|".join(v) or None) if isinstance(v, list) else _py(v)


def _int(v):
    """Whole numbers stay whole (1974, not 1974.0); missing -> None."""
    return None if v is None or pd.isna(v) else int(v)


def catalog_facts_text(r: pd.Series) -> str:
    year = f" ({int(r['year'])})" if pd.notna(r["year"]) else ""
    lines = [f"{r['title']}{year}."]
    if pd.notna(r.get("director")):
        lines.append(f"Directed by {r['director']}.")
    if pd.notna(r["year"]):
        lines.append(f"Decade: {int(r['year']) // 10 * 10}s.")
    country = _commas(r.get("countries"))
    if country:
        lines.append(f"Country: {country}.")
    for label, col in (("Genres", "genres"), ("Languages", "languages"), ("Top cast", "top_cast")):
        if _commas(r.get(col)):
            lines.append(f"{label}: {_commas(r[col])}.")
    if pd.notna(r.get("runtime")):
        lines.append(f"Runtime: {int(r['runtime'])} minutes.")
    if pd.notna(r.get("film_spine")) and not r.get("laserdisc_only"):
        lines.append(f"Criterion Collection spine number {int(r['film_spine'])}.")
    if isinstance(r.get("sets"), str) and r["sets"]:
        lines.append(f"Also released in: {r['sets']}.")
    return " ".join(lines)


def tmdb_text(r: pd.Series) -> str:
    parts = []
    if isinstance(r.get("overview"), str):
        parts.append(f"Overview: {r['overview']}")
    if isinstance(r.get("tagline"), str):
        parts.append(f"Tagline: {r['tagline']}")
    if _commas(r.get("keywords")):
        parts.append(f"Keywords: {_commas(r['keywords'])}")
    return "\n".join(parts)


def infobox_lines(raw_article_text: str) -> str:
    lines = [clean_text(l.replace("[infobox]", "", 1)) for l in raw_article_text.split("\n") if l.startswith("[infobox]")]
    return "\n".join(l for l in lines if l and len(l) < 400)


# ------------------------------------------------------------------ building the chunks

def permalink(title: str, revid, section: str | None = None) -> str:
    url = f"https://en.wikipedia.org/w/index.php?title={quote(title.replace(' ', '_'))}&oldid={int(revid)}"
    return url + (f"#{quote(section.replace(' ', '_'))}" if section else "")


def prepare_films(films: pd.DataFrame, manifest: pd.DataFrame, trust_low_tmdb: bool = False) -> pd.DataFrame:
    """Cleaned film table joined to the article status; TMDB fields blanked for low-confidence matches."""
    df = clean_catalog(films)
    art = manifest.rename(columns={"title": "article_title", "revid": "article_revid", "status": "article_status"})
    keep = [c for c in ("requested_title", "article_status", "article_title", "article_revid",
                        "year_in_page", "director_in_page") if c in art]
    df = df.merge(art[keep], left_on="wiki_title", right_on="requested_title", how="left")
    if not trust_low_tmdb and "match_confidence" in df:
        low = df["match_confidence"].eq("low")
        for c in TMDB_FIELDS:
            if c in df:
                df.loc[low, c] = pd.NA
    return df


def film_id(r: pd.Series) -> str:
    if isinstance(r.get("wiki_title"), str):
        return r["wiki_title"]
    return f"{r['title']}|{int(r['year']) if pd.notna(r['year']) else ''}"


def _make_chunks(base: dict, slug: str, sec: str, stype: str, texts: list[str], source: str, source_title: str,
                 revid, url, license_: str, spoilers: bool = False) -> list[dict]:
    label = f"{base['title']} ({base['year']})" if base["year"] else base["title"]
    return [{
        **base, "chunk_id": f"{slug}:{safe_name(sec)[:40]}:{i}", "section": sec, "section_type": stype,
        "chunk_index": i, "n_chunks": len(texts), "spoilers": spoilers, "source": source,
        "source_title": source_title, "revid": _int(revid), "url": url, "license": license_,
        "n_words": _words(t), "text": t, "embed_text": f"{label} | {sec}\n{t}",
    } for i, t in enumerate(texts)]


def _article_chunks(base: dict, slug: str, raw: str, a_title: str, a_rev, sizes: tuple, min_section_words: int) -> list[dict]:
    """Chunks of one Wikipedia article: infobox, lead, then each section."""
    lead, sections = split_sections(clean_text(raw))
    wiki = dict(source="wikipedia", source_title=a_title, revid=a_rev, license_="CC BY-SA 4.0 (Wikipedia)")
    out: list[dict] = []
    box = infobox_lines(raw)
    if box:
        out += _make_chunks(base, slug, "Infobox", "infobox", [box], url=permalink(a_title, a_rev), **wiki)
    parts = ([("Lead", lead)] if lead else []) + list(sections.items())
    for heading, body in parts:
        if _words(body) < min_section_words:
            continue
        stype = "lead" if heading == "Lead" else section_type(heading)
        out += _make_chunks(base, slug, heading, stype, chunk_section(body, *sizes),
                            url=permalink(a_title, a_rev, None if heading == "Lead" else heading),
                            spoilers=(stype == "plot"), **wiki)
    return out


def collection_chunks(sizes: tuple, min_section_words: int) -> list[dict]:
    """Chunks of the page about the Criterion Collection itself. It is not a film, so the film fields stay
    empty and it can never satisfy a film filter. Empty list if the page has not been downloaded."""
    meta_path = ARTICLES / f"{safe_name(COLLECTION_PAGE)}.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "ok":
        return []
    base = {"film_id": None, "title": meta["title"], "year": None, "decade": None, "director": None,
            "film_spine": None, "country": None, "genres": None, "runtime": None, "tmdb_id": None,
            "poster_url": None, "tmdb_url": None, "in_main_table": False, "quality_flags": ""}
    return _article_chunks(base, safe_name(COLLECTION_PAGE), article_text(COLLECTION_PAGE), meta["title"],
                           meta["revid"], sizes, min_section_words)


def build_chunks(films: pd.DataFrame, manifest: pd.DataFrame, target_words: int = 220, max_words: int = 320,
                 min_words: int = 40, overlap_sentences: int = 1, min_section_words: int = 15):
    """(list of chunk dicts, DataFrame of excluded articles with the reason)."""
    df = prepare_films(films, manifest)
    list_rev = int(df["revid"].dropna().iloc[0]) if "revid" in df and df["revid"].notna().any() else None
    sizes = (target_words, max_words, min_words, overlap_sentences)
    chunks: list[dict] = []
    excluded: list[dict] = []
    # Two catalog rows can share one article (Che: Part One and Two). Each row keeps its own
    # catalog and TMDB chunks; the article itself is chunked only once.
    # Rows are compared by the article's resolved title, so a link through a redirect counts as the same article.
    shared_articles = set(df.loc[df["article_title"].notna() & df["article_title"].duplicated(keep=False), "article_title"])
    articles_done: set[str] = set()

    for _, r in df.iterrows():
        fid = film_id(r)
        is_shared = r.get("article_title") in shared_articles
        slug = safe_name(f"{fid}|{r['title']}" if is_shared else fid)
        base = {
            "film_id": fid, "title": r["title"], "year": _int(r["year"]),
            "decade": _int(r["year"]) // 10 * 10 if pd.notna(r["year"]) else None,
            "director": _py(r.get("director")), "film_spine": _int(r.get("film_spine")),
            "country": _commas(r.get("countries")) or None,
            "genres": _pipe_text(r.get("genres")), "runtime": _int(r.get("runtime")), "tmdb_id": _int(r.get("tmdb_id")),
            "poster_url": _py(r.get("poster_url")), "tmdb_url": _py(r.get("tmdb_url")),
            "in_main_table": bool(r.get("in_main_table")), "quality_flags": _py(r.get("quality_flags")) or "",
        }

        list_url = f"https://en.wikipedia.org/w/index.php?title={LIST_TITLE}" + (f"&oldid={list_rev}" if list_rev else "")
        chunks += _make_chunks(base, slug, "Catalog facts", "catalog_facts", [catalog_facts_text(r)], "catalog",
                               "List of films in the Criterion Collection", list_rev, list_url,
                               "CC BY-SA 4.0 (Wikipedia) + TMDB")
        if tmdb_text(r):
            chunks += _make_chunks(base, slug, "TMDB overview", "tmdb_overview", [tmdb_text(r)], "tmdb", "TMDB", None,
                                   r.get("tmdb_url"), "TMDB (non-commercial, attribution required)")

        if r.get("article_status") != "ok":
            if isinstance(r.get("wiki_title"), str):
                excluded.append({"title": r["title"], "article": r["wiki_title"],
                                 "reason": f"article status: {r.get('article_status') or 'not downloaded'}"})
            continue
        if r.get("year_in_page") is False and r.get("director_in_page") is False:
            excluded.append({"title": r["title"], "article": r["wiki_title"],
                             "reason": "probably the wrong film (neither year nor director on the page)"})
            continue
        try:
            raw = article_text(r["wiki_title"])
        except FileNotFoundError:
            excluded.append({"title": r["title"], "article": r["wiki_title"], "reason": "article file not found"})
            continue
        a_title, a_rev = r["article_title"], r["article_revid"]
        if is_shared:
            if a_title in articles_done:
                continue
            articles_done.add(a_title)
            # The article covers every row that shares it, so it takes the article's title
            # and drops the fields that belong to one row only.
            base = {**base, "title": a_title, "runtime": None, "tmdb_id": None, "poster_url": None, "tmdb_url": None}
            slug = safe_name(fid)
        chunks += _article_chunks(base, slug, raw, a_title, a_rev, sizes, min_section_words)

    page = collection_chunks(sizes, min_section_words)
    if page:
        chunks += page
    else:
        excluded.append({"title": COLLECTION_PAGE, "article": COLLECTION_PAGE,
                         "reason": "page not downloaded: run the articles stage of download_data.py"})
    return chunks, pd.DataFrame(excluded, columns=["title", "article", "reason"])


# ------------------------------------------------------------------ for the app

def format_citation(chunk: dict) -> dict:
    """{'label': ..., 'url': ...} for showing a source under an answer."""
    t = f"{chunk['title']} ({chunk['year']})" if chunk.get("year") else chunk["title"]
    if chunk["source"] == "wikipedia":
        label = f"{t}: Wikipedia, \"{chunk['section']}\" (revision {chunk['revid']})"
    elif chunk["source"] == "tmdb":
        label = f"{t}: TMDB"
    else:
        label = f"{t}: Criterion Collection list on Wikipedia"
    return {"label": label, "url": chunk["url"]}


def save_chunks(chunks: list[dict], path: Path = OUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks(path: Path = OUT) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_report(chunks: list[dict], excluded: pd.DataFrame, films: pd.DataFrame) -> None:
    df = pd.DataFrame(chunks)
    print(f"{len(df):,} chunks from {df['film_id'].nunique():,} of {len(films):,} films")
    print("\nchunks by type:")
    print(df.groupby("section_type").agg(chunks=("chunk_id", "size"), median_words=("n_words", "median"),
                                          max_words=("n_words", "max"), films=("film_id", "nunique"))
          .sort_values("chunks", ascending=False).round(0).to_string())
    print(f"\nchunk size (words): median {df['n_words'].median():.0f}, "
          f"p10 {df['n_words'].quantile(.1):.0f}, p90 {df['n_words'].quantile(.9):.0f}, max {df['n_words'].max()}")
    print(f"spoiler chunks (plot): {int(df['spoilers'].sum()):,}")
    with_article = int(films["wiki_title"].isin(df.loc[df["source"] == "wikipedia", "film_id"].dropna()).sum())
    print(f"films with article chunks: {with_article:,}; catalog-and-TMDB only: {len(films) - with_article:,}")
    dup = df["chunk_id"].duplicated().sum()
    print(f"duplicate chunk ids: {dup}")
    if len(excluded):
        print(f"\nexcluded articles ({len(excluded)}):")
        print(excluded.groupby("reason").agg(n=("title", "size"), examples=("title", lambda s: "; ".join(s.head(3))))
              .to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-words", type=int, default=220)
    ap.add_argument("--max-words", type=int, default=320)
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--overlap", type=int, default=1, help="sentences repeated at the start of the next chunk")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    films, manifest = load_tables()
    chunks, excluded = build_chunks(films, manifest, args.target_words, args.max_words, args.min_words, args.overlap)
    save_chunks(chunks, Path(args.out))
    chunk_report(chunks, excluded, films)
    excluded.to_csv(Path(args.out).with_name("chunks_excluded.csv"), index=False)
    print(f"\nsaved {args.out} and chunks_excluded.csv")
    print("\nexample citation:", format_citation(next(c for c in chunks if c["source"] == "wikipedia")))


if __name__ == "__main__":
    main()
