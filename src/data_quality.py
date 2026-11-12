"""Section 4, data quality: find the problems, write down what you do about them, and put the
cleaning in functions the app can reuse.

Put these functions in src/functions.py (the notebook calls them, and so does the app).
Run this file directly for a text version of the whole section:
    python data_quality.py                    # catalog + TMDB texts + articles
    python data_quality.py --skip-articles    # skip the (slower) article audit

Checks (the issues listed in the brief):
  empty and very short texts   overview, tagline, article bodies
  HTML and entities            <br />, &#34;, &amp;, stray tags
  duplicates                   identical overviews, same film listed twice, two rows -> one TMDB
                               film, two titles -> one Wikipedia page
  texts in other languages     TMDB falls back to the original language or leaves the overview empty
  missing titles or metadata   year, director, article, TMDB match, genres, runtime, poster...

The "proposed action" text in the findings table is only a default. Read the examples, then
edit the decisions so they say what you actually did.
"""
from __future__ import annotations

import argparse
import html
import re
import unicodedata

import pandas as pd

# ------------------------------------------------------------------ cleaning (also used by the app)

_BR = re.compile(r"<\s*br\s*/?\s*>", re.I)
_BLOCK_END = re.compile(r"</\s*(p|div|li|h[1-6]|tr)\s*>", re.I)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")                 # real tags only: "a < b and c > d" is left alone
_MARKUP = re.compile(r"</?[a-zA-Z][^>]*>|&#?\w+;")      # tags or entities (for the audit)
_CITE = re.compile(r"\[(?:\d+|[a-z]|citation needed|clarification needed|when\?|who\?|note \d+)\]", re.I)
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)


def clean_text(text) -> str:
    """Plain, readable text: entities decoded, <br /> to a line break, tags and citation
    markers removed, odd whitespace and invisible characters normalized. Safe to call twice."""
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    s = str(text)
    for _ in range(2):                                   # twice: handles double-escaped "&amp;#34;"
        s = html.unescape(s)
    s = _BR.sub("\n", s)
    s = _BLOCK_END.sub("\n", s)
    s = _TAG.sub(" ", s)
    s = unicodedata.normalize("NFC", s).translate(_INVISIBLE).replace("\xa0", " ")
    s = _CITE.sub("", s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r" ?\n ?", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r" +([,.;:!?])", r"\1", s)                # "word ." left by removed markup -> "word."
    return s.strip()


def clean_pipe_list(value) -> str | list[str]:
    """'Drama| Drama |Action||' -> 'Drama|Action'. A list stays a list, cleaned the same way."""
    if isinstance(value, list):
        return list(dict.fromkeys(p for p in map(clean_text, value) if p))
    parts = [clean_text(p) for p in ("" if pd.isna(value) else str(value)).split("|")]
    return "|".join(dict.fromkeys(p for p in parts if p))


# ------------------------------------------------------------------ language

_STOPWORDS = {
    "en": "the and of to in is a that it with as for was on by his her he she at from an are be this "
          "which or their but not have had they who".split(),
    "fr": "le la les des du de et un une est dans qui que pour sur au aux avec son sa ses il elle pas "
          "ce cette sont par plus".split(),
    "es": "el la los las de del y en un una es que por con para su sus al se lo como más pero fue".split(),
    "de": "der die das und ist in den von zu mit sich des auf für nicht ein eine dem er sie es im als "
          "auch an".split(),
    "it": "il lo la i gli le di del della e un una è che per con su sono non si come più nel dei alla".split(),
    "pt": "o a os as de do da e um uma é que para com em seu sua não se por mais foi dos das".split(),
}
_SCRIPTS = [  # (code, regex of characters that identify the script)
    ("ja", r"[\u3040-\u30ff]"), ("ko", r"[\uac00-\ud7af]"), ("zh", r"[\u4e00-\u9fff]"),
    ("ru", r"[\u0400-\u04ff]"), ("ar", r"[\u0600-\u06ff]"), ("he", r"[\u0590-\u05ff]"),
    ("el", r"[\u0370-\u03ff]"), ("hi", r"[\u0900-\u097f]"), ("th", r"[\u0e00-\u0e7f]"),
]


def detect_language(text: str) -> str:
    """Language code, or 'unknown' for short or ambiguous text. Uses the `langdetect` package if
    it is installed (pip install langdetect), otherwise a small stopword-and-script heuristic."""
    if not text or len(text) < 20:
        return "unknown"
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(text)
    except ImportError:
        pass
    except Exception:
        return "unknown"
    for code, pattern in _SCRIPTS:
        if len(re.findall(pattern, text)) >= 5:
            return code
    tokens = re.findall(r"[^\W\d_]+", text.lower())
    if len(tokens) < 6:
        return "unknown"
    scores = {lang: sum(t in set(words) for t in tokens) / len(tokens) for lang, words in _STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, second = ranked[0], ranked[1]
    return best[0] if best[1] >= 0.12 and best[1] - second[1] >= 0.02 else "unknown"


# ------------------------------------------------------------------ audits

ACTIONS = {  # defaults: edit them so they say what you actually did
    "empty": "Keep the film. Build its fact card from the other fields and leave it out of overview-based retrieval.",
    "short": "Keep, but do not embed on its own: embed it together with title, genres and keywords.",
    "had_markup": "Run clean_text (decode entities, <br /> to a line break, strip tags) before storing or embedding.",
    "duplicate": "Inspect: identical texts often mean a placeholder or a wrong TMDB match. Fix the match or flag it.",
    "other_language": "Keep, tag the language, and say so in the limitations (no translation).",
}


def audit_text_column(df: pd.DataFrame, col: str, min_chars: int = 40, min_words: int = 6) -> pd.DataFrame:
    """One row per input row, with boolean flags for the issues in `col`."""
    raw = df[col]
    cleaned = raw.map(clean_text)
    empty = cleaned.eq("")
    words = cleaned.str.split().str.len()
    short = ~empty & ((cleaned.str.len() < min_chars) | (words < min_words))
    key = cleaned.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    language = cleaned.map(lambda t: detect_language(t) if t else "")
    return pd.DataFrame({
        "empty": empty,
        "short": short,
        "had_markup": raw.fillna("").astype(str).str.contains(_MARKUP, regex=True),
        "duplicate": ~empty & key.duplicated(keep=False),
        "other_language": ~empty & ~short & ~language.isin(["en", "unknown"]),
        "language": language,
    }, index=df.index)


def findings_table(flags: pd.DataFrame, labels: pd.Series, name: str, n_examples: int = 3) -> pd.DataFrame:
    """Turn flags into rows: where, issue, how many, examples, proposed action."""
    rows = []
    for issue in ACTIONS:
        mask = flags[issue]
        rows.append({"where": name, "issue": issue.replace("_", " "), "n": int(mask.sum()),
                     "examples": "; ".join(labels[mask].astype(str).head(n_examples)),
                     "proposed action": ACTIONS[issue]})
    return pd.DataFrame(rows)


def audit_articles(manifest: pd.DataFrame, min_chars: int = 2000) -> pd.DataFrame:
    """One row per downloaded article: length, sections, plot section, infobox, language, stub flag.
    Needs article_text (defined in first_look.py: keep it in the same functions.py)."""
    from first_look import article_text
    rows = []
    for title in manifest.loc[manifest["status"] == "ok", "requested_title"]:
        try:
            text = clean_text(article_text(title))
        except FileNotFoundError:
            rows.append({"wiki_title": title, "file_found": False})
            continue
        body = "\n".join(l for l in text.split("\n") if not l.startswith("[infobox]"))
        sections = re.findall(r"^## (.+)$", text, flags=re.M)
        rows.append({
            "wiki_title": title, "file_found": True, "chars": len(text), "words": len(text.split()),
            "n_sections": len(sections),
            "has_plot": any(re.match(r"(plot|synopsis|premise|story|summary)", s, re.I) for s in sections),
            "has_infobox": "[infobox]" in text,
            "language": detect_language(body[:1500]),
        })
    cols = ["wiki_title", "file_found", "chars", "words", "n_sections", "has_plot", "has_infobox", "language"]
    out = pd.DataFrame(rows, columns=cols)
    for c in ("file_found", "has_plot", "has_infobox"):
        out[c] = out[c].fillna(False).astype(bool)
    out["stub"] = out["file_found"] & (out["chars"] < min_chars)
    return out


def catalog_issues(films: pd.DataFrame, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Missing titles/metadata and duplicate keys in the film table, as findings rows."""
    rows = []

    def add(issue, mask, action, label_col="title"):
        rows.append({"where": "catalog", "issue": issue, "n": int(mask.sum()),
                     "examples": "; ".join(films.loc[mask, label_col].astype(str).head(3)),
                     "proposed action": action})

    blank_title = films["title"].isna() | films["title"].astype(str).str.strip().eq("")
    add("missing title", blank_title, "Drop the row: nothing to retrieve or show.", "wiki_title")
    add("missing year", films["year"].isna(), "Keep; it cannot match decade filters, so flag it.")
    add("missing director", films["director"].isna() | films["director"].astype(str).str.strip().eq(""),
        "Keep; fill from TMDB directors where available.")
    add("no Wikipedia article link", films["wiki_title"].isna(),
        "Keep for the catalog; answer from TMDB fields only and say there is no article.")
    if "tmdb_id" in films:
        add("no TMDB match", films["tmdb_id"].isna(), "Keep for the catalog, with no genres or runtime filters.")
        if "match_confidence" in films:
            add("low-confidence TMDB match", films["match_confidence"].eq("low"),
                "Review tmdb_review.csv; fix by hand in id_map.json or exclude the TMDB fields for these.")
        matched = films["tmdb_id"].notna()
        add("same TMDB film on 2 rows", matched & films.duplicated("tmdb_id", keep=False),
            "Check whether it is a wrong match or a genuine re-release; keep one row per film for retrieval.")
        for col in ("genres", "runtime", "poster_url"):
            if col in films:
                absent = films[col].isna() | films[col].map(lambda v: isinstance(v, list) and not v)
                if col == "runtime":                         # 0 minutes means "unknown" in TMDB
                    absent |= pd.to_numeric(films[col], errors="coerce").le(0)
                add(f"matched but no {col.replace('_', ' ')}", matched & absent,
                    "Keep; leave it out of filters and show a placeholder." if col == "poster_url"
                    else "Keep; exclude from filters on this field.")
    title_key = films["title"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    add("same title and year on 2 rows", ~blank_title & films.assign(k=title_key).duplicated(["k", "year"], keep=False),
        "Merge or drop the duplicate; keep the row with the article link.")
    if manifest is not None:
        art = manifest.set_index("requested_title")
        status = films["wiki_title"].map(art["status"])
        add("article is a disambiguation page", status.eq("disambiguation"),
            "Fix the Wikipedia title by hand or drop the article.", "wiki_title")
        add("article page missing", status.eq("missing"), "Treat as 'no article'.", "wiki_title")
        if "pageid" in manifest:
            pid = films["wiki_title"].map(art["pageid"])
            add("two rows share one Wikipedia page", pid.notna() & pid.duplicated(keep=False),
                "Redirects to the same page: merge the rows.", "wiki_title")
        if "year_in_page" in manifest:
            add("article may be the wrong film (year not found)", films["wiki_title"].map(art["year_in_page"]).eq(False),
                "Check by hand before using its text.", "wiki_title")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the cleaning function

TEXT_COLS = ["title", "director", "overview", "tagline", "tmdb_original_title"]
LIST_COLS = ["genres", "countries", "languages", "keywords", "top_cast", "tmdb_directors"]


def clean_catalog(films: pd.DataFrame, min_overview_chars: int = 40) -> pd.DataFrame:
    """Cleaned copy of the film table, ready for the app and the pipeline. No rows are dropped:
    problems are marked in `quality_flags` so the decisions stay visible and reversible."""
    df = films.copy()
    for c in TEXT_COLS:
        if c in df:
            df[c] = df[c].map(clean_text).replace("", pd.NA)
    for c in LIST_COLS:
        if c in df:
            df[c] = df[c].map(clean_pipe_list).replace("", pd.NA)
    if "runtime" in df:
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
        df.loc[df["runtime"] <= 0, "runtime"] = pd.NA            # TMDB uses 0 for unknown
        df["runtime"] = df["runtime"].round().astype("Int64")

    flags = pd.DataFrame(index=df.index)
    if "overview" in df:
        ov = df["overview"].fillna("")
        flags["no_overview"] = ov.eq("")
        flags["short_overview"] = ov.ne("") & ov.str.len().lt(min_overview_chars)
        lang = ov.map(lambda t: detect_language(t) if t else "")
        df["overview_language"] = lang.replace("", pd.NA)
        flags["overview_not_english"] = ~lang.isin(["", "en", "unknown"]) & ~flags["short_overview"]
    flags["no_tmdb"] = df["tmdb_id"].isna() if "tmdb_id" in df else False
    if "match_confidence" in df:
        flags["low_tmdb_match"] = df["match_confidence"].eq("low")
    flags["no_article"] = df["wiki_title"].isna() if "wiki_title" in df else False
    flags["no_year"] = df["year"].isna()
    df["quality_flags"] = flags.apply(lambda r: ";".join(r.index[r.astype(bool)]), axis=1)
    return df


# ------------------------------------------------------------------ the whole section

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-articles", action="store_true")
    args = ap.parse_args()
    from first_look import load_tables

    films, manifest = load_tables()
    parts = [catalog_issues(films, manifest)]
    flags = audit_text_column(films, "overview")          # taglines are optional by nature: not audited
    parts.append(findings_table(flags, films["title"], "TMDB overview"))
    print("overview languages:\n", flags.loc[~flags["empty"], "language"].value_counts().to_string(), "\n")

    if not args.skip_articles:
        arts = audit_articles(manifest)
        for issue, mask in [("stub article (short text)", arts["stub"]),
                            ("article without a plot section", arts["file_found"] & ~arts["has_plot"]),
                            ("article without an infobox", arts["file_found"] & ~arts["has_infobox"]),
                            ("article not in English", arts["file_found"] & ~arts["language"].isin(["en", "unknown"])),
                            ("article file not found", ~arts["file_found"])]:
            parts.append(pd.DataFrame([{"where": "articles", "issue": issue, "n": int(mask.sum()),
                                        "examples": "; ".join(arts.loc[mask, "wiki_title"].head(3)),
                                        "proposed action": "Decide by hand after reading the examples."}]))
        print(arts[["chars", "words", "n_sections"]].describe().round(0).to_string(), "\n")

    findings = pd.concat(parts, ignore_index=True)
    with pd.option_context("display.max_colwidth", 70, "display.width", 250):
        print(findings[findings["n"] > 0].to_string(index=False))

    cleaned = clean_catalog(films)
    marked = films["overview"].fillna("").str.contains(_MARKUP, regex=True)
    print(f"\nclean_catalog: {len(cleaned)} rows kept; rows with flags: {cleaned['quality_flags'].ne('').sum()}")
    for _, r in films[marked].head(3).iterrows():
        print(f"\nbefore: {r['overview'][:200]!r}\nafter:  {clean_text(r['overview'])[:200]!r}")


if __name__ == "__main__":
    main()
