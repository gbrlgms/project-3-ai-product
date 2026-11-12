"""Section 3, "First look": shape, columns, and a handful of rows read in full.

Put these functions in src/functions.py (the notebook only calls them).  Run this file
directly for a quick text version of the whole section:  python first_look.py

What is one row?  In criterion_films_enriched.csv, ONE ROW = ONE DISTINCT FILM, identified by
its Wikipedia article title (wiki_title).  It joins three sources, the analog of joining Amazon
reviews to product metadata on parent_asin:
    Criterion list (Wikipedia)  ->  title, year, director, spine number, sets
    TMDB                        ->  genres, runtime, countries, keywords, overview, poster_url
    Wikipedia article           ->  one HTML file per film (data/raw/wikipedia/articles/)
The join key is wiki_title (= requested_title in articles_manifest.csv).
"""
from __future__ import annotations

import hashlib
import re
import textwrap
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

_DATA = Path(__file__).resolve().parents[1] / "data"     # works from the repo root and from notebooks/
PROCESSED = _DATA / "processed"
ARTICLES = _DATA / "raw" / "wikipedia" / "articles"
MANIFEST = _DATA / "raw" / "wikipedia" / "articles_manifest.csv"

# Spelling errors in the source list, each confirmed against TMDB or the film's Wikipedia article.
TITLE_CORRECTIONS = {
    "The Wages for Fear": "The Wages of Fear",
    "Harikari": "Harakiri",
    "Berlin Alexanderplantz": "Berlin Alexanderplatz",
    "Vengence is Mine": "Vengeance Is Mine",
    "Lone Wolf and Cub: Sword of Vengence": "Lone Wolf and Cub: Sword of Vengeance",
    "Sweet Sweetback's Baadassss Song": "Sweet Sweetback's Baadasssss Song",
    "Poil de carvotte": "Poil de carotte",
    "Letters d'amour": "Lettres d'amour",
}
LIST_COLUMNS = ["keywords", "top_cast", "genres", "languages", "countries", "tmdb_directors"]     # '|'-separated text in the CSV


# ------------------------------------------------------------------ loading

def _split_pipes(value) -> list[str]:
    """'Drama|War' -> ['Drama', 'War']; a missing value -> []."""
    return [part.strip() for part in value.split("|")] if isinstance(value, str) and value else []


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(films with title typos corrected and LIST_COLUMNS as lists, article manifest)."""
    films = pd.read_csv(PROCESSED / "criterion_films_enriched.csv")
    films["title"] = films["title"].replace(TITLE_CORRECTIONS)
    for col in LIST_COLUMNS:
        films[col] = films[col].map(_split_pipes)
    manifest = pd.read_csv(MANIFEST)
    return films, manifest


# ------------------------------------------------------------------ shape and columns

def _short(value, width: int = 40) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def overview(df: pd.DataFrame, name: str = "table") -> pd.DataFrame:
    """Print the shape; return one line per column: type, missing values, distinct values, an example."""
    print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    # Pandas cannot count lists, so count their text form; an empty list counts as missing.
    counted = df.apply(lambda s: s.map(lambda v: ("|".join(v) or None) if isinstance(v, list) else v))
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": counted.notna().sum(),
        "missing_%": (counted.isna().mean() * 100).round(1),
        "n_unique": counted.nunique(dropna=True),
        "example": df.apply(lambda s: _short(s[counted[s.name].notna()].iloc[0]) if counted[s.name].notna().any() else ""),
    })


# ------------------------------------------------------------------ rows read in full

def read_in_full(df: pd.DataFrame, n: int = 5, seed: int = 0, width: int = 100) -> None:
    """Print n random rows field by field, with nothing truncated (pandas hides long text)."""
    for idx, row in df.sample(n=min(n, len(df)), random_state=seed).iterrows():
        print("=" * width)
        print(f"row {idx}")
        for col, val in row.items():
            missing = not val if isinstance(val, list) else pd.isna(val)
            shown = "(missing)" if missing else str(val)
            print(textwrap.fill(shown, width=width, initial_indent=f"  {col + ':':<22}",
                                subsequent_indent=" " * 24))
    print("=" * width)


# ------------------------------------------------------------------ what is one row?

def check_key(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Is `cols` a unique key?  Prints the verdict; returns the duplicated rows (empty = unique)."""
    present = df.dropna(subset=cols)
    dup = present[present.duplicated(cols, keep=False)].sort_values(cols)
    verdict = "UNIQUE" if dup.empty else f"NOT unique ({len(dup)} rows share a key)"
    print(f"{cols}: {len(present):,} rows with a value, {verdict}; {len(df) - len(present):,} missing")
    return dup


def coverage(films: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Join films to their article status on wiki_title and report how complete the join is."""
    art = manifest[["requested_title", "status", "year_in_page", "director_in_page"]].rename(
        columns={"requested_title": "wiki_title", "status": "article_status"})
    joined = films.merge(art, on="wiki_title", how="left")
    has_link = joined["wiki_title"].notna()
    article_ok = joined["article_status"] == "ok"
    tmdb = joined["tmdb_id"].notna()
    print(f"films: {len(joined):,}")
    print(f"  with a Wikipedia article link: {has_link.sum():,}")
    print(f"  article downloaded (status ok): {article_ok.sum():,}")
    print(f"  matched to TMDB:                {tmdb.sum():,}")
    print(f"  article AND TMDB (usable):      {(article_ok & tmdb).sum():,}")
    print(joined["article_status"].fillna("no article link").value_counts().to_string())
    return joined


# ------------------------------------------------------------------ reading an article

def safe_name(title: str) -> str:
    """Same file stem as download_data.safe_name (keep the two in sync)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")[:80]
    return f"{slug}_{hashlib.sha1(title.encode('utf-8')).hexdigest()[:8]}"


_SKIP_SECTIONS = {"references", "external links", "see also", "notes", "further reading",
                  "bibliography", "footnotes"}


def article_text(title: str, max_chars: int | None = None) -> str:
    """Readable text of a downloaded article: infobox lines, then headings and paragraphs."""
    path = ARTICLES / f"{safe_name(title)}.html"
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for junk in soup.select("style, sup.reference, span.mw-editsection, table.navbox, div.navbox, "
                            "div.reflist, ol.references, div.hatnote, .mw-empty-elt"):
        junk.extract()
    lines: list[str] = []
    infobox = soup.select_one("table.infobox")
    if infobox:
        for tr in infobox.find_all("tr"):
            th, td = tr.find("th"), tr.find("td")
            if th and td:
                lines.append(f"[infobox] {th.get_text(' ', strip=True)}: {td.get_text(' ', strip=True)}")
        infobox.extract()
    skipping = False
    for el in soup.find_all(["h2", "h3", "p"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name == "p":
            if not skipping:
                lines.append(text)
        else:
            skipping = text.lower() in _SKIP_SECTIONS
            if not skipping:
                lines.append(f"\n## {text}")
    out = "\n".join(lines)
    return out if max_chars is None else out[:max_chars]


# ------------------------------------------------------------------ the whole section

def main() -> None:
    films, manifest = load_tables()

    print("### Shape and columns")
    print(overview(films, "criterion_films_enriched").to_string())

    print("\n### A handful of rows, read in full (main spine-numbered table only)")
    read_in_full(films[films["in_main_table"]], n=5, seed=0)

    print("\n### What is one row?")
    check_key(films, ["wiki_title"])
    check_key(films, ["tmdb_id"])
    dup = check_key(films, ["film_spine"])
    if not dup.empty:
        print(dup[["film_spine", "title", "year", "laserdisc_only"]].head(10).to_string(index=False))

    print("\n### The join: films -> article -> TMDB")
    joined = coverage(films, manifest)

    print("\n### One article, read in full")
    ok = joined[(joined["article_status"] == "ok") & joined["tmdb_id"].notna()]
    if len(ok):
        row = ok.iloc[0]
        print(f"{row['title']} ({row['year']}), article '{row['wiki_title']}'\n")
        print(article_text(row["wiki_title"]))


if __name__ == "__main__":
    main()
