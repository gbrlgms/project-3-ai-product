"""Download all data for the Criterion cineclub guide, locally, in one resumable script.

Three stages, run in this order (each can also be run alone with --stages):

  list      Wikipedia's "List of films in the Criterion Collection" -> film catalogue
              data/processed/criterion_appearances.csv   one row per table row on the page
              data/processed/criterion_films.csv         one row per distinct film
  articles  every Wikipedia article linked from the catalogue (one API request each),
            plus the page about The Criterion Collection itself (not listed in the manifest)
              data/raw/wikipedia/articles/*.html + *.json, articles_manifest.csv
  tmdb      TMDB metadata: genres, runtime, countries, languages, overview, keywords,
            directors, cast, and the poster URL
              data/processed/criterion_films_enriched.csv   <- the table to use
              data/processed/tmdb_review.csv                matches to check by hand

Everything is cached under data/raw/, so you can stop (Ctrl+C) and rerun at any time.
Keep data/raw/ out of git: this script rebuilds it.

Setup
  pip install requests beautifulsoup4 pandas python-dotenv
  put TMDB_TOKEN=<your TMDB read access token> in .env   (gitignored)
  set CONTACT below (Wikimedia asks for a descriptive User-Agent with contact details)

Usage
  python download_data.py --limit 10          # trial run of every stage
  python download_data.py                     # everything
  python download_data.py --stages tmdb       # only one stage (or: list articles)
  python download_data.py --main-only         # only spine-numbered films (articles + tmdb)
  python download_data.py --poster-size w342  # w92 w154 w185 w342 w500 (default) w780 original

How films are matched to TMDB, most reliable first: the film's Wikidata item (TMDb ID, then
IMDb ID via TMDB /find), then a title + year search. Each match gets a confidence label.
To fix a wrong match, edit data/raw/tmdb/id_map.json (key "Title|year"): set "tmdb_id" to the
right number and "method" to "manual", then rerun.

Licences and attribution
  Wikipedia text: CC BY-SA. Credit Wikipedia and keep each article's title + revision id.
  TMDB: non-commercial use; "This product uses the TMDB API but is not   endorsed or certified
  by TMDB." Poster images are not downloaded, only their URLs on TMDB's image server.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ------------------------------------------------------------------ configuration

CONTACT = "https://github.com/gbrlgms/project-3-ai-product/"   # <- your email or project URL
USER_AGENT = f"CriterionCineclubGuide/0.1 (bootcamp project; {CONTACT})"

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_BASE = "https://en.wikipedia.org"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
TMDB_API = "https://api.themoviedb.org/3"
DEFAULT_IMAGE_BASE = "https://image.tmdb.org/t/p/"
LIST_PAGE = "List_of_films_in_the_Criterion_Collection"
COLLECTION_PAGE = "The Criterion Collection"

RAW_WIKI = Path("data/raw/wikipedia")
LIST_CACHE = RAW_WIKI / "criterion_list.json"
ARTICLES = RAW_WIKI / "articles"
MANIFEST = RAW_WIKI / "articles_manifest.csv"
RAW_TMDB = Path("data/raw/tmdb")
OUT = Path("data/processed")
FILMS_CSV = OUT / "criterion_films.csv"

# Obvious typos seen in the source table; extend after you look at the report.
COUNTRY_FIXES = {"Austrailia": "Australia", "United Kindom": "United Kingdom"}
FINAL_STATUSES = {"ok", "disambiguation", "missing"}   # articles not retried on a normal rerun
EXCLUDED_BOX_SET = "Beastie Boys Video Anthology"
SPLIT_LIST_ENTRIES = {
    "che": [("Che: Part One", 2008), ("Che: Part Two", 2008)],
    "0 4": [(f"Pan {n}", 1969) for n in range(5)],
    "697 700": [(f"Pan {n}", 1974) for n in range(697, 701)],
    "jimi plays monterey shake otis at monterey": [
        ("Jimi Plays Monterey", None),
        ("Shake! Otis at Monterey", None),
    ],
    "the human condition": [
        ("The Human Condition I: No Greater Love", 1959),
        ("The Human Condition II: Road to Eternity", 1959),
        ("The Human Condition III: A Soldier's Prayer", 1961),
    ],
    "persian series 1 3": [
        ("Persian Series #1", 1999),
        ("Persian Series #2", 1999),
        ("Persian Series #3", 1999),
    ],
}
SPLIT_FILM_TITLES = {title for parts in SPLIT_LIST_ENTRIES.values() for title, _ in parts}
TITLE_RENAMES = {
    "sea urchiins": "Sea Urchins",
    "untitled for marilyn": "For Marilyn",
    "arabic": "Arabic Numeral Series",
    "critical mass": "Hapax Legomena III: Critical Mass",
    "mangrove": "Small Axe",
    "lovers rock": "Small Axe",
    "alex wheatle": "Small Axe",
    "l15 8 last": "Le 15/8",
    "report on the party and guests": "Party and Guests",
    "living magoroku the living magoroku": "The Living Magoroku",
    "i ll got to santiago": "I'm Going to Santiago",
}
MANUAL_TMDB_IDS = {
    "Hapax Legomena III: Critical Mass": 151002,
    "Hapax Legomena II: Poetic Justice": 145497,
    "The Human Condition I: No Greater Love": 31217,
    "The Human Condition II: Road to Eternity": 34528,
    "The Human Condition III: A Soldier's Prayer": 34530,
    "Persian Series #1": 385623,
}
MANUAL_TMDB_TV_IDS = {"Dekalog": 42699}


# ------------------------------------------------------------------ shared helpers

def norm(s) -> str:
    """Lowercase, strip accents and punctuation, drop a leading article."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"^(the|a|an) ", "", s)


def get_json(session: requests.Session, url: str, params: dict | None = None, retries: int = 5):
    """GET -> JSON, with back-off on network errors, 429, 5xx and Wikimedia maxlag.
    Returns None on 404; raises RuntimeError after `retries` failed attempts."""
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt * 2)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("Retry-After", 2 ** attempt * 5))
            print(f"  server said {r.status_code}, waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        data = r.json()
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict) and err.get("code") == "maxlag":
            time.sleep(float(r.headers.get("Retry-After", 5)))
            continue
        return data
    raise RuntimeError(f"gave up on {url}")


def load_cache(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def wiki_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


# ==================================================================== STAGE 1: list

def fetch_list_html(session: requests.Session, refresh: bool = False) -> tuple[str, int | None]:
    """(html, revision id) of the list page. One API request, cached to disk."""
    if LIST_CACHE.exists() and not refresh:
        d = json.loads(LIST_CACHE.read_text(encoding="utf-8"))
        return d["html"], d.get("revid")
    data = get_json(session, WIKI_API, {
        "action": "parse", "page": LIST_PAGE, "prop": "text|revid", "redirects": 1,
        "format": "json", "formatversion": 2, "maxlag": 5})
    if not data or "parse" not in data:
        raise SystemExit(f"Could not fetch the list page: {data}")
    p = data["parse"]
    LIST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    save_cache(LIST_CACHE, {"revid": p["revid"], "html": p["text"],
                            "fetched_at": time.strftime("%Y-%m-%d")})
    return p["text"], p["revid"]


def expand_table(table) -> list[list]:
    """Rows of cells with rowspan/colspan expanded (same cell object repeated)."""
    rows, spans = [], {}          # spans: column -> [cell, rows still to cover]
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"], recursive=False)
        out, col, ci = [], 0, 0
        while ci < len(cells) or (spans and col <= max(spans)):
            if col in spans:
                cell, left = spans[col]
                out.append(cell)
                if left <= 1:
                    del spans[col]
                else:
                    spans[col][1] = left - 1
                col += 1
            elif ci < len(cells):
                c = cells[ci]
                ci += 1
                rs = int(re.sub(r"\D", "", c.get("rowspan", "1")) or 1)
                cs = int(re.sub(r"\D", "", c.get("colspan", "1")) or 1)
                for _ in range(cs):
                    out.append(c)
                    if rs > 1:
                        spans[col] = [c, rs - 1]
                    col += 1
            else:                  # gap before a later spanned column
                out.append(None)
                col += 1
        rows.append(out)
    return rows


def clean_text(cell) -> str:
    if cell is None:
        return ""
    for junk in cell.select("sup.reference, span.mw-editsection, style"):
        junk.extract()
    text = cell.get_text(" ", strip=True)
    text = re.sub(r"\s+,", ",", text)
    return re.sub(r"\s+", " ", text).strip()


# Wikipedia namespaces whose links are not film articles (File:, Category:, Help:...).
# A plain ":" must NOT be treated as a namespace: "Twin Peaks: Fire Walk with Me" is an article.
NAMESPACES = re.compile(
    r"^(file|image|media|category|help|wikipedia|wp|project|template|portal|special|talk|"
    r"user|draft|module|mediawiki|timedtext|book|topic)(_talk)?:", re.I)


def first_film_link(cell):
    """(wikipedia title, url, has_article) of the first article link in a cell."""
    if cell is None:
        return None, None, False
    for a in cell.find_all("a", href=True):
        href = a["href"]
        if "redlink=1" in href or "new" in (a.get("class") or []):
            return None, None, False           # red link: no article yet
        if not href.startswith("/wiki/"):
            continue                           # external or /w/index.php links
        path = href[len("/wiki/"):].split("#")[0]
        title = unquote(path).replace("_", " ")
        if NAMESPACES.match(title.replace(" ", "_")):
            continue                           # File:, Category:, Help:... not a film
        return a.get("title") or title, WIKI_BASE + "/wiki/" + path, True
    return None, None, False


def canon_header(h: str) -> str:
    h = re.sub(r"\[.*?\]", "", h.lower())
    h = re.sub(r"\s+", " ", h).strip()
    if h == "spine":
        return "spine"
    if "spine number" in h:
        return "film_spine"
    if "name of box set" in h:
        return "set_name"
    if "eclipse" in h:
        return "set_name"
    if h == "series":
        return "set_number"
    if "film" in h and "year" in h:
        return "year"
    if "film" in h:                       # "Film", "Included films"
        return "title"
    if "year" in h:
        return "year"
    if "director" in h:
        return "director"
    if "country" in h:
        return "country"
    if "note" in h:
        return "notes"
    return h


def to_int(raw: str):
    m = re.search(r"\d+", raw or "")
    return int(m.group()) if m else None


def parse_tables(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    records, section, t_index = [], "", 0
    for el in soup.find_all(["h2", "h3", "h4", "table"]):
        if el.name != "table":
            section = clean_text(el)
            continue
        if "wikitable" not in (el.get("class") or []):
            continue
        t_index += 1
        grid = expand_table(el)
        if not grid:
            continue
        headers = [canon_header(clean_text(c)) for c in grid[0]]
        if "title" not in headers:
            continue                       # not a film table
        if "set_name" in headers and "spine" in headers:
            kind = "box_set"               # box set with its own spine number
        elif "set_name" in headers and "film_spine" in headers:
            kind = "director_set"
        elif "set_name" in headers or "set_number" in headers:
            kind = "eclipse"
        else:
            kind = "main"
        for row in grid[1:]:
            if all(c is not None and c.name == "th" for c in row if c is not None) and row:
                continue                   # repeated header row
            cells = dict(zip(headers, row))
            title_cell = cells.get("title")
            raw_title = clean_text(title_cell)
            if not raw_title:
                continue
            marks = set(re.findall(r"[†*]", raw_title))
            title = re.sub(r"\s*[†‡*]+", "", raw_title).strip()
            wiki_title, wiki_url, has_article = first_film_link(title_cell)
            spine_raw = clean_text(cells.get("spine"))
            film_spine_raw = clean_text(cells.get("film_spine"))
            if kind == "main":
                film_spine, set_spine = to_int(spine_raw), None
            else:
                film_spine, set_spine = to_int(film_spine_raw), to_int(spine_raw)
            year_raw = clean_text(cells.get("year"))
            m_year = re.search(r"\d{4}", year_raw)
            country = clean_text(cells.get("country"))
            notes = clean_text(cells.get("notes"))
            records.append({
                "section": section,
                "table_index": t_index,
                "kind": kind,
                "film_spine": film_spine,
                "film_spine_raw": film_spine_raw if kind != "main" else spine_raw,
                "set_spine": set_spine,
                "set_name": clean_text(cells.get("set_name")),
                "set_number": clean_text(cells.get("set_number")),
                "title": title,
                "year": int(m_year.group()) if m_year else None,
                "year_raw": year_raw,
                "director": clean_text(cells.get("director")),
                "country": COUNTRY_FIXES.get(country, country),
                "notes": notes,
                "laserdisc_only": "*" in marks,
                "laserdisc_and_dvd": "†" in marks,
                "previously_released": "previously released" in notes.lower(),
                "wiki_title": wiki_title,
                "wiki_url": wiki_url,
                "has_article": has_article,
            })
    return pd.DataFrame(records)


def split_combined_entries(app: pd.DataFrame) -> pd.DataFrame:
    """Expand combined Criterion listings into their individual film entries."""
    sources = app[app["title"].map(lambda title: norm(title) in SPLIT_LIST_ENTRIES)]
    additions = []
    for _, row in sources.iterrows():
        parts = SPLIT_LIST_ENTRIES[norm(row["title"])]
        for title, year in parts:
            split_row = row.copy()
            split_row["title"] = title
            if year is not None:
                split_row["year"] = year
                split_row["year_raw"] = str(year)
            additions.append(split_row)
    if not sources.empty:
        app = app[~app["title"].map(lambda title: norm(title) in SPLIT_LIST_ENTRIES)]
        app = pd.concat([app, pd.DataFrame(additions)], ignore_index=True)

    renamed = []
    for _, row in app.iterrows():
        title, year = row["title"], row["year"]
        key = norm(title)
        new_title = TITLE_RENAMES.get(key)
        if key == "poetic justice" and pd.notna(year) and int(year) == 1972:
            new_title = "Hapax Legomena II: Poetic Justice"
        if key == "critical mass" and pd.notna(year) and int(year) != 1971:
            new_title = None
        if key == "arabic" and (pd.isna(year) or int(year) != 1982):
            new_title = None
        if new_title:
            row = row.copy()
            row["title"] = new_title
        # Standardize spaced dot sequences while preserving the rest of the title.
        row = row.copy()
        row["title"] = re.sub(r"\.\s+\.\s+\.", "...", str(row["title"]))
        renamed.append(row)
    return pd.DataFrame(renamed).reset_index(drop=True)


def build_films(app: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct film. Main-table rows win for spine, country and title."""
    app = app.copy()

    def make_key(r) -> str:
        wt = r["wiki_title"]
        if r["title"] == "Small Axe":
            return "small axe|2020|steve mcqueen"
        if r["title"] in SPLIT_FILM_TITLES:
            return f"{norm(r['title'])}|{r['year']}"
        return wt if isinstance(wt, str) and wt else f"{norm(r['title'])}|{r['year']}"

    app["key"] = app.apply(make_key, axis=1)
    order = {"main": 0, "box_set": 1, "director_set": 2, "eclipse": 3}
    app["_rank"] = app["kind"].map(order)
    app = app.sort_values(["_rank", "table_index"], kind="stable").reset_index(drop=True)

    rows = []
    for _, g in app.groupby("key", sort=False):
        best = g.iloc[0]
        spines = g["film_spine"].dropna().astype(int)
        rows.append({
            "title": best["title"], "year": best["year"], "director": best["director"],
            "country": next((c for c in g["country"] if isinstance(c, str) and c), ""),
            "film_spine": int(spines.iloc[0]) if len(spines) else None,
            "in_main_table": bool((g["kind"] == "main").any()),
            "laserdisc_only": bool(g["laserdisc_only"].all()),
            "wiki_title": best["wiki_title"], "wiki_url": best["wiki_url"],
            "has_article": bool(g["has_article"].any()),
            "n_appearances": len(g),
            "sets": " | ".join(sorted({s for s in g["set_name"] if isinstance(s, str) and s})),
        })
    films = pd.DataFrame(rows)
    films["film_spine"] = films["film_spine"].astype("Int64")
    films["year"] = films["year"].astype("Int64")
    return films.sort_values(["in_main_table", "film_spine"], ascending=[False, True],
                             kind="stable").reset_index(drop=True)


def list_report(app: pd.DataFrame, films: pd.DataFrame) -> None:
    print(f"\nrows on page: {len(app)}   distinct films: {len(films)}")
    print("rows by table kind:\n", app["kind"].value_counts().to_string())
    main = app[app["kind"] == "main"]
    print(f"\nmain table: {len(main)} rows, {main['laserdisc_only'].sum()} LaserDisc-only, "
          f"{main['film_spine'].isna().sum()} without a spine number")
    dvd = main[~main["laserdisc_only"] & main["film_spine"].notna()]
    dup = dvd[dvd.duplicated("film_spine", keep=False)].sort_values("film_spine")
    if len(dup):
        print("\nDVD spine numbers used more than once (check the page):")
        print(dup[["film_spine", "title", "year"]].to_string(index=False))
    print(f"\nrows with no year: {app['year'].isna().sum()}   "
          f"films without a Wikipedia article link: {(~films['has_article']).sum()}")
    odd = sorted(set(app["country"]) - {""})
    print(f"distinct countries ({len(odd)}): eyeball for typos:\n", odd)


def stage_list(refresh: bool = False) -> None:
    print("== stage 1/3: Criterion film list ==")
    html, revid = fetch_list_html(wiki_session(), refresh)
    app = parse_tables(html)
    # Drop appearances from this box set before deduplication. If a film also
    # appears elsewhere in the Criterion list, that independent appearance stays.
    excluded_name = norm(EXCLUDED_BOX_SET)
    excluded = app["set_name"].map(lambda value: excluded_name in norm(value))
    excluded_count = int(excluded.sum())
    app = app[~excluded].reset_index(drop=True)
    print(f"excluded {excluded_count} film rows from {EXCLUDED_BOX_SET}")
    app = split_combined_entries(app)
    for col in ("film_spine", "set_spine", "year"):
        app[col] = app[col].astype("Int64")
    films = build_films(app)
    app["revid"] = revid
    films["revid"] = revid
    OUT.mkdir(parents=True, exist_ok=True)
    app.to_csv(OUT / "criterion_appearances.csv", index=False)
    films.to_csv(FILMS_CSV, index=False)
    list_report(app, films)
    print(f"saved to {OUT}/  (list revision {revid})")


# ==================================================================== STAGE 2: articles

def safe_name(title: str) -> str:
    """Filesystem-safe, collision-proof file stem for any article title."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")[:80]
    return f"{slug}_{hashlib.sha1(title.encode('utf-8')).hexdigest()[:8]}"


def page_props(parse: dict) -> dict:
    raw = parse.get("properties", [])
    if isinstance(raw, dict):
        return dict(raw)
    return {d["name"]: d.get("value", d.get("*", "")) for d in raw}


def sanity_checks(html: str, year, director) -> dict:
    """Cheap heuristics to catch links that landed on the wrong article."""
    head = html[:60000].lower()
    out = {"year_in_page": None, "director_in_page": None}
    if pd.notna(year):
        out["year_in_page"] = str(int(year)) in head
    if isinstance(director, str) and director.strip():
        first = director.split(",")[0].strip()
        if first:
            out["director_in_page"] = first.split()[-1].lower() in head
    return out


def fetch_article(session, title: str, year, director) -> dict:
    """Fetch one article; return its metadata (and write the HTML if it exists)."""
    meta = {"requested_title": title, "fetched_at": time.strftime("%Y-%m-%d")}
    try:
        data = get_json(session, WIKI_API, {
            "action": "parse", "page": title, "prop": "text|revid|properties",
            "redirects": 1, "format": "json", "formatversion": 2, "maxlag": 5})
    except (RuntimeError, requests.HTTPError) as e:
        return {**meta, "status": "error", "detail": str(e)[:200]}
    if data is None:
        return {**meta, "status": "error", "detail": "HTTP 404"}
    if "error" in data:
        code = data["error"].get("code", "")
        status = "missing" if code in ("missingtitle", "invalidtitle") else "error"
        return {**meta, "status": status, "detail": code}
    p = data["parse"]
    props = page_props(p)
    html = p["text"]
    (ARTICLES / f"{safe_name(title)}.html").write_text(html, encoding="utf-8")
    return {
        **meta,
        "status": "disambiguation" if "disambiguation" in props else "ok",
        "title": p["title"], "pageid": p.get("pageid"), "revid": p["revid"],
        "wikibase_item": props.get("wikibase_item"),
        "bytes": len(html.encode("utf-8")),
        **sanity_checks(html, year, director),
    }


def load_article_meta(title: str):
    path = ARTICLES / f"{safe_name(title)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def needs_download(title: str, retry_errors: bool, refresh: bool) -> bool:
    meta = load_article_meta(title)
    done = meta is not None and (meta["status"] in FINAL_STATUSES
                                 or (meta["status"] == "error" and not retry_errors))
    return refresh or not done


def fetch_collection_page(session, retry_errors: bool = False, refresh: bool = False) -> None:
    """Download the page about the Criterion Collection. It is not a film, so it stays out of the manifest."""
    if not needs_download(COLLECTION_PAGE, retry_errors, refresh):
        return
    meta = fetch_article(session, COLLECTION_PAGE, None, None)
    save_cache(ARTICLES / f"{safe_name(COLLECTION_PAGE)}.json", meta)
    print(f"{COLLECTION_PAGE}: {meta['status']} {meta.get('detail', '')}".rstrip())


def stage_articles(limit=None, main_only=False, pause=1.0, retry_errors=False, refresh=False) -> None:
    print("== stage 2/3: Wikipedia articles ==")
    ARTICLES.mkdir(parents=True, exist_ok=True)
    films = pd.read_csv(FILMS_CSV)
    films = films[films["has_article"].astype(bool) & films["wiki_title"].notna()]
    if main_only:
        films = films[films["in_main_table"].astype(bool)]
    films = films.drop_duplicates("wiki_title").reset_index(drop=True)

    todo = [f for _, f in films.iterrows() if needs_download(f["wiki_title"], retry_errors, refresh)]
    if limit:
        todo = todo[:limit]
    print(f"{len(films)} articles listed, {len(todo)} to download "
          f"(about {len(todo) * (pause + 0.5) / 60:.0f} min at this pace)")

    session = wiki_session()
    fetch_collection_page(session, retry_errors, refresh)
    start = time.time()
    try:
        for i, f in enumerate(todo, 1):
            meta = fetch_article(session, f["wiki_title"], f.get("year"), f.get("director"))
            save_cache(ARTICLES / f"{safe_name(f['wiki_title'])}.json", meta)
            if meta["status"] != "ok":
                print(f"  [{meta['status']}] {f['wiki_title']} {meta.get('detail', '')}")
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} done, {(time.time() - start) / 60:.1f} min elapsed")
            time.sleep(pause)
    except KeyboardInterrupt:
        print("\nstopped: rerun the same command to continue where you left off")

    rows = []
    for _, f in films.iterrows():
        meta = load_article_meta(f["wiki_title"]) or {"status": "not_fetched"}
        rows.append({"film_title": f["title"], "film_year": f["year"],
                     "film_spine": f.get("film_spine"), **meta})
    man = pd.DataFrame(rows)
    man.to_csv(MANIFEST, index=False)
    print("status counts:\n", man["status"].value_counts().to_string())
    for col in ("year_in_page", "director_in_page"):
        if col in man:
            bad = man[man[col] == False]  # noqa: E712 (explicit False, not NaN)
            print(f"\n{col} is False for {len(bad)} articles (possible wrong article, check by hand):")
            print(bad[["film_title", "film_year", "title"]].head(20).to_string(index=False))
    print(f"manifest: {MANIFEST}")


# ==================================================================== STAGE 3: TMDB

def claim_value(entity: dict, prop: str):
    """First non-deprecated value of a Wikidata external-identifier claim."""
    for st in entity.get("claims", {}).get(prop, []):
        if st.get("rank") == "deprecated":
            continue
        v = st.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(v, str) and v:
            return v
    return None


def wikidata_ids(session, qids: list[str], cache: dict, pause: float) -> dict:
    """qid -> {"tmdb_id": str|None, "imdb_id": str|None}; 50 items per request.
    P4947 = TMDb movie ID, P345 = IMDb ID."""
    todo = [q for q in qids if q not in cache]
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        data = get_json(session, WIKIDATA_API, {
            "action": "wbgetentities", "ids": "|".join(batch), "props": "claims",
            "format": "json", "maxlag": 5})
        for q in batch:
            ent = (data or {}).get("entities", {}).get(q, {})
            cache[q] = {"tmdb_id": claim_value(ent, "P4947"), "imdb_id": claim_value(ent, "P345")}
        time.sleep(pause)
    return cache


def title_variants(title: str) -> list[str]:
    """Full title first, then versions without a subtitle ("X: Y", "X, or Y", "X - Y")."""
    out = [title]
    for sep in (r",\s*or\b", r":", r"\s+-\s+"):
        head = re.split(sep, title, maxsplit=1)[0].strip(" ,")
        if head:
            out.append(head)
    return list(dict.fromkeys(out))


def search_tmdb(session, title, director, pause):
    crit_dirs = {norm(p).split()[-1] for p in re.split(r",| and ", str(director or "")) if norm(p)}
    if not crit_dirs:
        return None, "title search skipped: no Criterion director was available for verification"
    saw_title_candidate = False
    for q in title_variants(title):
        params = {"query": q, "include_adult": "false", "language": "en-US"}
        for kind in ("movie", "tv"):
            data = get_json(session, f"{TMDB_API}/search/{kind}", params)
            time.sleep(pause)
            candidates = []
            for c in (data or {}).get("results", []):
                names = (c.get("title"), c.get("original_title"), c.get("name"), c.get("original_name"))
                ratio = max(difflib.SequenceMatcher(None, norm(title), norm(n or "")).ratio() for n in names)
                wanted_numbers = set(re.findall(r"\d+", norm(title)))
                candidate_numbers = set(re.findall(r"\d+", norm(" ".join(n or "" for n in names))))
                if wanted_numbers and candidate_numbers and wanted_numbers != candidate_numbers:
                    continue
                if ratio >= 0.7:
                    candidates.append((ratio, c["id"]))
            saw_title_candidate = saw_title_candidate or bool(candidates)
            for ratio, tmdb_id in sorted(candidates, reverse=True):
                if kind == "movie":
                    credits = get_json(session, f"{TMDB_API}/movie/{tmdb_id}/credits") or {}
                    names = [c.get("name", "") for c in credits.get("crew", [])
                             if c.get("job") == "Director"]
                else:
                    show = get_json(session, f"{TMDB_API}/tv/{tmdb_id}") or {}
                    names = [c.get("name", "") for c in show.get("created_by", [])]
                time.sleep(pause)
                tmdb_dirs = {norm(n).split()[-1] for n in names if norm(n)}
                if crit_dirs & tmdb_dirs:
                    return (tmdb_id, kind), None
    if saw_title_candidate:
        return None, "title candidates found, but none had a matching director/creator surname"
    return None, "no movie or TV title candidate met the 0.70 similarity threshold"


def resolve_id(session, row, wd: dict, pause):
    """Return (tmdb_id, method, kind, no_match_note) for a movie or TV entry."""
    if row.get("title") in MANUAL_TMDB_TV_IDS:
        return MANUAL_TMDB_TV_IDS[row["title"]], "manual", "tv", None
    if row.get("title") in MANUAL_TMDB_IDS:
        return MANUAL_TMDB_IDS[row["title"]], "manual", "movie", None
    qid = row.get("wikibase_item")
    skip_article_ids = row.get("title") in SPLIT_FILM_TITLES or row.get("title") == "Small Axe"
    # The Wikipedia article covers both parts and may expose only one shared
    # external ID, so identify each part independently by title and credits.
    ids = wd.get(qid) if isinstance(qid, str) and not skip_article_ids else None
    if ids and ids["tmdb_id"]:
        return int(ids["tmdb_id"]), "wikidata", "movie", None
    if ids and ids["imdb_id"]:
        data = get_json(session, f"{TMDB_API}/find/{ids['imdb_id']}", {"external_source": "imdb_id"})
        time.sleep(pause)
        results = data or {}
        res = results.get("movie_results", []) or results.get("tv_results", [])
        if res:
            kind = "tv" if results.get("tv_results") and not results.get("movie_results") else "movie"
            return res[0]["id"], "imdb_find", kind, None
    match, search_note = search_tmdb(session, row["title"], row.get("director"), pause)
    if match:
        tid, kind = match
        return tid, f"search_{kind}", kind, None
    id_note = "Wikidata had no usable TMDB/IMDb ID" if ids else "no Wikidata IDs were available"
    return None, None, "movie", f"no TMDB match: {id_note}; {search_note}"


def image_base(session, poster_size: str) -> str:
    """TMDB's image server base URL (from /configuration), checking the poster size exists."""
    path = RAW_TMDB / "configuration.json"
    cfg = load_cache(path)
    if not cfg:
        data = get_json(session, f"{TMDB_API}/configuration")
        cfg = (data or {}).get("images", {})
        if cfg:
            save_cache(path, cfg)
    sizes = cfg.get("poster_sizes")
    if sizes and poster_size not in sizes:
        raise SystemExit(f"--poster-size must be one of {sizes}")
    return cfg.get("secure_base_url", DEFAULT_IMAGE_BASE)


def fetch_details(session, tmdb_id: int, pause: float, kind="movie"):
    path = RAW_TMDB / f"{kind}_{tmdb_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = get_json(session, f"{TMDB_API}/{kind}/{tmdb_id}",
                    {"append_to_response": "credits,keywords", "language": "en-US"})
    time.sleep(pause)
    if data is not None:
        save_cache(path, data)
    return data


def flatten(d: dict, base: str, poster_size: str) -> dict:
    crew = d.get("credits", {}).get("crew", [])
    cast = d.get("credits", {}).get("cast", [])
    # TMDB uses `created_by` for TV series rather than a crew member whose
    # job is `Director`. Keep the same output field so confidence checks work
    # for either kind of credit.
    directors = [c["name"] for c in crew if c.get("job") == "Director"]
    directors.extend(c["name"] for c in d.get("created_by", []) if c.get("name"))
    directors = list(dict.fromkeys(directors))
    is_tv = "name" in d and "title" not in d
    release_date = d.get("first_air_date" if is_tv else "release_date")
    m = re.match(r"\d{4}", release_date or "")
    poster = d.get("poster_path")
    return {
        "tmdb_id": d["id"],
        "imdb_id": d.get("imdb_id"),
        "tmdb_title": d.get("name" if is_tv else "title"),
        "tmdb_original_title": d.get("original_name" if is_tv else "original_title"),
        "tmdb_release_date": release_date or None,
        "tmdb_year": int(m.group()) if m else None,
        "runtime": d.get("runtime") or (d.get("episode_run_time") or [None])[0],
        "genres": "|".join(g["name"] for g in d.get("genres", [])),
        "countries": "|".join(c.get("name", c.get("iso_3166_1", ""))
                                 for c in d.get("production_countries", [])) or
                     "|".join(d.get("origin_country", [])),
        "languages": "|".join(l.get("english_name") or l["name"] for l in d.get("spoken_languages", [])),
        "original_language": d.get("original_language"),
        "tagline": d.get("tagline") or None,
        "overview": d.get("overview") or None,
        "keywords": "|".join(k["name"] for k in d.get("keywords", {}).get("keywords",
                                                                      d.get("keywords", {}).get("results", []))),
        "tmdb_directors": "|".join(directors),
        "top_cast": "|".join(c["name"] for c in cast[:8]),
        "poster_path": poster,
        "poster_url": f"{base}{poster_size}{poster}" if poster else None,
        "tmdb_url": f"https://www.themoviedb.org/{'tv' if is_tv else 'movie'}/{d['id']}",
    }


def confidence(row: dict, method: str, tmdb: dict) -> tuple[str, str]:
    """('high'|'low', reason). Compares year and director with the Criterion list."""
    if method == "manual":                       # you set this ID by hand in id_map.json
        return "high", ""
    reasons = []
    ty, y = tmdb.get("tmdb_year"), row.get("year")
    if ty is None or pd.isna(y) or abs(ty - y) > 1:
        reasons.append(f"year {y} vs TMDB {ty}")
    crit_dirs = {norm(p).split()[-1] for p in re.split(r",| and ", str(row.get("director") or "")) if norm(p)}
    tmdb_dirs = {norm(p).split()[-1] for p in str(tmdb.get("tmdb_directors") or "").split("|") if norm(p)}
    if crit_dirs and tmdb_dirs and not (crit_dirs & tmdb_dirs):
        reasons.append(f"director {sorted(crit_dirs)} vs TMDB {sorted(tmdb_dirs)}")
    if method == "search" and not reasons and not (crit_dirs & tmdb_dirs):
        reasons.append("matched by search and director could not be confirmed")
    return ("low", "; ".join(reasons)) if reasons else ("high", "")


def stage_tmdb(limit=None, main_only=False, pause=0.3, poster_size="w500") -> None:
    print("== stage 3/3: TMDB metadata and posters ==")
    token = os.environ.get("TMDB_TOKEN")
    if not token:
        raise SystemExit("Set TMDB_TOKEN (your TMDB read access token) in .env or the environment.")
    RAW_TMDB.mkdir(parents=True, exist_ok=True)

    films = pd.read_csv(FILMS_CSV)
    if main_only:
        films = films[films["in_main_table"].astype(bool)].reset_index(drop=True)
    if MANIFEST.exists():
        man = pd.read_csv(MANIFEST)[["requested_title", "wikibase_item"]].drop_duplicates("requested_title")
        films = films.merge(man, left_on="wiki_title", right_on="requested_title", how="left")
        films = films.drop(columns="requested_title")
    else:
        films["wikibase_item"] = None
        print("articles_manifest.csv not found: skipping Wikidata IDs (title search only).")
    if limit:
        films = films.head(limit)

    wd_path = RAW_TMDB / "wikidata_ids.json"
    wd = wikidata_ids(wiki_session(), list(films["wikibase_item"].dropna().unique()),
                      load_cache(wd_path), pause)
    save_cache(wd_path, wd)

    tm = requests.Session()
    tm.headers.update({"Authorization": f"Bearer {token}", "accept": "application/json",
                       "User-Agent": USER_AGENT})
    base = image_base(tm, poster_size)

    map_path = RAW_TMDB / "id_map.json"
    id_map = load_cache(map_path)
    # On reruns, retry only films whose previous output had no match confidence.
    # Other cached IDs (including manual fixes) remain untouched.
    enriched_path = OUT / "criterion_films_enriched.csv"
    retry_keys = set()
    if enriched_path.exists():
        previous = pd.read_csv(enriched_path)
        if {"title", "year", "match_confidence"}.issubset(previous.columns):
            unmatched = previous[previous["match_confidence"] == "none"]
            retry_keys = {
                f"{r.title}|{int(r.year) if pd.notna(r.year) else ''}"
                for r in unmatched.itertuples()
            }
    rows = []
    for i, (_, f) in enumerate(films.iterrows(), 1):
        key = f"{f['title']}|{int(f['year']) if pd.notna(f['year']) else ''}"
        force_manual = f["title"] in MANUAL_TMDB_IDS or f["title"] in MANUAL_TMDB_TV_IDS
        if key not in id_map or key in retry_keys or force_manual:
            tid, method, kind, no_match_note = resolve_id(tm, f, wd, pause)
            id_map[key] = {"tmdb_id": tid, "method": method, "kind": kind,
                           "no_match_note": no_match_note}
            save_cache(map_path, id_map)
        tid, method = id_map[key]["tmdb_id"], id_map[key]["method"]
        kind = id_map[key].get("kind", "tv" if method == "search_tv" else "movie")
        rec = f.to_dict()
        details = fetch_details(tm, tid, pause, kind) if tid else None
        if details:
            flat = flatten(details, base, poster_size)
            conf, why = confidence(rec, method, flat)
            rec.update(flat, match_method=method, match_confidence=conf, match_note=why)
        else:
            rec.update(match_method=None, match_confidence="none",
                       match_note=id_map[key].get("no_match_note") or
                       "no TMDB match: cached miss; rerun to retry lookup")
        rows.append(rec)
        if i % 50 == 0:
            print(f"{i}/{len(films)} films processed")

    out = pd.DataFrame(rows)
    for col in ("tmdb_id", "tmdb_year", "runtime"):
        out[col] = out[col].astype("Int64")
    out.to_csv(OUT / "criterion_films_enriched.csv", index=False)
    review = out[out["match_confidence"] != "high"][
        ["title", "year", "director", "film_spine", "match_method", "match_confidence",
         "match_note", "tmdb_id", "tmdb_title", "tmdb_year", "tmdb_directors", "tmdb_url"]]
    review.to_csv(OUT / "tmdb_review.csv", index=False)

    print(f"{len(out)} films")
    print("match method:\n", out["match_method"].fillna("none").value_counts().to_string())
    print("confidence:\n", out["match_confidence"].value_counts().to_string())
    matched = out[out["tmdb_id"].notna()]
    print(f"\nof {len(matched)} matched: missing runtime {matched['runtime'].isna().sum()}, "
          f"no genres {(matched['genres'] == '').sum()}, no overview {matched['overview'].isna().sum()}, "
          f"no poster {matched['poster_url'].isna().sum()}")
    print(f"saved {OUT / 'criterion_films_enriched.csv'}; review file {OUT / 'tmdb_review.csv'} "
          f"({len(review)} rows)")


# ==================================================================== command line

def main() -> None:
    ap = argparse.ArgumentParser(description="Download all data for the Criterion cineclub guide.")
    ap.add_argument("--stages", nargs="+", choices=["list", "articles", "tmdb"],
                    default=["list", "articles", "tmdb"], help="stages to run (default: all, in order)")
    ap.add_argument("--limit", type=int, help="only process the first N films in the articles and tmdb stages")
    ap.add_argument("--main-only", action="store_true",
                    help="only the main spine-numbered table (the enriched file then has only those films)")
    ap.add_argument("--pause", type=float, default=0.05, help="seconds between Wikipedia requests")
    ap.add_argument("--tmdb-pause", type=float, default=0.05, help="seconds between TMDB/Wikidata requests")
    ap.add_argument("--poster-size", default="w780", help="TMDB poster width: w92 w154 w185 w342 w500 w780 original")
    ap.add_argument("--retry-errors", action="store_true", help="retry articles whose last attempt failed")
    ap.add_argument("--refresh", action="store_true", help="re-download the list page and all articles")
    args = ap.parse_args()

    if "list" in args.stages:
        stage_list(refresh=args.refresh)
    if "articles" in args.stages:
        stage_articles(args.limit, args.main_only, args.pause, args.retry_errors, args.refresh)
    if "tmdb" in args.stages:
        stage_tmdb(args.limit, args.main_only, args.tmdb_pause, args.poster_size)
    print("\ndone.")


if __name__ == "__main__":
    main()
