"""Download the data behind the Criterion film guide. Run from the repo root:  python download_data.py

  list      Wikipedia's "List of films in the Criterion Collection" -> one row per distinct film
              data/processed/criterion_films.csv
  articles  every Wikipedia article linked from that list, plus the page about the Collection itself
              data/raw/wikipedia/articles/*.html + *.json, articles_manifest.csv
  tmdb      genres, runtime, countries, languages, overview, keywords, directors, cast, poster URL
              data/processed/criterion_films_enriched.csv   <- the table the pipeline reads

Everything is cached under data/raw/, so you can stop at any time and rerun to continue.

Setup
  pip install requests beautifulsoup4 pandas python-dotenv
  put TMDB_TOKEN=<your TMDB read access token> in .env

Films are matched to TMDB by their Wikidata item (TMDb ID, then IMDb ID), then by title search with
the director confirmed. To fix a wrong match, add "Title|year" to MANUAL_TMDB_IDS and rerun.

Licences and attribution
  Wikipedia text: CC BY-SA. Credit Wikipedia and keep each article's title + revision id.
  TMDB: non-commercial use; "This product uses the TMDB API but is not endorsed or certified by TMDB."
  Poster images are not downloaded, only their URLs on TMDB's image server.
"""
from __future__ import annotations

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
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ configuration

CONTACT = "https://github.com/gbrlgms/project-3-ai-product/"
USER_AGENT = f"CriterionCineclubGuide/0.1 (bootcamp project; {CONTACT})"

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
TMDB_API = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w780"
LIST_PAGE = "List_of_films_in_the_Criterion_Collection"
COLLECTION_PAGE = "The Criterion Collection"
PAUSE = 0.05     # seconds between requests

RAW_WIKI = Path("data/raw/wikipedia")
LIST_CACHE = RAW_WIKI / "criterion_list.json"
ARTICLES = RAW_WIKI / "articles"
MANIFEST = RAW_WIKI / "articles_manifest.csv"
RAW_TMDB = Path("data/raw/tmdb")
OUT = Path("data/processed")
FILMS_CSV = OUT / "criterion_films.csv"

COUNTRY_FIXES = {"Austrailia": "Australia", "United Kindom": "United Kingdom",
                 "United States": "United States of America"}      # the list's names, written as TMDB writes them
ORIGIN_COUNTRY_NAMES = {"US": "United States of America", "FR": "France"}   # TMDB country codes used when it has no production country
DIRECTOR_FIXES = {"Stan Brakage": "Stan Brakhage"}
WIKI_LINK_FIXES = {"Les Vacances de Monsieur Hulot": "Monsieur Hulot's Holiday"}    # a redirect that hid a duplicate film
FINAL_STATUSES = {"ok", "disambiguation", "missing"}   # articles not downloaded again on a rerun
EXCLUDED_BOX_SETS = ("Beastie Boys Video Anthology", "100 Years of Olympic Films: 1912-2012")
SPLIT_LIST_ENTRIES = {       # one row on the page -> several films
    "che": [("Che: Part One", 2008), ("Che: Part Two", 2008)],
    "0 4": [(f"Pan {n}", 1969) for n in range(5)],
    "697 700": [(f"Pan {n}", 1974) for n in range(697, 701)],
    "jimi plays monterey shake otis at monterey": [
        ("Jimi Plays Monterey", None),
        ("Shake! Otis at Monterey", None),
    ],
    "persian series 1 3": [
        ("Persian Series #1", 1999),
        ("Persian Series #2", 1999),
        ("Persian Series #3", 1999),
    ],
    "human condition": [
        ("The Human Condition I: No Greater Love", 1959),
        ("The Human Condition II: Road to Eternity", 1959),
        ("The Human Condition III: A Soldier's Prayer", 1961),
    ],
}
SPLIT_FILM_TITLES = {title for parts in SPLIT_LIST_ENTRIES.values() for title, _ in parts}
TITLE_RENAMES = {
    "sea urchiins": "Sea Urchins",
    "untitled for marilyn": "For Marilyn",
    "arabic": "Arabic Numeral Series",
    "critical mass": "Hapax Legomena III: Critical Mass",
    "l15 8 last": "Le 15/8",
    "report on the party and guests": "Party and Guests",
    "living magoroku the living magoroku": "The Living Magoroku",
    "i ll got to santiago": "I'm Going to Santiago",
    "visions in medication 4 d h lawrence": "Visions in Meditation #4 (D.H. Lawrence)",
    "gap toothed woman": "Gap-Toothed Women",
}
MANUAL_TMDB_IDS = {          # "Title|year", spelled exactly as in the catalogue
    # Split listings
    "The Human Condition I: No Greater Love|1959": 31217,
    "The Human Condition II: Road to Eternity|1959": 34528,
    "The Human Condition III: A Soldier's Prayer|1961": 34530,
    "Persian Series #1|1999": 385623,
    # Hollis Frampton
    "Hapax Legomena II: Poetic Justice|1972": 145497,
    "Hapax Legomena III: Critical Mass|1971": 151002,
    "Gloria!|1979": 147168,
    # Stan Brakhage
    "By Brakhage: An Anthology, Volume One|2003": 469501,
    "By Brakhage: An Anthology, Volume Two|2010": 469502,
    "The Dead|1960": 145033,
    "Desert|1976": 164590,
    "Visions in Meditation #4 (D.H. Lawrence)|1990": 164601,
    "“...” Reel Five|1998": 163360,
    # Jean Painlevé
    "Bluebeard|1938": 148243,
    "The Vampire|1945": 144694,
    # Films that picked up the record of a related film
    "Sea Urchins|1954": 117593,
    "Ivan the Terrible, Part II|1958": 43118,
    "Fårö Document 1979|1979": 119421,
    # Others
    "Martha Graham: Dance on Film|1959": 1404798,
    "Colors|1976": 201650,
    "Treasure Island|1969": 470474,
    "Gap-Toothed Women|1987": 154032,
    "Adventures on the New Frontier|1961": 379895,
}
MANUAL_TMDB_TV_IDS = {
    "Dekalog|1988": 42699,
    "Cartesius|1974": 239616,
    # All five Small Axe films are episodes of one TMDB series.
    "Mangrove|2020": 90705,
    "Lovers Rock|2020": 90705,
    "Red, White and Blue|2020": 90705,
    "Alex Wheatle|2020": 90705,
    "Education|2020": 90705,
}


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
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt * 5)))
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


def surnames(names) -> set[str]:
    """Last words of a comma- or 'and'-separated list of people."""
    return {norm(p).split()[-1] for p in re.split(r",| and ", str(names or "")) if norm(p)}


# ==================================================================== STAGE 1: list

def fetch_list_html(session: requests.Session) -> tuple[str, int]:
    """(html, revision id) of the list page. One API request, cached to disk."""
    if LIST_CACHE.exists():
        d = load_cache(LIST_CACHE)
        return d["html"], d["revid"]
    data = get_json(session, WIKI_API, {
        "action": "parse", "page": LIST_PAGE, "prop": "text|revid", "redirects": 1,
        "format": "json", "formatversion": 2, "maxlag": 5})
    if not data or "parse" not in data:
        raise SystemExit(f"Could not fetch the list page: {data}")
    p = data["parse"]
    LIST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    save_cache(LIST_CACHE, {"revid": p["revid"], "html": p["text"]})
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


def first_film_link(cell) -> str | None:
    """Wikipedia title of the first article link in a cell (None if there is no article)."""
    if cell is None:
        return None
    for a in cell.find_all("a", href=True):
        href = a["href"]
        if "redlink=1" in href or "new" in (a.get("class") or []):
            return None                        # red link: no article yet
        if not href.startswith("/wiki/"):
            continue                           # external or /w/index.php links
        title = unquote(href[len("/wiki/"):].split("#")[0]).replace("_", " ")
        if NAMESPACES.match(title.replace(" ", "_")):
            continue                           # File:, Category:, Help:... not a film
        return a.get("title") or title
    return None


def canon_header(h: str) -> str:
    h = re.sub(r"\[.*?\]", "", h.lower())
    h = re.sub(r"\s+", " ", h).strip()
    if h == "spine":
        return "spine"
    if "spine number" in h:
        return "film_spine"
    if "name of box set" in h or "eclipse" in h:
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
    return h


def to_int(raw: str):
    m = re.search(r"\d+", raw or "")
    return int(m.group()) if m else None


def parse_tables(html: str) -> pd.DataFrame:
    """One row per film row of each table on the list page."""
    soup = BeautifulSoup(html, "html.parser")
    records, t_index = [], 0
    for el in soup.find_all("table"):
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
            m_year = re.search(r"\d{4}", clean_text(cells.get("year")))
            country = clean_text(cells.get("country"))
            director = clean_text(cells.get("director"))
            records.append({
                "table_index": t_index,
                "kind": kind,
                "film_spine": to_int(clean_text(cells.get("spine" if kind == "main" else "film_spine"))),
                "set_spine": None if kind == "main" else to_int(clean_text(cells.get("spine"))),
                "set_name": clean_text(cells.get("set_name")),
                "title": re.sub(r"\s*[†‡*]+", "", raw_title).strip(),
                "year": int(m_year.group()) if m_year else None,
                "director": DIRECTOR_FIXES.get(director, director),
                "country": COUNTRY_FIXES.get(country, country),
                "laserdisc_only": "*" in raw_title,
                "wiki_title": WIKI_LINK_FIXES.get(first_film_link(title_cell), first_film_link(title_cell)),
            })
    return pd.DataFrame(records)


def corrected_title(row: pd.Series) -> str:
    key = norm(row["title"])
    title = TITLE_RENAMES.get(key, row["title"])
    if key == "poetic justice" and row["year"] == 1972:
        title = "Hapax Legomena II: Poetic Justice"
    return re.sub(r"\.\s+\.\s+\.", "...", str(title))


def split_combined_entries(app: pd.DataFrame) -> pd.DataFrame:
    """Expand combined listings into their individual films, then correct the titles."""
    combined = app["title"].map(lambda title: norm(title) in SPLIT_LIST_ENTRIES)
    additions = []
    for _, row in app[combined].iterrows():
        for title, year in SPLIT_LIST_ENTRIES[norm(row["title"])]:
            part = row.copy()
            part["title"] = title
            if year is not None:
                part["year"] = year
            additions.append(part)
    app = pd.concat([app[~combined], pd.DataFrame(additions)], ignore_index=True)
    app["title"] = app.apply(corrected_title, axis=1)
    return app


def build_films(app: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct film. Main-table rows win for spine, country and title."""
    app = app.copy()

    def make_key(r) -> str:
        wt = r["wiki_title"]
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
        # A film with no spine of its own takes the spine of its box set.
        spines = pd.concat([g["film_spine"].dropna(), g["set_spine"].dropna()]).astype(int)
        rows.append({
            "title": best["title"], "year": best["year"], "director": best["director"],
            "country": next((c for c in g["country"] if isinstance(c, str) and c), ""),
            "film_spine": int(spines.iloc[0]) if len(spines) else None,
            "in_main_table": bool((g["kind"] == "main").any()),
            "laserdisc_only": bool(g["laserdisc_only"].all()),
            "wiki_title": best["wiki_title"],
            "sets": " | ".join(sorted({s for s in g["set_name"] if isinstance(s, str) and s})),
        })
    films = pd.DataFrame(rows)
    films["film_spine"] = films["film_spine"].astype("Int64")
    films["year"] = films["year"].astype("Int64")
    return films.sort_values(["in_main_table", "film_spine"], ascending=[False, True],
                             kind="stable").reset_index(drop=True)


def stage_list() -> None:
    print("== stage 1/3: Criterion film list ==")
    html, revid = fetch_list_html(wiki_session())
    app = parse_tables(html)
    # Drop these box sets before deduplication: a film also listed elsewhere keeps that appearance.
    excluded = app["set_name"].map(lambda value: any(norm(name) in norm(value) for name in EXCLUDED_BOX_SETS))
    app = split_combined_entries(app[~excluded].reset_index(drop=True))
    for col in ("film_spine", "set_spine", "year"):
        app[col] = app[col].astype("Int64")
    films = build_films(app)
    films["revid"] = revid
    OUT.mkdir(parents=True, exist_ok=True)
    films.to_csv(FILMS_CSV, index=False)
    print(f"{len(films)} distinct films (list revision {revid})")


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
    meta = {"requested_title": title}
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
        **sanity_checks(html, year, director),
    }


def download_article(session, title: str, year=None, director=None) -> dict:
    """Metadata of one article, downloading it unless it is already on disk."""
    path = ARTICLES / f"{safe_name(title)}.json"
    meta = load_cache(path)
    if meta.get("status") not in FINAL_STATUSES:
        meta = fetch_article(session, title, year, director)
        save_cache(path, meta)
        time.sleep(PAUSE)
        if meta["status"] != "ok":
            print(f"  [{meta['status']}] {title} {meta.get('detail', '')}")
    return meta


def stage_articles() -> None:
    print("== stage 2/3: Wikipedia articles ==")
    ARTICLES.mkdir(parents=True, exist_ok=True)
    films = pd.read_csv(FILMS_CSV)
    films = films[films["wiki_title"].notna()].drop_duplicates("wiki_title")

    session = wiki_session()
    download_article(session, COLLECTION_PAGE)    # not a film, so it stays out of the manifest
    rows = []
    for i, (_, f) in enumerate(films.iterrows(), 1):
        rows.append(download_article(session, f["wiki_title"], f["year"], f["director"]))
        if i % 50 == 0:
            print(f"{i}/{len(films)} articles")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False)
    print("status counts:\n", manifest["status"].value_counts().to_string())


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


def wikidata_ids(session, qids: list[str], cache: dict) -> dict:
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
        time.sleep(PAUSE)
    return cache


def title_variants(title: str) -> list[str]:
    """Full title first, then versions without a subtitle ("X: Y", "X, or Y", "X - Y")."""
    out = [title]
    for sep in (r",\s*or\b", r":", r"\s+-\s+"):
        head = re.split(sep, title, maxsplit=1)[0].strip(" ,")
        if head:
            out.append(head)
    return list(dict.fromkeys(out))


def search_tmdb(session, title, director):
    """(tmdb_id, kind) of the closest title whose director or creator matches, else None."""
    crit_dirs = surnames(director)
    wanted_numbers = set(re.findall(r"\d+", norm(title)))
    for q in title_variants(title):
        params = {"query": q, "include_adult": "false", "language": "en-US"}
        for kind in ("movie", "tv"):
            data = get_json(session, f"{TMDB_API}/search/{kind}", params)
            time.sleep(PAUSE)
            candidates = []
            for c in (data or {}).get("results", []):
                names = (c.get("title"), c.get("original_title"), c.get("name"), c.get("original_name"))
                ratio = max(difflib.SequenceMatcher(None, norm(title), norm(n or "")).ratio() for n in names)
                candidate_numbers = set(re.findall(r"\d+", norm(" ".join(n or "" for n in names))))
                if wanted_numbers and candidate_numbers and wanted_numbers != candidate_numbers:
                    continue
                if ratio >= 0.7:
                    candidates.append((ratio, c["id"]))
            for ratio, tmdb_id in sorted(candidates, reverse=True):
                if kind == "movie":
                    credits = get_json(session, f"{TMDB_API}/movie/{tmdb_id}/credits") or {}
                    people = [c.get("name", "") for c in credits.get("crew", []) if c.get("job") == "Director"]
                else:
                    show = get_json(session, f"{TMDB_API}/tv/{tmdb_id}") or {}
                    people = [c.get("name", "") for c in show.get("created_by", [])]
                time.sleep(PAUSE)
                if crit_dirs & {norm(n).split()[-1] for n in people if norm(n)}:
                    return tmdb_id, kind
    return None


def film_key(row) -> str:
    return f"{row['title']}|{int(row['year']) if pd.notna(row['year']) else ''}"


def resolve_id(session, row, wd: dict):
    """(tmdb_id, method, kind) of a film or series; tmdb_id is None when nothing matches."""
    key = film_key(row)
    if key in MANUAL_TMDB_TV_IDS:
        return MANUAL_TMDB_TV_IDS[key], "manual", "tv"
    if key in MANUAL_TMDB_IDS:
        return MANUAL_TMDB_IDS[key], "manual", "movie"
    qid = row["wikibase_item"]
    # An article covering several parts may expose one shared ID, so each part is matched by title.
    shared_article = row["title"] in SPLIT_FILM_TITLES
    ids = wd.get(qid) if isinstance(qid, str) and not shared_article else None
    if ids and ids["tmdb_id"]:
        return int(ids["tmdb_id"]), "wikidata", "movie"
    if ids and ids["imdb_id"]:
        results = get_json(session, f"{TMDB_API}/find/{ids['imdb_id']}", {"external_source": "imdb_id"}) or {}
        time.sleep(PAUSE)
        found = results.get("movie_results", []) or results.get("tv_results", [])
        if found:
            kind = "tv" if results.get("tv_results") and not results.get("movie_results") else "movie"
            return found[0]["id"], "imdb_find", kind
    match = search_tmdb(session, row["title"], row["director"])
    if match:
        return match[0], f"search_{match[1]}", match[1]
    return None, None, "movie"


def fetch_details(session, tmdb_id: int, kind: str):
    path = RAW_TMDB / f"{kind}_{tmdb_id}.json"
    if path.exists():
        return load_cache(path)
    data = get_json(session, f"{TMDB_API}/{kind}/{tmdb_id}",
                    {"append_to_response": "credits,keywords", "language": "en-US"})
    time.sleep(PAUSE)
    if data is not None:
        save_cache(path, data)
    return data


def flatten(d: dict) -> dict:
    crew = d.get("credits", {}).get("crew", [])
    cast = d.get("credits", {}).get("cast", [])
    # TV series list `created_by` instead of a crew member whose job is Director.
    directors = [c["name"] for c in crew if c.get("job") == "Director"]
    directors.extend(c["name"] for c in d.get("created_by", []) if c.get("name"))
    directors = list(dict.fromkeys(directors))
    is_tv = "name" in d and "title" not in d
    m = re.match(r"\d{4}", d.get("first_air_date" if is_tv else "release_date") or "")
    poster = d.get("poster_path")
    keywords = d.get("keywords", {})
    return {
        "tmdb_id": d["id"],
        "tmdb_original_title": d.get("original_name" if is_tv else "original_title"),
        "tmdb_year": int(m.group()) if m else None,
        "runtime": d.get("runtime") or (d.get("episode_run_time") or [None])[0],
        "genres": "|".join(g["name"] for g in d.get("genres", [])),
        "countries": "|".join(c.get("name", c.get("iso_3166_1", "")) for c in d.get("production_countries", []))
                     or "|".join(ORIGIN_COUNTRY_NAMES.get(c, c) for c in d.get("origin_country", [])),
        "languages": "|".join(l.get("english_name") or l["name"] for l in d.get("spoken_languages", [])),
        "original_language": d.get("original_language"),
        "tagline": d.get("tagline") or None,
        "overview": d.get("overview") or None,
        "keywords": "|".join(k["name"] for k in keywords.get("keywords", keywords.get("results", []))),
        "tmdb_directors": "|".join(directors),
        "top_cast": "|".join(c["name"] for c in cast[:8]),
        "poster_url": f"{POSTER_BASE}{poster}" if poster else None,
        "tmdb_url": f"https://www.themoviedb.org/{'tv' if is_tv else 'movie'}/{d['id']}",
    }


def confidence(row: dict, method: str, tmdb: dict) -> str:
    """'high' or 'low': compares year and director with the Criterion list."""
    if method == "manual":
        return "high"
    tmdb_year, year = tmdb["tmdb_year"], row["year"]
    if tmdb_year is None or pd.isna(year) or abs(tmdb_year - year) > 1:
        return "low"
    crit_dirs = surnames(row["director"])
    tmdb_dirs = {norm(p).split()[-1] for p in tmdb["tmdb_directors"].split("|") if norm(p)}
    return "low" if crit_dirs and tmdb_dirs and not crit_dirs & tmdb_dirs else "high"


def stage_tmdb() -> None:
    print("== stage 3/3: TMDB metadata ==")
    token = os.environ.get("TMDB_TOKEN")
    if not token:
        raise SystemExit("Set TMDB_TOKEN (your TMDB read access token) in .env or the environment.")
    RAW_TMDB.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(MANIFEST)[["requested_title", "wikibase_item"]].drop_duplicates("requested_title")
    films = pd.read_csv(FILMS_CSV).merge(man, left_on="wiki_title", right_on="requested_title", how="left")
    films = films.drop(columns="requested_title")

    wd_path = RAW_TMDB / "wikidata_ids.json"
    wd = wikidata_ids(wiki_session(), list(films["wikibase_item"].dropna().unique()), load_cache(wd_path))
    save_cache(wd_path, wd)

    tm = requests.Session()
    tm.headers.update({"Authorization": f"Bearer {token}", "accept": "application/json",
                       "User-Agent": USER_AGENT})

    map_path = RAW_TMDB / "id_map.json"
    id_map = load_cache(map_path)
    rows = []
    for i, (_, f) in enumerate(films.iterrows(), 1):
        key = film_key(f)
        if key not in id_map or key in MANUAL_TMDB_IDS or key in MANUAL_TMDB_TV_IDS:      # manual IDs beat the cache
            tid, method, kind = resolve_id(tm, f, wd)
            id_map[key] = {"tmdb_id": tid, "method": method, "kind": kind}
            save_cache(map_path, id_map)
        match = id_map[key]
        rec = f.to_dict()
        details = fetch_details(tm, match["tmdb_id"], match["kind"]) if match["tmdb_id"] else None
        if details:
            flat = flatten(details)
            rec.update(flat, match_confidence=confidence(rec, match["method"], flat))
        else:
            rec.update(match_confidence="none")
        rows.append(rec)
        if i % 50 == 0:
            print(f"{i}/{len(films)} films")

    out = pd.DataFrame(rows)
    # TMDB's countries are not trusted on a doubtful match, so those films keep the Criterion list's country.
    doubtful = out["match_confidence"].ne("high")
    out.loc[doubtful, "countries"] = out.loc[doubtful, "country"]
    out = out.drop(columns=["wikibase_item", "tmdb_year", "country"])
    for col in ("tmdb_id", "runtime"):
        out[col] = out[col].astype("Int64")
    out.to_csv(OUT / "criterion_films_enriched.csv", index=False)
    print("match confidence:\n", out["match_confidence"].value_counts().to_string())


if __name__ == "__main__":
    stage_list()
    stage_articles()
    stage_tmdb()
    print("\ndone.")
