"""Criterion Guide: film recommendations from the Criterion Collection, with sources.
Run it from the repo root:

    streamlit run app/streamlit_app.py

Pages (in the top bar): the guide (a chatbot), the catalog, and search. Each film has its own page,
reached from the links in answers, in the catalog and in search results.
The OpenAI key is read from .env, or from .streamlit/secrets.toml, never from the code.
"""

import datetime
import html
import os
import sys
import threading
import time

import streamlit as st

sys.path.append(".")                       # so `src` is importable when you run from the repo root
from src.functions import *                # noqa: E402,F401  the same functions as the notebooks
from src.functions import _embed, _keyword_search  # noqa: E402  used once to warm up the pipeline

PRODUCT = "Criterion Guide"
EXAMPLES = ["Suggest a gentle Japanese drama from the 1950s, under two hours.",
            "What are some German Expressionism films in the Collection?",
            "Recommend a 1970s American horror film.",
            "Name a few films whose plot involves a train.",
            "I loved The Godfather. What else should I watch?",
            "What was the production of Boyhood like?",
            ]
GRID_COLUMNS, GRID_PAGE, LIST_PAGE = 5, 50, 100
SESSION_LIMIT = 20       # questions per visit
DAILY_LIMIT = 500        # questions per day for the whole server, about $0.50 at $0.001 a question
TMDB_NOTICE = "This product uses the TMDB API but is not endorsed or certified by TMDB."

st.set_page_config(page_title=PRODUCT, page_icon="🎞️", layout="wide")

# Colors and the body font are set in .streamlit/config.toml. These styles add the title font and the gold labels.
# Each licensed font comes first; until its file is in app/static/fonts/, a free Google font stands in.
st.html("""<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@800&family=Archivo+Narrow:wght@600&display=swap');
@font-face { font-family: 'Extenda 30 Deca'; src: url('app/static/fonts/Extenda30Deca.ttf'); }
button, button p { font-family: 'ITC Franklin Gothic LT Condensed', 'Archivo Narrow', sans-serif; }   /* body text is Book Compressed */
h1.brand { font-family: 'Extenda 30 Deca', 'Big Shoulders Display', sans-serif !important; font-weight: normal !important; font-size: min(12.5rem, 22vw) !important;
         line-height: 1; letter-spacing: -0.005em; text-transform: uppercase; color: #ffffff; margin: 0 0 0.5rem; }
p.eyebrow { font-family: 'ITC Franklin Gothic LT Condensed', 'Archivo Narrow', sans-serif; font-weight: 600;
           color: #b4841e; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.95rem !important; margin: 0.75rem 0 0.25rem; }
[data-testid="stPageLink"] p { white-space: normal; }     /* long titles wrap instead of being cut off */
[data-testid="stImage"] img { aspect-ratio: 2 / 3; object-fit: cover; border-radius: 4px; }   /* every poster the same shape */
/* Body text scale: change the "* 1" factors below. Headings, the title, the tagline and the gold labels keep their own sizes. */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { font-size: calc(1rem * 1.05); }
[data-testid="stTopNavLink"] p, [data-testid="stPageLink"] p, [data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: calc(0.875rem * 1.1); }
/* Placeholders in italic. A multiselect draws its placeholder as a plain div, the only one in the box with no children. */
[data-testid="stTextInput"] input::placeholder, [data-testid="stChatInput"] textarea::placeholder { font-style: italic; }
[data-testid="stMultiSelect"] [data-baseweb="select"] div:not(:has(*)):not([data-baseweb="tag"] *) { font-style: italic; }
[class*="st-key-filter_"] [data-testid="stWidgetLabel"] p { font-size: calc(0.875rem * 1.5) !important; }   /* 1.5 times the usual label */
a:hover, a:hover p, a:hover span { color: #b4841e !important; }     /* clickable text turns gold on hover */
p.row-number { width: 2.5em; margin: 0; text-align: right; font-variant-numeric: tabular-nums;   /* list view: the film's number */
               color: rgba(255, 255, 255, 0.5); font-size: calc(0.875rem * 1) !important; }
.stElementContainer:has(p.row-number) * { margin-bottom: 0 !important; }   /* so the number centers on its title */
/* Example questions: the two buttons in a row share one height, even when one wraps. */
.st-key-examples [data-testid="stHorizontalBlock"] { align-items: stretch; }
.st-key-examples [data-testid="stColumn"] { align-self: stretch !important; margin: 0 !important; }
.st-key-examples [data-testid="stColumn"] > div, .st-key-examples [data-testid="stColumn"] .stElementContainer,
.st-key-examples .stButton, .st-key-examples .stButton button { height: 100% !important; }
.st-key-examples button:not(:hover) p { color: rgba(255, 255, 255, 0.6); }   /* the tagline's gray, until hovered */
p.notice { background-color: rgba(180, 132, 30, 0.15); border: 1px solid rgba(180, 132, 30, 0.45);   /* a message box in shades of the gold */
           color: #e0b65a; border-radius: 8px; padding: 0.9rem 1rem; }
p.tagline { font-size: 22px !important; line-height: 1.3; color: rgba(255, 255, 255, 0.6); }
/* Catalog grid: each poster and title is one card. The film link is stretched over the whole card, so a click anywhere
   opens the film page in the same session, and the card lifts on hover. */
[class*="st-key-card_"] { position: relative; padding: 6px 6px 14px; border-radius: 8px;
                          transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease; }
[class*="st-key-card_"]:hover { transform: translateY(-8px) scale(1.03); background-color: #1f262d;
                                box-shadow: 0 14px 28px rgba(0, 0, 0, 0.55); }
[class*="st-key-card_"] .stElementContainer, [class*="st-key-card_"] [data-testid="stPageLink"],
[class*="st-key-card_"] [data-testid="stPageLink"] * { position: static !important; }
[class*="st-key-card_"] [data-testid="stPageLink"] a::after { content: ""; position: absolute; inset: 0; z-index: 2; }
[class*="st-key-card_"] [data-testid="stPageLink"] p { line-height: 1; }      /* half the default spacing between lines */
[class*="st-key-card_"] [data-testid="stPageLink"] a { background: transparent !important; }   /* the card lifts instead */
[class*="st-key-card_"] [data-testid="stElementToolbar"] { display: none; }
</style>""")

def secret(name: str) -> str | None:
    """A value from .streamlit/secrets.toml (or the Cloud app's secrets), else from the environment or .env."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:                      # no secrets file is fine
        pass
    return os.environ.get(name)


if secret("OPENAI_API_KEY"):               # a key in the secrets wins over .env
    os.environ["OPENAI_API_KEY"] = secret("OPENAI_API_KEY")


@st.cache_resource(show_spinner="Downloading the film index (first start only, a minute or two)...")
def app_files() -> list[str]:
    """Make sure the gitignored files the app reads are here. On a fresh clone or a Cloud server they are
    downloaded once from the private Hugging Face dataset named in the secrets, with the licensed fonts.
    Returns any required file still missing. The fonts are optional: without them the app uses free fallbacks."""
    def missing():
        return [path for path in ("models/chroma/chroma.sqlite3", "models/config.json",
                                  "data/processed/criterion_films_enriched.csv",
                                  "data/raw/wikipedia/articles_manifest.csv") if not (ROOT / path).exists()]
    no_fonts = not (ROOT / "app" / "static" / "fonts" / "Extenda30Deca.ttf").exists()
    if (missing() or no_fonts) and secret("HF_DATASET"):
        download_app_files(secret("HF_DATASET"), secret("HF_TOKEN"))
    return missing()


if app_files():
    st.error("The film index is missing. Build it with the notebooks (see the README), or set HF_DATASET and "
             "HF_TOKEN in the secrets so the app can download it.")
    st.stop()


@st.cache_resource
def daily_usage() -> dict:
    """Questions asked today on this server, shared by every visitor, so a public app has a ceiling on its cost."""
    return {"day": None, "count": 0, "lock": threading.Lock()}


def limit_reached() -> str | None:
    """Count one question, or return why it cannot be asked: the session or the server has used up its share."""
    if st.session_state.get("asked", 0) >= SESSION_LIMIT:
        return f"You have asked {SESSION_LIMIT} questions, the limit for one visit. Thank you for trying the guide!"
    usage = daily_usage()
    with usage["lock"]:
        today = datetime.date.today()
        if usage["day"] != today:
            usage.update(day=today, count=0)
        if usage["count"] >= DAILY_LIMIT:
            return "The guide has answered all the questions it can today. Please come back tomorrow."
        usage["count"] += 1
    st.session_state["asked"] = st.session_state.get("asked", 0) + 1
    return None


@st.cache_resource(show_spinner="Loading the film index (about 30 seconds, once)...")
def load_core():
    """The Chroma index, the catalog titles and the saved settings, loaded once per server.
    The embedding model and the keyword index are built now, so the first question is not slow."""
    collection, titles, settings = load_pipeline()
    _embed(["warm up"])
    _keyword_search("warm up", collection, 1, None)
    return collection, titles, settings


@st.cache_data(show_spinner="Loading the catalog...")
def catalog():
    return load_catalog()


@st.cache_resource
def poster_placeholder():
    """A gray card with the shape of a TMDB poster (2:3), for films without one, so the grid stays aligned."""
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (342, 513), "#2c3440")
    draw = ImageDraw.Draw(image)
    draw.rectangle([12, 12, 329, 500], outline="#445566", width=2)
    font = ImageFont.load_default(size=30)
    draw.text((171, 256), "No poster", fill="#9ab", font=font, anchor="mm")
    return image


def eyebrow(text: str) -> None:
    """A subtitle or highlight: gold, in capitals."""
    st.markdown(f'<p class="eyebrow">{html.escape(text)}</p>', unsafe_allow_html=True)


def year_text(year) -> str:
    return "" if pd.isna(year) else str(int(year))


def film_link(key: str, label: str | None = None) -> None:
    """A link to one film's page."""
    film = catalog().loc[key]
    st.page_link(FILM_PAGE, label=label or f"{film['title']} ({year_text(film['year'])})", query_params={"id": key})


# ------------------------------------------------------------------ the guide (chatbot)

def show_reply(message: dict) -> None:
    """One answer: the text, links to the films it names, its sources, and its cost and time."""
    st.markdown(message["answer"])
    if message["films"]:
        eyebrow("Films in this answer")
        for key in message["films"]:
            film_link(key)
    if message["sources"]:
        with st.expander(f"Sources ({len(message['sources'])})"):
            for source in message["sources"]:
                st.markdown(f"- [{source['title']}, {source['section']}]({source['url']}) · {source['source']}")
    searched = f"Searched for: {message['searched']} · " if message["searched"] else ""
    st.caption(f"{searched}{message['seconds']:.1f} s")


def ask(question: str, history: list[dict]) -> dict:
    """Answer the latest message in the light of the conversation so far: a follow-up is first rewritten
    into a question that stands on its own, then goes through the same pipeline as the evaluation."""
    collection, titles, settings = load_core()
    turns = [{"role": m["role"], "text": m["text"] if m["role"] == "user" else m["answer"]} for m in history]
    start = time.time()
    standalone, _, _ = standalone_question(question, turns)
    reply = answer_question(standalone, collection, catalog=titles, **settings)
    return {"role": "assistant", "answer": reply["answer"], "refused": reply["refused"],
            "films": films_in_reply(reply, catalog()), "sources": sources_in_reply(reply),
            "searched": standalone if norm(standalone) != norm(question) else None,
            "seconds": time.time() - start}


def guide_page() -> None:
    st.markdown(f'<h1 class="brand">{PRODUCT}</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Ask for a film from the Criterion Collection: a mood, a decade, a country, a director.<br>'
                "Every answer comes from Wikipedia and TMDB, with its sources, and it says when it doesn't know.</p>",
                unsafe_allow_html=True)
    load_core()
    messages = st.session_state.setdefault("messages", [])

    question = st.chat_input("What would you like to watch?", disabled=st.session_state.get("asked", 0) >= SESSION_LIMIT)
    examples = st.empty()
    if not messages and not question:
        with examples.container(key="examples"):
            eyebrow("Try one of these")
            for start in range(0, len(EXAMPLES), 2):          # two per row, each filling its column edge to edge
                for column, example in zip(st.columns(2), EXAMPLES[start:start + 2]):
                    if column.button(example, width="stretch"):
                        question = example
    if question:
        examples.empty()

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                show_reply(message)
            else:
                st.markdown(message["text"])

    if question and (reason := limit_reached()):
        st.markdown(f'<p class="notice">{html.escape(reason)}</p>', unsafe_allow_html=True)
        question = None

    if question:
        history = list(messages)
        messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the Collection..."):
                    message = ask(question, history)
            except Exception as error:     # a missing key or an API error should not crash the page
                messages.pop()
                st.error(f"The guide could not answer: {error}")
                return
            messages.append(message)
            show_reply(message)

    if messages and st.button("Start a new conversation"):
        st.session_state["messages"] = []
        st.rerun()


# ------------------------------------------------------------------ one film

def film_page() -> None:
    films = catalog()
    key = st.query_params.get("id")
    if key not in films.index:
        st.warning("That film is not in the catalog.")
        st.page_link(CATALOG_PAGE, label="Browse the catalog", icon=":material/grid_view:")
        return
    film = films.loc[key]
    _, poster, details, _ = st.columns([0.7, 0.9, 0.9, 0.5], gap="large")    # empty side columns center the content
    with poster:
        st.image(film["poster_url"] if isinstance(film["poster_url"], str) else poster_placeholder(), width="stretch")
    with details:
        st.title(film["title"])
        director = film["director"] if isinstance(film["director"], str) else "Director unknown"
        eyebrow(f"{director} · {year_text(film['year'])}")
        st.write(film["overview"] if isinstance(film["overview"], str) else "No overview available.")
        facts = [f"{int(film['runtime'])} min" if pd.notna(film["runtime"]) else None,
                 ", ".join(film["countries"]) or None, ", ".join(film["genres"]) or None]
        st.caption(" · ".join(fact for fact in facts if fact))
        links = [f"[TMDB]({film['tmdb_url']})" if isinstance(film["tmdb_url"], str) else None,
                 f"[Wikipedia]({film['wikipedia_url']})" if isinstance(film["wikipedia_url"], str) else None]
        if any(links):
            st.markdown(" · ".join(link for link in links if link))


# ------------------------------------------------------------------ the catalog

def decade_label(decade: int) -> str:
    return f"{decade}s"


def catalog_page() -> None:
    films = catalog()
    st.title("THE CATALOG")
    country_col, decade_col, genre_col = st.columns(3)
    countries = country_col.multiselect("Country", key="filter_country", options=sorted({c for values in films["countries"] for c in values}))
    decades = decade_col.multiselect("Decade", key="filter_decade", options=sorted(films["decade"].dropna().astype(int).unique()), format_func=decade_label)
    genres = genre_col.multiselect("Genre", key="filter_genre", options=sorted({g for values in films["genres"] for g in values}))
    shown = filter_catalog(films, countries, decades, genres)

    count_col, view_col = st.columns([3, 1])
    with view_col.container(horizontal_alignment="right"):
        view = st.segmented_control("View", ["Grid", "List"], default="Grid", required=True, label_visibility="collapsed")
    with count_col:
        eyebrow(f"{len(shown):,} of {len(films):,} films")
    if shown.empty:
        st.markdown('<p class="notice">No film matches these filters.</p>', unsafe_allow_html=True)
        return

    per_page = GRID_PAGE if view == "Grid" else LIST_PAGE
    pages = (len(shown) - 1) // per_page + 1
    if st.session_state.get("catalog_page", 1) > pages:     # the filters changed and the page no longer exists
        st.session_state["catalog_page"] = 1
    page = st.number_input(f"Page (of {pages})", min_value=1, max_value=pages, step=1, key="catalog_page", width=150) if pages > 1 else 1
    rows = shown.iloc[(page - 1) * per_page: page * per_page]

    if view == "Grid":
        for start in range(0, len(rows), GRID_COLUMNS):
            for column, (key, film) in zip(st.columns(GRID_COLUMNS), rows.iloc[start:start + GRID_COLUMNS].iterrows()):
                with column, st.container(key=f"card_{key}"):      # styled as one clickable card (see the styles above)
                    poster = film["poster_url"]
                    st.image(poster.replace("/w780/", "/w342/") if isinstance(poster, str) else poster_placeholder(),
                             width="stretch")
                    film_link(key)
    else:
        first = (page - 1) * per_page + 1                    # numbering continues across pages
        for number, (key, film) in enumerate(rows.iterrows(), first):
            director = film["director"] if isinstance(film["director"], str) else ""
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f'<p class="row-number">{number}</p>', unsafe_allow_html=True, width="content")
                film_link(key, f"{film['title']} ({year_text(film['year'])}) · {director}".rstrip(" ·"))


# ------------------------------------------------------------------ search

def search_page() -> None:
    films = catalog()
    st.title("SEARCH")
    query = st.text_input("Search by title or director...", placeholder="e.g. Kurosawa, or The Seventh Seal")
    if not query.strip():
        return
    found = search_films(films, query)
    eyebrow(f"{len(found):,} films found" + (", showing the first 100" if len(found) > 100 else ""))
    for key, film in found.head(100).iterrows():
        director = film["director"] if isinstance(film["director"], str) else ""
        film_link(key, f"{film['title']} ({year_text(film['year'])}) · {director}".rstrip(" ·"))


# ------------------------------------------------------------------ navigation

GUIDE_PAGE = st.Page(guide_page, title="Guide", icon=":material/chat:", url_path="guide", default=True)
CATALOG_PAGE = st.Page(catalog_page, title="Catalog", icon=":material/grid_view:", url_path="catalog")
SEARCH_PAGE = st.Page(search_page, title="Search", icon=":material/search:", url_path="search")
FILM_PAGE = st.Page(film_page, title="Film", url_path="film", visibility="hidden")

with st.sidebar:
    eyebrow("About")
    st.write("A guide to the 1,928 films of the Criterion Collection. Answers come only from Wikipedia "
             "(the Criterion list and each film's article) and TMDB, retrieved and cited for every answer.")
    st.write("Follow-up questions build on the conversation, which lasts until you start a new one or close the page.")
    st.write("It has a knowledge base up to November 2026 and no live information. "
             "How well a pick fits a mood is subjective and measured on TMDB keywords and Wikipedia plot descriptions.")
    st.caption(f" Developed by Gabriel Gomes da Silva, 2026. \n Wikipedia text: CC BY-SA 4.0. {TMDB_NOTICE}")

st.navigation([GUIDE_PAGE, CATALOG_PAGE, SEARCH_PAGE, FILM_PAGE], position="top").run()
