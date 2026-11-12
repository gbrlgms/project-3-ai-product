"""Section 5, exploring the texts: lengths, the most frequent words, how texts differ between
groups, and a reading list of hard cases.

Put these functions in src/functions.py (they use clean_text, detect_language, article_text
from the previous steps, which live in the same file there).  Run this file directly for a text
version of the whole section:
    python explore_texts.py                      # prints everything
    python explore_texts.py --save-figs reports  # also saves the figures

Your data has no star ratings, so the brief's comparisons map like this:
    ratings   -> decades (periods), genres and countries (the groups whose texts may differ)
    "3-star, sarcastic and rating-vs-text disagreeing reviews" -> the hard cases for THIS product:
        1. partial matches    films that fit a query by metadata OR by text, but not both
                              ("tense 1970s political thriller": a Thriller from 1975 about love,
                              or a Drama from 1976 whose plot is a political conspiracy)
        2. genre disagreement TMDB's genre labels vs the genre words in Wikipedia's first sentence
        3. thin texts         stubs, articles with no plot section, text that is not English
Read them: write down what you see and what it means for retrieval and for the evaluation set.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from data_quality import clean_text, detect_language
from first_look import article_text, load_tables, read_in_full

# ------------------------------------------------------------------ words

STOP = set("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your yours yourself
film films movie movies directed director starring released release one two first second three new his's
""".split())
_WORD = re.compile(r"[^\W\d_]{3,}")


def tokens(text: str, stop: set = STOP) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in stop]


# ------------------------------------------------------------------ the corpus

_PLOT = re.compile(r"^(plot|synopsis|premise|story|summary)", re.I)


def split_sections(text: str) -> tuple[str, dict[str, str]]:
    """(lead paragraph(s), {section heading: text}) from article_text() output, without the infobox."""
    body = "\n".join(l for l in text.split("\n") if not l.startswith("[infobox]"))
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    sections: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip()
        sections[key] = (sections.get(key, "") + " " + parts[i + 1]).strip()
    return parts[0].strip(), sections


def build_corpus(films: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per film with a downloaded article: cleaned text pieces, lengths and groups."""
    ok = set(manifest.loc[manifest["status"] == "ok", "requested_title"])
    rows = []
    for _, f in films[films["wiki_title"].isin(ok)].iterrows():
        try:
            text = clean_text(article_text(f["wiki_title"]))
        except FileNotFoundError:
            continue
        lead, sections = split_sections(text)
        plot = next((v for k, v in sections.items() if _PLOT.match(k)), "")
        body = " ".join([lead, *sections.values()]).strip()
        countries = f.get("countries")
        rows.append({
            "wiki_title": f["wiki_title"], "title": f["title"], "year": f["year"],
            "country": countries[0] if countries else None,
            "genres": f["genres"],
            "overview": clean_text(f.get("overview")),
            "lead": lead, "first_sentence": re.split(r"(?<=[.!?])\s+(?=[A-Z])", lead)[0] if lead else "",
            "plot": plot, "body": body,
            "n_words_body": len(body.split()), "n_words_plot": len(plot.split()),
            "n_words_overview": len(clean_text(f.get("overview")).split()),
        })
    df = pd.DataFrame(rows)
    df["decade"] = (pd.to_numeric(df["year"], errors="coerce") // 10 * 10).astype("Int64")
    return df


# ------------------------------------------------------------------ lengths

def length_summary(corpus: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Word-count summary: overall for the three text pieces, or the article body per group."""
    if by is None:
        return corpus[["n_words_body", "n_words_plot", "n_words_overview"]].describe(
            percentiles=[.1, .5, .9]).round(0)
    g = corpus.groupby(by)["n_words_body"]
    return pd.DataFrame({"n": g.size(), "median": g.median(), "p10": g.quantile(.1),
                         "p90": g.quantile(.9)}).round(0)


def plot_lengths(corpus: pd.DataFrame, path: str | None = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].hist(corpus["n_words_body"].clip(upper=corpus["n_words_body"].quantile(.99)), bins=40)
    ax[0].set(title="Article length (words, top 1% clipped)", xlabel="words", ylabel="films")
    plot_words = corpus.loc[corpus["n_words_plot"] > 0, "n_words_plot"]
    ax[1].hist(plot_words.clip(upper=plot_words.quantile(.99)), bins=40)
    ax[1].set(title="Plot section length (words)", xlabel="words")
    by = corpus.dropna(subset=["decade"]).groupby("decade")["n_words_body"].median()
    ax[2].bar(by.index.astype(str), by.values)
    ax[2].set(title="Median article length by decade", xlabel="decade", ylabel="words")
    ax[2].tick_params(axis="x", rotation=60)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------ frequent and distinctive words

def word_frequencies(texts, top: int = 30, stop: set = STOP) -> pd.DataFrame:
    """Most frequent words with their count and the share of documents that contain them."""
    texts = list(texts)
    counts, docs = Counter(), Counter()
    for t in texts:
        toks = tokens(t, stop)
        counts.update(toks)
        docs.update(set(toks))
    out = pd.DataFrame(counts.most_common(top), columns=["word", "count"])
    out["docs"] = out["word"].map(docs)
    out["doc_share"] = (out["docs"] / max(len(texts), 1)).round(2)
    return out


def plot_top_words(freq: pd.DataFrame, title: str = "Most frequent words", path: str | None = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(freq) + 1))
    ax.barh(freq["word"][::-1], freq["count"][::-1])
    ax.set(title=title, xlabel="occurrences")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def distinctive_words(corpus: pd.DataFrame, group_col: str, text_col: str = "plot", top: int = 8,
                      min_docs: int = 8, multi: bool = False, prior: float = 500, min_count: int = 5) -> pd.DataFrame:
    """Words that set each group apart from the rest (z-scored log-odds with an informative
    prior, Monroe et al. 2008). multi=True for list columns such as genres.
    Frequent words are NOT distinctive: this ranks by contrast, not by count."""
    data = corpus[[group_col, text_col]].copy()
    data = data[data[text_col].fillna("").str.len() > 0].dropna(subset=[group_col])
    doc_tokens = [Counter(tokens(t)) for t in data[text_col]]
    total = Counter()
    for c in doc_tokens:
        total.update(c)
    vocab = [w for w, n in total.items() if n >= min_count]
    if not vocab:
        return pd.DataFrame(columns=["group", "n_docs", "distinctive words"])
    index = {w: i for i, w in enumerate(vocab)}
    y_all = np.array([total[w] for w in vocab], dtype=float)
    n_all = y_all.sum()
    alpha = prior * y_all / n_all

    keys = data[group_col].map(lambda v: v if multi else [v])
    members: dict = {}
    for i, ks in enumerate(keys):
        for k in ks:
            members.setdefault(k, []).append(i)

    rows = []
    for g, idx in members.items():
        if len(idx) < min_docs:
            continue
        y_i = np.zeros(len(vocab))
        for i in idx:
            for w, n in doc_tokens[i].items():
                if w in index:
                    y_i[index[w]] += n
        n_i = y_i.sum()
        y_j, n_j = y_all - y_i, n_all - n_i
        delta = (np.log((y_i + alpha) / (n_i + prior - y_i - alpha))
                 - np.log((y_j + alpha) / (n_j + prior - y_j - alpha)))
        z = delta / np.sqrt(1 / (y_i + alpha) + 1 / (y_j + alpha))
        best = np.argsort(-z)[:top]
        rows.append({"group": g, "n_docs": len(idx),
                     "distinctive words": ", ".join(f"{vocab[b]}" for b in best if y_i[b] > 0)})
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


# ------------------------------------------------------------------ hard cases

POLITICAL = set("""political politics politician politicians government governments conspiracy corruption
assassination assassinate regime dictatorship dictator election elections senator president coup terrorism
terrorist terrorists revolution revolutionary junta fascist fascism communist minister ministry cia fbi
""".split())
TENSION = set("""thriller suspense tension tense paranoia paranoid surveillance murder murdered investigation
investigates investigator killed kidnapped hunted spy spies assassin menace fear threat threatened
""".split())


def partial_matches(corpus: pd.DataFrame, year_min: int = 1970, year_max: int = 1979,
                    genre: str = "Thriller", min_political: int = 2, min_tension: int = 2) -> pd.DataFrame:
    """Within the decade, which films does the METADATA call a `genre` and which does the TEXT call
    political and tense?  'metadata only' and 'text only' are the hard cases for retrieval."""
    df = corpus[corpus["year"].between(year_min, year_max)].copy()
    meta = df["genres"].map(lambda g: genre in g)
    cue_text = (df["lead"] + " " + df["plot"]).map(lambda t: Counter(tokens(t, set())))
    df["political_hits"] = cue_text.map(lambda c: sorted(w for w in c if w in POLITICAL))
    df["tension_hits"] = cue_text.map(lambda c: sorted(w for w in c if w in TENSION))
    text = (df["political_hits"].str.len() >= min_political) & (df["tension_hits"].str.len() >= min_tension)
    df["category"] = np.select([meta & text, meta & ~text, ~meta & text],
                               ["both", "metadata only", "text only"], default="neither")
    df["tmdb_genres"] = df["genres"]
    keep = df[df["category"] != "neither"].sort_values(["category", "year"])
    return keep[["category", "title", "year", "tmdb_genres", "political_hits", "tension_hits", "first_sentence"]]


GENRE_WORDS = {
    "comedy": "Comedy", "thriller": "Thriller", "drama": "Drama", "documentary": "Documentary",
    "horror": "Horror", "western": "Western", "crime": "Crime", "romance": "Romance",
    "romantic": "Romance", "war": "War", "mystery": "Mystery", "fantasy": "Fantasy",
    "animated": "Animation", "animation": "Animation", "science fiction": "Science Fiction",
    "musical": "Music", "adventure": "Adventure", "action": "Action",
}


def genre_text_disagreement(corpus: pd.DataFrame) -> pd.DataFrame:
    """Wikipedia's first sentence usually names the genre ("a 1974 American political thriller
    film"). Compare it with TMDB's genre labels: where do they disagree?"""
    rows = []
    for _, r in corpus.iterrows():
        sentence = r["first_sentence"].lower()
        wiki = {g for w, g in GENRE_WORDS.items() if re.search(rf"\b{re.escape(w)}\b", sentence)}
        tmdb = set(r["genres"])
        if not wiki or not tmdb:
            continue
        if wiki - tmdb or not (wiki & tmdb):
            rows.append({"title": r["title"], "year": r["year"], "kind": "no overlap" if not (wiki & tmdb)
                         else "text names a genre TMDB lacks", "wikipedia_says": ", ".join(sorted(wiki)),
                         "tmdb_says": ", ".join(sorted(tmdb)), "first_sentence": r["first_sentence"]})
    return pd.DataFrame(rows)


def thin_texts(corpus: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Shortest plots, articles with no plot section, and leads that are not in English."""
    df = corpus.copy()
    df["lead_language"] = df["lead"].map(lambda t: detect_language(t[:600]) if t else "unknown")
    df["problem"] = np.select(
        [df["lead_language"].isin(["en", "unknown"]) == False, df["n_words_plot"] == 0, df["n_words_plot"] < 80],
        ["not English", "no plot section", "very short plot"], default="")
    cols = ["problem", "title", "year", "n_words_body", "n_words_plot", "lead_language", "lead"]
    bad = df[df["problem"] != ""].sort_values("n_words_body")
    return bad[cols].head(n) if n else bad[cols]


# ------------------------------------------------------------------ the whole section

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-figs", metavar="DIR", help="also save the figures there")
    args = ap.parse_args()
    figs = Path(args.save_figs) if args.save_figs else None
    if figs:
        figs.mkdir(parents=True, exist_ok=True)

    films, manifest = load_tables()
    corpus = build_corpus(films, manifest)
    print(f"corpus: {len(corpus)} films with a downloaded article\n")

    print("### Lengths (words)")
    print(length_summary(corpus).to_string())
    for by in ("decade", "country"):
        print(f"\nmedian article length by {by} (groups with at least 5 films)")
        s = length_summary(corpus, by)
        print(s[s["n"] >= 5].to_string())
    plot_lengths(corpus, str(figs / "lengths.png") if figs else None)

    print("\n### Most frequent words (plot sections; stopwords removed)")
    freq = word_frequencies(corpus["plot"][corpus["plot"] != ""], top=25)
    print(freq.to_string(index=False))
    plot_top_words(freq, "Plot sections: most frequent words", str(figs / "top_words.png") if figs else None)

    print("\n### What sets groups apart (plot sections)")
    for col, multi in (("decade", False), ("genres", True), ("country", False)):
        print(f"\nby {col}:")
        print(distinctive_words(corpus, col, "plot", multi=multi).to_string(index=False))

    with pd.option_context("display.max_colwidth", 80, "display.width", 250):
        print("\n### Hard case 1: 'tense 1970s political thriller', metadata vs text")
        pm = partial_matches(corpus)
        print(pm["category"].value_counts().to_string())
        print(pm.head(12).to_string(index=False))

        print("\n### Hard case 2: TMDB genres vs the genre words in Wikipedia's first sentence")
        gd = genre_text_disagreement(corpus)
        print(f"{len(gd)} of {len(corpus)} films disagree")
        print(gd.head(12).to_string(index=False))

        print("\n### Hard case 3: thin texts")
        print(thin_texts(corpus).to_string(index=False))

    print("\n### Read a few of them in full")
    if len(pm):
        read_in_full(corpus[corpus["title"].isin(pm["title"].head(20))][["title", "year", "genres", "lead", "plot"]],
                     n=2, seed=0)


if __name__ == "__main__":
    main()
