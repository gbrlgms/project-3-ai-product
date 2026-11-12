"""Evaluation helpers for the Criterion cineclub guide (LLM track).

Put this file in src/ (or merge it into src/functions.py). It works with
evaluation_set.csv, which you freeze BEFORE building the system.

Expected catalogue columns (one row per film, built from the Wikipedia list
plus TMDB): spine, title, director, year, genres, countries, runtime.
`genres` and `countries` may be lists or "|"-separated strings.

Both the baseline (LLM alone) and your system should return an `Answer`.
Ask the model for JSON with `picks` (film titles) and `sources`, so nothing
has to be parsed out of prose.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Answer:
    text: str = ""
    picks: list[str] = field(default_factory=list)    # film titles recommended or listed
    sources: list[str] = field(default_factory=list)  # e.g. "A:Seven Samurai#Plot", "L:spine=2"
    refused: bool = False                             # system said "I don't know"
    cost_usd: float = 0.0


# ---------------------------------------------------------------- helpers

def norm(s) -> str:
    """Lowercase, strip accents and punctuation, drop a leading article."""
    s = str(s).replace("’", "'")     # a curly apostrophe would otherwise vanish and join the words around it
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"^(the|a|an) ", "", s)


def _as_list(x) -> list[str]:
    if isinstance(x, (list, tuple, set)):
        return [str(i) for i in x]
    if pd.isna(x) or x == "":
        return []
    return [i.strip() for i in str(x).split("|")]


EXTRA_ALIASES = {        # real alternate titles missing from the TMDB title fields and the article title
    "Mulholland Drive": ["Mulholland Dr."],
    "Invention for Destruction": ["The Fabulous World of Jules Verne"],
}


def prepare_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    cat = df.copy()
    cat["title_norm"] = cat["title"].map(norm)
    cat["genres"] = cat["genres"].map(_as_list)
    cat["countries"] = cat["countries"].map(_as_list)
    # Keywords and alternate titles from a low-confidence TMDB match cannot be trusted.
    low = cat["match_confidence"].eq("low")
    cat["keywords"] = [[] if is_low else [k.lower() for k in _as_list(v)] for is_low, v in zip(low, cat["keywords"])]
    cat["alt_norms"] = [_alt_titles(r, is_low) for (_, r), is_low in zip(cat.iterrows(), low)]
    return cat


def _alt_titles(row: pd.Series, low_confidence: bool) -> set[str]:
    """Other accepted names for a film: its TMDB titles and its Wikipedia article title."""
    names = [] if low_confidence else [row["tmdb_original_title"]]
    names += EXTRA_ALIASES.get(row["title"], [])
    if isinstance(row["wiki_title"], str):
        names.append(re.sub(r"\s*\([^)]*\)$", "", row["wiki_title"]))
    return {norm(n) for n in names if isinstance(n, str)} - {""}     # non-Latin titles normalize to ""


def lookup(cat: pd.DataFrame, title: str):
    """The film with this title. Catalogue titles come first, then alternate titles."""
    key = norm(title)
    hit = cat[cat["title_norm"] == key]
    if hit.empty:
        hit = cat[cat["alt_norms"].map(lambda names: key in names)]
    return hit.iloc[0] if len(hit) else None


def _canon(cat: pd.DataFrame, title: str) -> str:
    """The film's catalogue title (normalized), so that alternate titles compare as the same film."""
    row = lookup(cat, title)
    return row["title_norm"] if row is not None else norm(title)


def _is_tagged(cat: pd.DataFrame, title: str, keywords) -> bool:
    """Does the film carry any of these TMDB keywords?"""
    wanted = {k.lower() for k in keywords}
    rows = cat[cat["title_norm"] == _canon(cat, title)]        # a remake can share the title
    return any(wanted & set(tags) for tags in rows["keywords"])


def filter_catalogue(cat, director=None, year_min=None, year_max=None,
                     country=None, genre=None) -> pd.DataFrame:
    m = pd.Series(True, index=cat.index)
    if director:
        m &= cat["director"].map(lambda d: norm(director) in norm(d))
    if year_min is not None:
        m &= cat["year"] >= year_min
    if year_max is not None:
        m &= cat["year"] <= year_max
    if country:
        m &= cat["countries"].map(lambda c: norm(country) in [norm(i) for i in c])
    if genre:
        m &= cat["genres"].map(lambda g: norm(genre) in [norm(i) for i in g])
    return cat[m]


def numbers_in(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))]


# ---------------------------------------------------------------- checks
# Each check returns (passed: bool, detail: str).

def check_constraints(ans, cat, k=3, year_min=None, year_max=None, genre=None,
                      country=None, runtime_max=None, keywords=None):
    """First k picks must be in the catalogue and meet every constraint. `keywords` is not a
    constraint: tags are too sparse to fail a film on, so keyword_support reports it separately."""
    picks = ans.picks[:k]
    if len(picks) < k:
        return False, f"only {len(picks)} picks, need {k}"
    problems = []
    for p in picks:
        row = lookup(cat, p)
        if row is None:
            problems.append(f"{p}: not in catalogue")
            continue
        if year_min is not None and row["year"] < year_min:
            problems.append(f"{p}: year {row['year']} too early")
        if year_max is not None and row["year"] > year_max:
            problems.append(f"{p}: year {row['year']} too late")
        if genre and norm(genre) not in [norm(g) for g in row["genres"]]:
            problems.append(f"{p}: genres {row['genres']} lack {genre}")
        if country and norm(country) not in [norm(c) for c in row["countries"]]:
            problems.append(f"{p}: countries {row['countries']} lack {country}")
        if runtime_max is not None and not row["runtime"] <= runtime_max:
            problems.append(f"{p}: runtime {row['runtime']} over {runtime_max}")
    return not problems, "; ".join(problems) or "all picks meet constraints"


def check_seed_excluded(ans, cat, seed, k=3):
    picks = ans.picks[:k]
    if len(picks) < k:
        return False, f"only {len(picks)} picks, need {k}"
    if norm(seed) in [norm(p) for p in picks]:
        return False, "recommended the seed film itself"
    missing = [p for p in picks if lookup(cat, p) is None]
    return not missing, f"not in catalogue: {missing}" if missing else "ok"


def check_decades_differ(ans, cat, k=2):
    rows = [lookup(cat, p) for p in ans.picks[:k]]
    if len(rows) < k or any(r is None for r in rows):
        return False, "need two picks, both in the catalogue"
    decades = {int(r["year"]) // 10 for r in rows}
    return len(decades) == k, f"decades: {sorted(d * 10 for d in decades)}"


def expected_titles(cat, filter, extra_titles=(), optional_titles=()) -> set[str]:
    """The films a set question requires: the filter's matches, plus extras, minus the optional ones."""
    found = set(filter_catalogue(cat, **filter)["title_norm"]) | {_canon(cat, t) for t in extra_titles}
    return found - {_canon(cat, t) for t in optional_titles}


def list_coverage(question, titles, cat) -> float | None:
    """For a set question: the share of the required films among `titles`. None for other questions."""
    if question["auto_check"] != "set_from_catalog":
        return None
    params = json.loads(question["auto_params"])
    expected = expected_titles(cat, params["filter"], params.get("extra_titles", ()), params.get("optional_titles", ()))
    return len(expected & {_canon(cat, t) for t in titles}) / len(expected)


def check_set_from_catalogue(ans, cat, filter, min_recall=1.0, min_precision=1.0, extra_titles=(),
                             optional_titles=()):
    """Compare the picks with the films matching `filter`, plus `extra_titles`
    (films that qualify on facts the catalogue columns lack, such as filming locations).
    `optional_titles` are not required, and naming them is not counted as a mistake."""
    unknown = [t for t in (*extra_titles, *optional_titles) if lookup(cat, t) is None]
    if unknown:
        return False, f"titles not in the catalogue: {unknown}"
    optional = {_canon(cat, t) for t in optional_titles}
    expected = expected_titles(cat, filter, extra_titles, optional_titles)
    got = {_canon(cat, p) for p in ans.picks} - optional
    if not expected:
        return False, "expected set is empty: check the filter or the catalogue"
    hits = expected & got
    recall = len(hits) / len(expected)
    precision = len(hits) / len(got) if got else 0.0
    ok = recall >= min_recall and precision >= min_precision
    return ok, f"recall {recall:.2f}, precision {precision:.2f}, missing {sorted(expected - got)}"


def check_title(ans, cat, title):
    """The first pick must be this film. With no picks, the answer text must name it."""
    if ans.picks:
        return _canon(cat, ans.picks[0]) == _canon(cat, title), f"expected {title}, got {ans.picks[0]}"
    row, text = lookup(cat, title), norm(ans.text)
    names = {norm(title)} | (row["alt_norms"] if row is not None else set())
    return any(name in text for name in names), f"expected {title} in the answer text"


def check_keyword_list(ans, cat, keywords, min_tagged=2):
    """At least min_tagged of the named films must carry one of the keywords."""
    tagged = [p for p in ans.picks if _is_tagged(cat, p, keywords)]
    outside = [p for p in ans.picks if lookup(cat, p) is None]
    return len(tagged) >= min_tagged, f"{len(tagged)} tagged {tagged}; not in catalogue {outside}"


def check_count_from_catalogue(ans, cat, filter):
    expected = len(filter_catalogue(cat, **filter))
    return expected in numbers_in(ans.text), f"expected {expected}"


def check_number(ans, expected, tol=0):
    nums = numbers_in(ans.text)
    return any(abs(n - expected) <= tol for n in nums), f"expected {expected} +/- {tol}, found {nums}"


def check_contains_groups(ans, groups):
    """Every group needs at least one of its terms in the answer text."""
    text = norm(ans.text)
    missing = [g for g in groups if not any(norm(t) in text for t in g)]
    return not missing, f"missing one of: {missing}" if missing else "ok"


_REFUSAL = re.compile(
    r"(i don'?t know|i do not know|not in my sources|can'?t find|cannot find|"
    r"no information|not able to (find|answer)|don'?t have (that|this|any)|"
    r"not in the (criterion )?collection)", re.I)


def check_refusal(ans):
    refused = ans.refused or bool(_REFUSAL.search(ans.text))
    note = "refused" if refused else "answered instead of refusing"
    if refused and ans.sources:
        note += " (but still cited sources: check by hand)"
    return refused, note


def source_hit(ans, expected_sources: str):
    """True if any expected source (e.g. 'A:Burden of Dreams') appears in the cited sources."""
    if not isinstance(expected_sources, str):   # empty CSV cell reads back as NaN
        return None
    expected = [e for e in expected_sources.split("|") if e.strip()]
    if not expected:
        return None
    cited = [norm(s) for s in ans.sources]
    return any(norm(e) in c for e in expected for c in cited)


def keyword_support(eval_df: pd.DataFrame, answers: dict, cat: pd.DataFrame) -> pd.DataFrame:
    """For the mood questions: how many picks carry a matching TMDB keyword, next to the base rate
    (the share of eligible films with a tag), which is what picking at random would score."""
    rows = []
    tagged_films = cat[cat["keywords"].str.len() > 0]
    for _, q in eval_df[eval_df["auto_params"].fillna("").str.contains('"keywords"')].iterrows():
        params = json.loads(q["auto_params"])
        if q["auto_check"] != "constraints":
            continue
        keywords, k = params.pop("keywords"), params.pop("k", 3)
        picks = answers.get(q["id"], Answer()).picks[:k]
        eligible = filter_catalogue(tagged_films, **params)
        wanted = {w.lower() for w in keywords}
        n_eligible_tagged = sum(bool(wanted & set(tags)) for tags in eligible["keywords"])
        tagged = [p for p in picks if _is_tagged(cat, p, keywords)]
        rows.append({"id": q["id"], "question": q["question"], "picks": " | ".join(picks),
                     "tagged_picks": " | ".join(tagged), "picks_tagged": len(tagged), "k": k,
                     "share_tagged": round(len(tagged) / k, 2), "eligible_tagged": n_eligible_tagged,
                     "eligible": len(eligible), "base_rate": round(n_eligible_tagged / max(len(eligible), 1), 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- dispatcher

def run_checks(row, ans: Answer, cat: pd.DataFrame) -> dict:
    raw = row["auto_params"]
    params = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    name = row["auto_check"]
    answerable = int(row["answerable"]) == 1

    if name == "constraints":
        ok, detail = check_constraints(ans, cat, **params)
    elif name == "seed_excluded":
        ok, detail = check_seed_excluded(ans, cat, **params)
    elif name == "decades_differ":
        ok, detail = check_decades_differ(ans, cat, **params)
    elif name == "set_from_catalog":
        ok, detail = check_set_from_catalogue(ans, cat, **params)
    elif name == "title":
        ok, detail = check_title(ans, cat, **params)
    elif name == "keyword_list":
        ok, detail = check_keyword_list(ans, cat, **params)
    elif name == "count_from_catalog":
        ok, detail = check_count_from_catalogue(ans, cat, **params)
    elif name == "number":
        ok, detail = check_number(ans, **params)
    elif name == "contains_groups":
        ok, detail = check_contains_groups(ans, **params)
    elif name == "refusal":
        ok, detail = check_refusal(ans)
    else:
        ok, detail = None, f"no automatic check named '{name}': score by hand"

    refused = check_refusal(ans)[0]
    return {
        "auto_pass": ok,
        "auto_detail": detail,
        "source_hit": source_hit(ans, row["expected_sources"]) if answerable else None,
        "refused": refused,
        "false_refusal": bool(refused and answerable),
    }


def score_run(eval_df: pd.DataFrame, answers: dict, cat: pd.DataFrame, run_name: str) -> pd.DataFrame:
    """One row per question."""
    rows = []
    for _, q in eval_df.iterrows():
        ans = answers.get(q["id"], Answer())
        r = run_checks(q, ans, cat)
        rows.append({
            "run": run_name, "id": q["id"], "answer": ans.text,
            "picks": " | ".join(ans.picks), "sources": " | ".join(ans.sources),
            "cost_usd": ans.cost_usd, **r,
        })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, eval_df: pd.DataFrame) -> pd.Series:
    """Headline metrics for the report (answerable and unanswerable kept apart)."""
    df = results.merge(eval_df[["id", "answerable", "group"]], on="id")
    ans, unans = df[df["answerable"] == 1], df[df["answerable"] == 0]
    auto = ans["auto_pass"].dropna()
    hits = ans["source_hit"].dropna()
    return pd.Series({
        "n_answerable": len(ans),
        "auto_pass_rate": auto.astype(bool).mean() if len(auto) else None,
        "retrieval_hit_rate": hits.astype(bool).mean() if len(hits) else None,
        "false_refusal_rate": ans["false_refusal"].astype(bool).mean(),
        "refusal_rate_unanswerable": unans["refused"].astype(bool).mean() if len(unans) else None,
        "mean_cost_usd": df["cost_usd"].mean(),
    })
