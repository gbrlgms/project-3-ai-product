# Criterion Guide

*Film recommendations from the Criterion Collection, with sources you can check.*

Ironhack Data Science & Machine Learning, Week 9 final project · **LLM track** · Gabriel Gomes

**Slides:** [[Canva link]](https://canva.link/8ccrjlx2jo0kgvr)

---

## The user and the question

**Who it is for.** A film fan or cineclub organizer who is choosing what to watch or program next from the Criterion Collection, and who wants a recommendation they can trust, not a generic "best films" list.

**What they ask.** A request in plain language: a mood, a decade, a country, a director, a runtime. For example, *"Suggest a gentle Japanese drama from the 1950s, under two hours."*

**What they get.** About three films from the Collection, each with a reason, a link to its page in the app, and citations to its Wikipedia article and TMDB record. If the sources cannot answer, for example what is streaming tonight, the guide says "I don't know" instead of guessing.

**The decision it supports.** Which film or films to watch or screen, for example a double feature. The citations let the user check a pick before committing.

## The data

| Source | What it gives | Licence |
|---|---|---|
| Wikipedia's [List of films in the Criterion Collection](https://en.wikipedia.org/wiki/List_of_films_in_the_Criterion_Collection) | The catalog: about 1,920 distinct films, with title, director, year, country and spine number | CC BY-SA |
| The Wikipedia article of each film (about 1,760), plus the page on the Collection itself | Plot, production, release, reception and themes | CC BY-SA |
| [TMDB](https://www.themoviedb.org/), The Movie Database, a community-built film database with a public API | Genres, runtime, countries, overview, keywords, cast and poster URL | Non-commercial use |

**Unit of retrieval.** One chunk, from one section of one source, with the film's metadata attached (year, decade, director, country, genres, runtime) so it can be filtered and cited. There are 27,576 chunks in all.

**Traps found and handled.**

- **Spelling and matching.** The list contains spelling errors, and a film can appear more than once. Some films matched the wrong TMDB record, so some joins were checked and fixed by hand.
- **Inconsistent metadata.** Country names came in several forms ("US", "United States", "United States of America"), and these were unified.
- **Wrong articles.** Disambiguation pages, and articles that are probably about another film, are left out and listed in a report.
- **Spoilers.** Plot chunks are flagged as spoilers, so recommendations can lean on other sections.

This product uses the TMDB API but is not endorsed or certified by TMDB. Poster images are not downloaded, only their URLs on TMDB's servers. Wikipedia text is used under CC BY-SA, and each chunk keeps its article's title, revision and section. **The data is never committed**: anyone cloning the repo runs `download_data.py`.

## The metric and the baseline

**The evaluation set.** 30 questions written before the build (`data/evaluation_set.csv`). 25 can be answered from the data, and 5 cannot, for example streaming availability or a licence price. Each question has an automatic check: the films expected or allowed, the constraints a pick must meet, and the expected source.

**The metric.** The number of the 25 answerable questions that pass their check, with precision on recommendation questions: the share of picks that are in the Collection and meet the constraints (decade, genre, country, runtime). It is chosen from the cost of each mistake:

1. **A wrong answer costs most,** for example a film that is not in the Collection, because the user may plan a screening around it.
2. **An answer with no source** comes next, since the user cannot check it.
3. **An unnecessary refusal** costs least, as long as all five unanswerable questions are still refused.

**The baseline.** The same LLM with no retrieval and no tools. It passed **11 of 25** (44%), with a precision of 0.70. It named 8 titles out of 41 that are not in the Collection, cited nothing, and wrongly refused 3 answerable questions.

## What was built

A retrieval-augmented pipeline with two LLM calls, in `src/functions.py`:

1. **Extract the constraints.** A first call to `gpt-6-luna` reads the question and returns the hard limits (years, genre, country, director, runtime) and the type of question as structured JSON.
2. **Route it.**
   - **List and count questions on catalog fields** go to a read-only **catalog tool**, which answers from the catalog rows and cites them, with no second call.
   - **Everything else goes to retrieval.** Local `all-MiniLM-L6-v2` embeddings in Chroma and BM25 keyword search are merged by reciprocal rank fusion. The 40 best passages that meet the limits are kept.
3. **Answer only from the passages.** The second call answers from the retrieved passages and cites them. Runtimes are given in a header, and a rule asks for firm picks. Without enough evidence, it says "I don't know".
4. **Check the catalog.** Every film named as a pick is checked against the Collection. If one is outside it, the model is asked once more, and any film still outside is removed.

Each feature was added one at a time and measured on all 30 questions (notebooks 02 and 03):

| Version | Passes (of 25) |
|---|---|
| LLM alone (baseline) | 11 |
| Retrieval with catalog filter, `k` = 5 | 8 |
| `k` = 10 | 10 |
| `k` = 20 | 12 |
| `k` = 40 | 16 |
| + runtime header | 17 |
| + firm picks | 18 |
| + hybrid search (BM25 + embeddings) | 20 |
| **+ catalog tool (final)** | **23** |

The final settings are saved in `models/config.json` and read by the app.

## The result

| Measure | LLM alone | Final |
|---|---|---|
| Automatic checks passed (of 25) | 11 | **23** |
| Precision on recommendation questions | 0.70 | **1.00** |
| Titles named outside the Collection | 8 of 41 | **0 of 77** |
| Questions citing an expected source (of 17) | 0 | **17** |
| Answerable questions wrongly refused (of 25) | 3 | **0** |
| Unanswerable questions refused (of 5) | 4 | **5** |

The costliest mistake in the framing, recommending a film outside the Collection, did not occur once.

**Where it goes wrong.**

- **Question 7** (Brazilian productions or films shot in Brazil) is a **generation failure.** 7 of the 8 required films were retrieved, but the answer named only 3.
- **Question 11** ("Was Citizen Kane the first Criterion release?") is a **retrieval failure.** The passage saying the first DVD was *Grand Illusion* was not retrieved, so the answer confirmed the premise.

The full analysis is in `notebooks/03_evaluate.ipynb`, the report.

## Cost and speed

- **$0.00092 per question** on average: about $0.0010 through retrieval, and $0.00009 through the catalog tool. Embeddings run locally, at no cost.
- **About $27.60 a month** at 1,000 questions a day, or about 2.8 times the LLM alone, which buys sources for every answer and no films outside the Collection.
- **Speed.** About 3 to 8 seconds per question with both calls. The app answers follow-ups with one extra call. The first question of a session takes about 30 seconds, while the embedding model and keyword index load.

Prices are those of `gpt-6-luna` in November 2026, set in `USD_PER_1M_TOKENS` in `src/functions.py`.

## Limitations

- **The scores are optimistic.** The features were chosen by looking at what this evaluation set asks for, and each version was run once. 23 of 25 says more about these 30 questions than about new ones.
- **Taste is not measured.** The checks test hard constraints, so a pick can pass and still be a weak fit. Treat the picks as a shortlist.
- **Lists that need article facts can be incomplete.** Lists on director, year, country or genre come from the catalog and are complete. A list that also depends on the articles, such as filming locations, can miss films (question 7).
- **False premises.** It can confirm a false premise when the correcting passage is not retrieved (question 11).
- **The metadata is TMDB's.** Filters are only as good as TMDB's fields. 71 films have no runtime, so a runtime limit leaves them out.
- **Live facts are out of reach.** Streaming, showtimes, prices and future releases are not in the sources, so it refuses them.
- **Frozen snapshot.** The data is a November 2026 snapshot, in English. Films the Collection adds later are unknown to it.
- **Follow-up questions are untested.** The app's chat rewrites a follow-up into a standalone question, but this step was not part of the evaluation.

## How to run it

Tested with Python 3.14 (Anaconda), Streamlit 1.58, Chroma 1.5 and the OpenAI SDK 2.49. Run every command from the repo root unless noted.

**1. Clone and install.**

```bash
git clone https://github.com/gbrlgms/project-3-ai-product.git
cd project-3-ai-product
pip install -r requirements.txt     # on conda, read the note at the top of the file first
```

**2. Add your keys.** Create a file named `.env` at the repo root. It is gitignored, so never commit it.

```text
OPENAI_API_KEY=sk-...
TMDB_TOKEN=...          # a TMDB "API Read Access Token", free at themoviedb.org/settings/api
```

The app also accepts `OPENAI_API_KEY` in `.streamlit/secrets.toml`, which is gitignored too, and which wins over `.env`.

**3. Download the data.** This fetches the Criterion list, every linked Wikipedia article and the TMDB records into `data/raw/`, and writes the catalog tables to `data/processed/`. Everything is cached, so you can stop it and run it again to continue.

```bash
python download_data.py
```

**4. Build the chunks.** This writes `data/processed/chunks.jsonl` and `chunks_excluded.csv`, the articles left out.

```bash
cd src && python prepare_documents.py && cd ..
```

**5. Build the index and save the settings.** Run `notebooks/02_build.ipynb` top to bottom. The index takes about 3 minutes the first time and is saved in `models/chroma`. The final cell writes `models/config.json`. On a fresh clone there are no saved runs, so the notebook calls the API for its comparisons. That costs a few cents.

**6. Run the app.**

```bash
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`. It has four pages:

- **Guide**, a chat where you ask for films. Answers link to each film's page and list their sources, and follow-up questions keep the context.
- **Catalog**, every film in alphabetical order, as a poster grid or a list, filtered by country, decade and genre.
- **Search**, by title or director.
- **Film pages**, with the poster, director, year, overview and links to TMDB and Wikipedia.

**Fonts.** The app's look uses licensed fonts (Extenda 30 Deca and ITC Franklin Gothic) that are not in the public repository. The deployed app downloads them from the private dataset, with the index. Without them it falls back to free Google Fonts, and everything works the same. See [`app/static/fonts/README.md`](app/static/fonts/README.md).

**Deploying it on Streamlit Community Cloud.** The index and the catalog tables are gitignored, since they hold TMDB data that must not be republished, and so are the licensed fonts. All of them live in a private Hugging Face dataset, and the deployed app downloads them on its first start.

1. Upload them once with a Hugging Face token that has write access (and again whenever the index is rebuilt):

   ```bash
   python -c "from src.functions import upload_app_files; upload_app_files('<hf-user>/criterion-guide-data', '<write token>')"
   ```

2. On [share.streamlit.io](https://share.streamlit.io), create an app from this repo. Set the main file to `app/streamlit_app.py` and Python to 3.13 under *Advanced settings*. The app installs from `app/requirements.txt`, a slimmer list than the root one.
3. Add the secrets, with a **read-only** Hugging Face token:

   ```toml
   OPENAI_API_KEY = "sk-..."
   HF_DATASET = "<hf-user>/criterion-guide-data"
   HF_TOKEN = "hf_..."
   ```

The public app caps its cost at 20 questions per visit and 500 per day for the whole server, set in `SESSION_LIMIT` and `DAILY_LIMIT` at the top of the app. A monthly budget in the OpenAI dashboard is the final backstop.

**To reproduce the evaluation**, run the three notebooks in order. Notebook 01 scores the baseline, 02 builds and compares the versions, and 03 is the report. Runs are saved in `data/processed/`, so a second run makes no new API calls.

## Repository layout

```text
.
├── README.md                          this file
├── requirements.txt
├── download_data.py                   fetches the Criterion list, Wikipedia and TMDB
├── .streamlit/config.toml             the app's theme
├── app/
│   ├── streamlit_app.py               the app
│   └── static/fonts/                  the licensed fonts go here (gitignored)
├── data/
│   ├── evaluation_set.csv             the 30 questions, written before the build
│   ├── raw/                           downloads, gitignored
│   └── processed/                     tables, chunks and saved runs, gitignored
├── models/                            Chroma index and config.json, gitignored
├── notebooks/
│   ├── 01_framing_and_baseline.ipynb  framing, data quality, documents, evaluation set, baseline
│   ├── 02_build.ipynb                 the pipeline and the versions compared
│   └── 03_evaluate.ipynb              the report: evaluation, failures, cost, limitations
└── src/
    ├── functions.py                   the pipeline, evaluation helpers and app helpers
    ├── eval_checks.py                 the automatic checks
    ├── first_look.py                  loads the tables and articles
    ├── data_quality.py                cleaning and quality checks
    ├── explore_texts.py               splits articles into sections
    └── prepare_documents.py           chunking and citations
```

## Credits

- Film text from [Wikipedia](https://en.wikipedia.org/), under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- Film metadata and posters from [TMDB](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB.
- The Criterion Collection is a trademark of The Criterion Collection, Inc. This is an independent student project, not affiliated with Criterion.
