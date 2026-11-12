# Font files

The app uses these files. They are licensed fonts, so git ignores them and they are not in the repository:

| File | Font | Used for |
| --- | --- | --- |
| `Extenda30Deca.ttf` | Extenda 30 Deca (Zetafonts) | the product's title |
| `FranklinGothicLTCompressed-Book.ttf` | ITC Franklin Gothic LT Book Compressed | body text |
| `FranklinGothicLTCondensed-Medium.otf` | ITC Franklin Gothic LT Medium Condensed | headings, buttons and gold labels |
| `FranklinGothicLTCompressed-BookItalic.ttf` | ITC Franklin Gothic LT Book Compressed Italic | italic body text |
| `FranklinGothicLTCondensed-MediumItalic.otf` | ITC Franklin Gothic LT Medium Condensed Italic | italic headings and labels |

The Franklin files are set in `.streamlit/config.toml`, and Extenda in the styles at the top of `app/streamlit_app.py`.
The deployed app downloads them from the private Hugging Face dataset named in its secrets (see the main README), so they are never published.
Without the files, the app falls back to Big Shoulders Display and Archivo Narrow from Google Fonts.
