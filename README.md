# Streamlit Sports Chatbot

A simple Streamlit chatbot that uses a `LogisticRegression` model trained on
sports-related intents defined in `intents.json`.

## Files

- `chatbot.py` — Streamlit app entrypoint
- `intents.json` — chatbot training data (56 intents)
- `requirements.txt` — Python dependencies
- `.gitignore` — ignored files for Git

## Run locally

```bash
pip install -r requirements.txt
streamlit run chatbot.py
```

Open the browser link Streamlit prints and start chatting.

## Deploy to Streamlit Community Cloud

1. Push this folder to a **public** GitHub repo. At minimum it must contain:
   `chatbot.py`, `intents.json`, `requirements.txt`.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick your repo/branch, and set **Main file path** to `chatbot.py`.
4. Click **Deploy**. First build takes 1–2 minutes.

## Notes / known limitations

- The app loads `intents.json` if present, otherwise `intent.json`. Only
  `intents.json` is meant to be used — delete `intent.json` unless you have
  a reason to keep an empty fallback file around.
- This is a **TF-IDF + Logistic Regression intent classifier**, not a
  generative model. It can only reply with one of the pre-written responses
  in `intents.json`, and it will always pick *some* tag even for nonsense
  input — there's no confidence threshold, so off-topic input can get a
  confidently wrong answer. If that matters for your use case, add a
  `clf.predict_proba` threshold check and fall back to a "I don't understand"
  response below a cutoff.
- Model training is cached with `@st.cache_resource` so it only retrains
  when the app restarts or `intents.json` changes — not on every message.
