# Deploying to Streamlit Community Cloud

Free, no card, no subscription — unlike Hugging Face Spaces, which now require
PRO for anything other than a static Space.

**Deploy the probe first.** This service needs 1,008 MB steady and 1,344 MB peak
(`results/benchmark_metrics.json`). Streamlit's FAQ quotes "690MB minimum,
2.7GBs maximum", which is a range rather than a guarantee, and the post is from
February 2024. Our figure sits inside that range, so reading the documentation
cannot settle it. Fifteen minutes of measurement can.

## 1. Probe the ceiling

<https://share.streamlit.io> → **Create app** → **Deploy a public app from GitHub**

| Field | Value |
|---|---|
| Repository | `Nishant8677/fake-news-detection-api` |
| Branch | `main` |
| Main file path | `deploy/streamlit/probe_app.py` |

Under **Advanced settings**, leave secrets empty — the probe needs none.

Streamlit reads a requirements file from the entrypoint's directory, so it will
pick up `deploy/streamlit/requirements.txt`. That file includes torch, which the
probe does not need but which makes the install slow. To probe faster, rename
`probe_requirements.txt` to `requirements.txt` in a scratch branch, or simply
wait — the probe is throwaway either way.

Read the verdict:

- **> 400 MB headroom** — build the wrapper, step 2.
- **thin margin** — expect intermittent OOM restarts rather than clean failure.
- **negative** — it does not fit. Quantise to int8 ONNX first. That is then a
  real constraint, and the resulting "4× smaller, here is the accuracy I traded"
  is a genuine engineering story rather than an elective one.

Delete the probe app once the number is recorded.

## 2. Deploy the app

Same flow, with:

| Field | Value |
|---|---|
| Main file path | `deploy/streamlit/streamlit_app.py` |

Under **Advanced settings → Secrets**, paste:

```toml
MODEL_DIR = "Nishant8677/fake-news-liar-bert"
```

This is why the weights are on the Hub rather than in git: 418 MB has no
business in a repository, and `from_pretrained` resolves a Hub id exactly as it
resolves a local path, so no code changes to serve this way.

Without the secret the app still starts and tells you the weights did not load,
rather than crashing. That is the intended failure, not a broken deploy.

That behaviour is not free: an earlier version guarded with
`"MODEL_DIR" in st.secrets`, which does not return False when no secrets file
exists — it raises while searching for one, crashing the app on startup in
precisely the case the guard was for. `tests/test_streamlit_app.py` pins both
paths so it cannot come back.

## 3. Verify

Open the URL and confirm:

- the accuracy warning renders **above** the input
- classifying *"Our state has the lowest unemployment rate in the entire
  country."* returns **`mostly-true` at 54.6% confidence**

That value has now come from the host, the repository container, and the Space
container. If Streamlit returns it too, four independent paths agree — which is
what calling the same `predict()` everywhere is for.

## Notes

- **First load is slow.** Community Cloud sleeps idle apps; waking one pays
  container start plus the 0.67 s model load, and the first run also downloads
  418 MB from the Hub. Worth saying on the portfolio link so a reviewer does not
  read a cold start as a slow model.
- **Prediction logging is forced off** (`LOG_PREDICTIONS=0`). Leave it that way.
- **This is a demo surface, not the product.** The REST service, its Dockerfile
  and its benchmarks are the engineering; this page exists so the link is
  clickable.
