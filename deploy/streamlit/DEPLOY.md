# Deploying to Streamlit Community Cloud

Free, no card, no subscription — unlike Hugging Face Spaces, which now require
PRO for anything other than a static Space.

## Measured, 8 August 2026

The probe below was run rather than reasoned about. What Community Cloud
actually gives an app:

| | |
|---|---|
| cgroup memory limit | **3,072 MB** |
| CPU | 16 logical / 8 physical |
| Disk free | 110 GB |
| Default Python | **3.14.7** |
| Host RAM (not the limit) | 128,817 MB |

Against 1,344 MB peak that leaves **1,728 MB of headroom**. It fits.

Note the last two rows. The host reports 128 GB, which is irrelevant — the
cgroup limit is what terminates an app, and it is 42× smaller. Reading
`virtual_memory().total` would have suggested unlimited room.

Streamlit's FAQ quotes "690MB minimum, 2.7GBs maximum". The real figure is
3,072 MB, close to but not equal to the documented ceiling, which is why this
was measured rather than assumed.

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
| **Python version** | **3.11** |

Set the Python version deliberately. Community Cloud defaults to 3.14, and
`BENCHMARK.md` measured 3.11.4 — the pinned stack installs on both (torch,
numpy and pandas all ship cp314 wheels; transformers is pure Python), so this is
about consistency rather than compatibility. The point of pinning the serving
packages is that the deployment and the published latency figures describe the
same environment, and the interpreter is part of that environment.

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

## Live

<https://fake-news-detection-api-ktb2chsjvps7cqgsmknbcn.streamlit.app/>

Running on Python 3.11.15 with weights from
[`Nishant8677/fake-news-liar-bert`](https://huggingface.co/Nishant8677/fake-news-liar-bert).

### Checking it from outside, correctly

`curl` without a cookie jar returns `303` to `share.streamlit.io/-/auth/app` and
loops until it hits the redirect cap. That is **not** a private app — it is
Streamlit's anonymous session bootstrap, which sets a cookie and redirects back.
A browser completes it invisibly; curl cannot unless told to keep cookies:

```bash
curl -s -L -c /tmp/j -b /tmp/j -o /dev/null -w '%{http_code}\n' https://fake-news-detection-api-ktb2chsjvps7cqgsmknbcn.streamlit.app/
```

Expect `200`. Without `-c`/`-b` this looks exactly like a permissions problem
and will send you to change settings that were never wrong.

Note also that the served HTML is a ~9 KB shell containing no app text —
Streamlit delivers the interface over a websocket after load. Grepping the HTML
for page content proves nothing either way.

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
