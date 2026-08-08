# Deploying the Space

Two repositories on Hugging Face: a **model** repo holding the 418 MB of weights,
and a **Space** running this Dockerfile. They are separate because weights do not
belong in a Space's git history any more than they belong in this one's.

Run these yourself — the token should not pass through a terminal you do not own.

## 1. Authenticate

```bash
pip install -U "huggingface_hub[cli]"
```

```bash
hf auth login
```

Paste a token with **write** scope from <https://huggingface.co/settings/tokens>.

## 2. Push the weights to a model repo

From the repository root, where `model/` lives:

```bash
hf upload Nishant8677/fake-news-liar-bert ./model . --repo-type=model --create
```

Uploads `model.safetensors`, `config.json`, and the tokenizer files. Large files
go through LFS automatically. `model/` stays gitignored here — this is the only
place the weights are published.

## 3. Create the Space and push it

```bash
hf repo create Nishant8677/fake-news-detection-api --repo-type=space --space_sdk docker
```

```bash
git clone https://huggingface.co/spaces/Nishant8677/fake-news-detection-api /tmp/fn-space
```

```bash
cp deploy/hf-space/{Dockerfile,README.md,app.py,requirements.txt} /tmp/fn-space/
```

```bash
cd /tmp/fn-space && git add -A && git commit -m "feat: serve the fake-news API with a Gradio UI" && git push
```

Do **not** copy `DEPLOY.md` — it is instructions for you, not part of the image.

## 4. Point it at the weights

In the Space UI: **Settings → Variables and secrets → New variable**

| Name | Value |
|---|---|
| `MODEL_DIR` | `Nishant8677/fake-news-liar-bert` |

A *variable*, not a secret — it is a public repo id, and putting it in the build
log is fine. The Space rebuilds automatically.

Until this is set, `MODEL_DIR` is empty, weights fail to load, and `/health`
returns `503 {"status":"degraded"}` by design. That is the healthcheck working,
not a broken deploy.

## 5. Verify

```bash
curl -s https://nishant8677-fake-news-detection-api.hf.space/health
```

Expect `{"status":"ok","model_loaded":true,"dataset_version":"v1.0-liar"}`.

```bash
curl -s -X POST https://nishant8677-fake-news-detection-api.hf.space/predict -H "Content-Type: application/json" -d '{"text":"Our state has the lowest unemployment rate in the entire country."}'
```

Then open the Space URL — the Gradio page should render with the accuracy
disclaimer above the input box.

## Deploying a later version

The Dockerfile clones this repository at `REPO_REF` (default `main`) rather than
vendoring a copy of `inference/`, so the Space always serves code that exists
here. To pin a specific commit instead of tracking `main`, add a build argument
in the Space's `Dockerfile`:

```dockerfile
ARG REPO_REF=a09635a
```

The build writes the resolved SHA to `SERVING_COMMIT` inside the image, so a
running Space can always name the commit it is serving.

## Free-tier notes

- 2 vCPU / 16 GB is enough; the model is 418 MB and inference is CPU-only.
- Spaces sleep after inactivity. First request after a sleep pays the 0.67 s
  cold model load plus container start — worth saying on the portfolio link so a
  reviewer hitting a cold Space does not read it as a slow API.
- `LOG_PREDICTIONS=0` is set in the Dockerfile. Leave it. Submitted text belongs
  to whoever typed it, and the Space filesystem is ephemeral anyway.
