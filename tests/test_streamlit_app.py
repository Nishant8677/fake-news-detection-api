"""Regression tests for the Streamlit deployment surface.

Both cases run WITHOUT model weights, matching the rest of this suite, so they
work anywhere the dependency is installed.

Skipped when streamlit is absent: it is a deployment dependency listed in
deploy/streamlit/requirements.txt, not part of the serving stack that
BENCHMARK.md measures, so it is deliberately not in the root requirements.
"""

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="deployment-only dependency")

from streamlit.testing.v1 import AppTest  # noqa: E402

# Absolute: AppTest resolves a relative path against the working directory,
# which is not necessarily the repository root when pytest is invoked.
APP = str(Path(__file__).resolve().parents[1] / "deploy" / "streamlit" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def _no_weights(monkeypatch):
    """Points the app at weights that cannot exist, and makes that stick.

    Two layers of caching fight this. inference/app.py loads the model at
    import time and sets MODEL_LOADED once, so an earlier test in the session
    that imported it with real weights leaves MODEL_LOADED=True for everything
    after -- setting MODEL_DIR alone would silently test the opposite of what
    is intended. And st.cache_resource persists across AppTest instances in the
    same process.

    Dropping the module from sys.modules and clearing the Streamlit cache
    forces a fresh import under the patched environment.
    """
    import streamlit as st

    monkeypatch.setenv("MODEL_DIR", "/nonexistent-weights")
    monkeypatch.setenv("LOG_PREDICTIONS", "0")

    sys.modules.pop("inference.app", None)
    st.cache_resource.clear()
    yield
    sys.modules.pop("inference.app", None)
    st.cache_resource.clear()


def test_app_starts_without_a_secrets_file():
    """The app must not crash when no secrets.toml exists anywhere.

    `"MODEL_DIR" in st.secrets` does not return False in that situation; it
    raises StreamlitSecretNotFoundError while searching for a file. Guarding on
    membership therefore crashed the app on startup in precisely the case the
    guard existed to handle -- a fresh deployment before secrets are set, and
    every local run. Regression test for that fix.
    """
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()

    assert at.exception == [], f"app raised on startup: {at.exception}"
    assert at.title, "no title rendered, so the script did not reach the body"


def test_missing_weights_warn_rather_than_crash():
    """Absent weights should degrade visibly, the way /health returns 503."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()

    assert at.exception == []
    assert len(at.warning) == 1, "expected exactly one warning about the weights"
    assert "MODEL_DIR" in at.warning[0].value


def test_accuracy_disclaimer_precedes_the_input():
    """The 25.7% figure must be on screen, not buried in a footer.

    A public classifier that is wrong more often than it is right should say so
    before it accepts input.
    """
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()

    disclaimers = [e.value for e in at.error if "25.7%" in e.value]
    assert disclaimers, "the accuracy disclaimer is missing from the page"
    assert "do not use it to decide" in disclaimers[0].lower()


def test_env_var_takes_precedence_over_secrets():
    """MODEL_DIR from the environment must win, so local runs need no secrets."""
    assert os.environ["MODEL_DIR"] == "/nonexistent-weights"

    at = AppTest.from_file(APP, default_timeout=200)
    at.run()

    assert at.exception == []
