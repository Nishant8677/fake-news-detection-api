"""Measures what Streamlit Community Cloud actually gives an app.

Streamlit's FAQ states "690MB minimum, 2.7GBs maximum", which is a range rather
than a guarantee, and the post dates from February 2024. This service needs
1,008 MB steady and 1,344 MB peak (results/benchmark_metrics.json), which sits
inside that range -- so reading the documentation cannot answer whether it fits.

Deploy this first. It imports nothing heavier than psutil, so it will start
regardless, and it reports the real ceiling on this account before anything is
built against it.

Entrypoint: deploy/streamlit/probe_app.py
"""

from __future__ import annotations

import os
import platform
import sys

import psutil
import streamlit as st

# Measured on the machine BENCHMARK.md describes.
NEEDED_STEADY_MB = 1007.54
NEEDED_PEAK_MB = 1344.34

st.set_page_config(page_title="Resource probe", page_icon="📏")
st.title("Streamlit Community Cloud — resource probe")
st.caption(
    "Throwaway app. Answers one question: is there room here for a BERT "
    "service that peaks at 1,344 MB?"
)

vm = psutil.virtual_memory()
proc = psutil.Process(os.getpid())

total_mb = vm.total / 1024**2
available_mb = vm.available / 1024**2
rss_mb = proc.memory_info().rss / 1024**2

# cgroup limits are the number that actually kills the app; virtual_memory()
# often reports the host's memory rather than the container's allowance.
cgroup_limit_mb = None
for path in (
    "/sys/fs/cgroup/memory.max",                    # cgroup v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
):
    try:
        with open(path) as fh:
            raw = fh.read().strip()
        if raw and raw != "max":
            value = int(raw)
            # v1 reports a sentinel near 2^63 when unlimited.
            if value < 2**60:
                cgroup_limit_mb = value / 1024**2
        break
    except (OSError, ValueError):
        continue

st.subheader("What this container reports")
col1, col2, col3 = st.columns(3)
col1.metric("Total RAM", f"{total_mb:,.0f} MB")
col2.metric("Available", f"{available_mb:,.0f} MB")
col3.metric("This process (RSS)", f"{rss_mb:,.0f} MB")

if cgroup_limit_mb is not None:
    st.metric("cgroup memory limit", f"{cgroup_limit_mb:,.0f} MB")
    ceiling_mb = cgroup_limit_mb
    ceiling_source = "the cgroup limit"
else:
    st.info(
        "No cgroup memory limit is readable, so the figures above may describe "
        "the host rather than this app's allowance. Treat the verdict as "
        "indicative and confirm by watching for an out-of-memory restart."
    )
    ceiling_mb = total_mb
    ceiling_source = "reported total RAM"

st.subheader("Verdict")
headroom = ceiling_mb - NEEDED_PEAK_MB
st.write(
    f"The service needs **{NEEDED_STEADY_MB:,.0f} MB** steady and "
    f"**{NEEDED_PEAK_MB:,.0f} MB** peak. Against {ceiling_source} "
    f"(**{ceiling_mb:,.0f} MB**) that leaves **{headroom:,.0f} MB** at peak, "
    "before Streamlit's own overhead."
)

if headroom > 400:
    st.success("Fits with room to spare. Build the wrapper.")
elif headroom > 0:
    st.warning(
        "Fits, but thinly. Streamlit's overhead could consume the margin — "
        "expect intermittent out-of-memory restarts rather than a clean failure."
    )
else:
    st.error(
        "Does not fit. Quantise to int8 ONNX first, or host somewhere with more "
        "memory. This is a real constraint, not an elective one."
    )

with st.expander("Environment"):
    st.json(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count_logical": psutil.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_ram_mb": round(total_mb, 1),
            "cgroup_limit_mb": round(cgroup_limit_mb, 1) if cgroup_limit_mb else None,
            "disk_free_gb": round(psutil.disk_usage("/").free / 1024**3, 2),
            "executable": sys.executable,
        }
    )

st.caption(
    "Delete this app once the number is recorded. Its only job is to replace a "
    "guess with a measurement."
)
