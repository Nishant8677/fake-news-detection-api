import pytest
from preprocess import LABEL_MAP

def test_label_mapping():
    assert LABEL_MAP["pants-fire"] == 0
    assert LABEL_MAP["true"] == 5
    assert len(LABEL_MAP) == 6

def test_label_map_coverage():
    expected_keys = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    for key in expected_keys:
        assert key in LABEL_MAP
