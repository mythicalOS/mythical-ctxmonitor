"""Shared test wiring: import the scorer from bin/ and locate the fixtures."""
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BIN = os.path.join(REPO, "bin")
TESTDATA = os.path.join(BIN, "testdata")
FIXTURE = os.path.join(TESTDATA, "golden-session.jsonl")
EXPECTED = os.path.join(TESTDATA, "golden-session.expected.json")

sys.path.insert(0, BIN)
import context_quality as cq  # noqa: E402  (path set up first, deliberately)


def load_expected():
    with open(EXPECTED, encoding="utf-8") as f:
        return json.load(f)


def load_records():
    return cq.read_jsonl(FIXTURE)
