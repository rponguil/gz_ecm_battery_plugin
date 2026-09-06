#!/usr/bin/env python3
"""Single source of truth for every number the paper quotes.

Each validation script records its headline results here, at the point where
they are computed, instead of the numbers being copied by hand into prose. The
accumulated manifest lands in results/paper_numbers.json.

Why this exists: numbers written into a manuscript by hand go stale silently
when the pipeline changes. A figure that was correct when it was typed stays in
the text long after the run that produced it stopped existing, and nothing
fails. Recording them at the source makes "the paper agrees with the artifact"
something a script can check rather than something an author has to promise.

Usage from a validation script:

    from paper_numbers import record
    record("panasonic.rmse_mv", 34.34, "mV", "same-session OCV, constant R0/R1")

Then, after running the validators:

    python3 report_paper_numbers.py        # print the manifest
"""

import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
MANIFEST = os.path.join(RESULTS_DIR, "paper_numbers.json")


def _load():
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST, encoding="utf-8") as fh:
                return json.load(fh)
        except (ValueError, OSError):
            pass
    return {}


def record(key, value, unit="", note="", source=None):
    """Record one reported quantity, overwriting any earlier value for `key`.

    `source` defaults to the script that called this, so the manifest says
    which run produced each number.
    """
    import inspect

    if source is None:
        frame = inspect.stack()[1]
        source = os.path.basename(frame.filename)

    data = _load()
    data[key] = {
        "value": round(float(value), 6),
        "unit": unit,
        "note": note,
        "source": source,
        "recorded": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return value


def load_manifest():
    return _load()
