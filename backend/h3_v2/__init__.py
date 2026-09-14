# -*- coding: utf-8 -*-
"""H3 Shorts Studio v2 backend package.

v1 (`app/h3app/*`, `_hetero_test/*`, manifests) is never imported for
mutation here - only the pure graph builder is reused for byte-parity.
Backend selection: config overlay `backend/h3_v2/config.json` (absent ->
`legacy_v1`). v1 files are not touched to add this.
"""
from __future__ import annotations

BACKEND_ID = "h3_v2"
LEGACY_BACKEND_ID = "legacy_v1"

__all__ = ["BACKEND_ID", "LEGACY_BACKEND_ID"]
