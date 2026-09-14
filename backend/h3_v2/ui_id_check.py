# -*- coding: utf-8 -*-
"""Static UI id cross-check: every $(id) in app.js must exist in index.html."""
import re
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent.parent / "app" / "web"
js = (WEB / "app.js").read_text(encoding="utf-8")
html = (WEB / "index.html").read_text(encoding="utf-8")
used = set(re.findall(r'\$\("([\w-]+)"\)', js))
used |= set(re.findall(r"getElementById\('([\w-]+)'\)", js))
have = set(re.findall(r'id="([\w-]+)"', html))
missing = sorted(u for u in used if u not in have)
print(f"used={len(used)} missing={missing if missing else 'NONE'}")
