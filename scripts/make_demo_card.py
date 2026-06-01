#!/usr/bin/env python3
"""Render a terminal-style SVG card from REAL mcp-gpu mock output.

This is intentionally faithful: it imports the actual server, forces mock
mode, calls the real tools, and draws their genuine output. Re-run any time
to regenerate ``assets/demo.svg``::

    uv run python scripts/make_demo_card.py

No screenshot, no Photoshop — the text on the card is exactly what the tools
return.
"""

from __future__ import annotations

import json
import html
from pathlib import Path

from mcp_gpu import server

# --- Capture real output (mock mode) --------------------------------------
server.set_mock(True)
summary = server.gpu_summary()
power = server.set_power_limit(0, 350)
gpus_json = json.dumps(server.list_gpus()[0], indent=2)

# --- Compose the terminal lines -------------------------------------------
PROMPT = "#2dd4bf"  # teal prompt
DIM = "#7d8590"  # comments / dim
FG = "#e6edf3"  # default foreground
ACC = "#f0883e"  # accent (numbers / values)

lines: list[tuple[str, str]] = []  # (text, color)


def add(text: str = "", color: str = FG) -> None:
    lines.append((text, color))


add("$ uv run mcp-gpu --mock        # synthetic GPUs, no hardware needed", DIM)
add("")
add("> gpu_summary()", PROMPT)
for ln in summary.splitlines():
    add("  " + ln, ACC if ln.startswith("GPU ") else FG)
add("")
add("> set_power_limit(index=0, watts=350)", PROMPT)
add("  " + power, FG)
add("")
add("> list_gpus()[0]", PROMPT)
for ln in gpus_json.splitlines():
    add("  " + ln, FG)

# --- Geometry --------------------------------------------------------------
char_w = 8.4
line_h = 22
pad_x = 28
pad_top = 64
width = 900
height = pad_top + line_h * len(lines) + 28

rows = []
y = pad_top
for text, color in lines:
    safe = html.escape(text) or " "
    rows.append(
        f'<text x="{pad_x}" y="{y}" xml:space="preserve" fill="{color}">{safe}</text>'
    )
    y += line_h

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="mcp-gpu mock-mode demo output">
  <defs>
    <style>
      text {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace; font-size: 14px; }}
    </style>
  </defs>
  <rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#0d1117"/>
  <rect x="0" y="0" width="{width}" height="40" rx="12" fill="#161b22"/>
  <rect x="0" y="28" width="{width}" height="12" fill="#161b22"/>
  <circle cx="22" cy="20" r="6" fill="#ff5f56"/>
  <circle cx="44" cy="20" r="6" fill="#ffbd2e"/>
  <circle cx="66" cy="20" r="6" fill="#27c93f"/>
  <text x="{width/2}" y="25" text-anchor="middle" fill="#7d8590" font-family="ui-monospace, monospace" font-size="13px">mcp-gpu — MCP tools over stdio</text>
  {chr(10).join("  " + r for r in rows)}
</svg>
"""

out = Path(__file__).resolve().parent.parent / "assets" / "demo.svg"
out.write_text(svg, encoding="utf-8")
print(f"wrote {out} ({len(svg)} bytes, {len(lines)} lines)")
