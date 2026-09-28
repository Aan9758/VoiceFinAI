"""
patch_ui.py — one-shot patch for templates/index.html
=====================================================
Adds three things the new pipeline needs:

  1. A per-browser session_id sent with every /query, so follow-up turns
     ("pehle wale", "compare karo", "start it") resolve against the last turn.
  2. View routing on the `view` field the server now returns — fund_list,
     detail, comparison, confirmation, message. That is the four-template UI
     from §5B of the architecture doc.
  3. Category on the cards, and a graceful dash when expense ratio is unknown
     instead of a fabricated figure.

Writes templates/index.html.bak before touching anything. Idempotent — running
it twice is a no-op.

    python patch_ui.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HTML = Path("templates/index.html")

PATCHES: list[tuple[str, str, str]] = []

# ── 1. session id ────────────────────────────────────────────────────────────
PATCHES.append((
    "session id",
    """async function runPipeline(query) {
  setSub('Fetching funds…', S.PROCESSING);
  try {
    const res  = await fetch('/query', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query})});""",
    """const SESSION_ID = (() => {
  let s = sessionStorage.getItem('vfai_sid');
  if (!s) { s = Math.random().toString(36).slice(2,10); sessionStorage.setItem('vfai_sid', s); }
  return s;
})();

async function runPipeline(query) {
  setSub('Fetching funds…', S.PROCESSING);
  try {
    const res  = await fetch('/query', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query, session_id: SESSION_ID})});""",
))

# ── 2. view routing ──────────────────────────────────────────────────────────
PATCHES.append((
    "view routing",
    """function handleResult(data) {
  if (data.type === 'recommendation') {
    renderCards(data);
    showView('B');
    ariaSpeak(data.voice_text, data.audio_token);
  } else if (data.type === 'out_of_scope') {
    showOOS(data.voice_text);
    showView('B');
    ariaSpeak(data.voice_text, data.audio_token);
  } else if (data.type === 'clarify') {
    showClarify(data.voice_text);
    showView('B');
    ariaSpeak(data.voice_text, data.audio_token);
  } else {
    listen();
  }
}""",
    """const VIEW_TITLES = {
  fund_list:    'Fund Recommendations',
  detail:       'Fund Detail',
  comparison:   'Side by Side',
  confirmation: 'Confirm Your SIP',
};

function handleResult(data) {
  if (data.type === 'noop') { listen(); return; }

  if (data.type === 'recommendation' && data.cards && data.cards.length) {
    renderCards(data, VIEW_TITLES[data.view] || 'Fund Recommendations');
  } else if (data.intent === 'out_of_scope') {
    showOOS(data.voice_text);
  } else {
    showClarify(data.voice_text);
  }

  showView('B');
  ariaSpeak(data.voice_text, data.audio_token);
}""",
))

# ── 3. card rendering: title, category, honest expense ──────────────────────
PATCHES.append((
    "card rendering",
    """function renderCards(data) {
  document.getElementById('cardsHeader').textContent = 'Fund Recommendations';
  document.getElementById('cardsGrid').innerHTML = data.cards.map((c,i)=>`""",
    """function renderCards(data, title) {
  document.getElementById('cardsHeader').textContent = title || 'Fund Recommendations';
  document.getElementById('cardsGrid').className =
    'cards-grid' + (data.cards.length === 1 ? ' single' : data.cards.length === 2 ? ' pair' : '');
  document.getElementById('cardsGrid').innerHTML = data.cards.map((c,i)=>`""",
))

PATCHES.append((
    "card body",
    """          <div class="fname">${esc(c.name_spoken)}</div>
          <div class="fhouse">${esc(c.house)}</div>""",
    """          <div class="fname">${esc(c.name_spoken)}</div>
          <div class="fhouse">${esc(c.house)}${c.category ? ' · ' + esc(c.category) : ''}</div>""",
))

PATCHES.append((
    "expense cell",
    """<div class="mi"><div class="ml">Expense</div><div class="mv">${c.expense}</div></div>""",
    """<div class="mi"><div class="ml">Expense</div><div class="mv" title="${c.expense === '—' ? 'Not published by mfapi.in — requires AMC factsheet' : ''}">${c.expense}</div></div>""",
))


def main() -> int:
    if not HTML.exists():
        print(f"  !! {HTML} not found. Run this from the project root.")
        return 1

    text = HTML.read_text(encoding="utf-8")
    applied, skipped, missing = [], [], []

    for name, old, new in PATCHES:
        if new in text:
            skipped.append(name)
        elif old in text:
            text = text.replace(old, new, 1)
            applied.append(name)
        else:
            missing.append(name)

    if missing:
        print("  !! could not locate these sections — patch them by hand:")
        for m in missing:
            print(f"       - {m}")

    if not applied:
        print("  nothing to do (already patched)")
        return 0 if not missing else 1

    backup = HTML.with_suffix(".html.bak")
    shutil.copy2(HTML, backup)
    HTML.write_text(text, encoding="utf-8")

    print(f"  backup  → {backup}")
    for a in applied:
        print(f"  patched → {a}")
    for s in skipped:
        print(f"  already → {s}")

    print("\n  Optional CSS for the 1- and 2-card views — add to your <style>:\n")
    print("    .cards-grid.single { grid-template-columns: minmax(280px, 460px); }")
    print("    .cards-grid.pair   { grid-template-columns: repeat(2, minmax(240px, 1fr)); }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
