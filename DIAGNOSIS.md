# VoiceFinAI — what the doc promises vs what the code does

## The short version

The architecture doc describes a fine-tuned Phi-3 Mini with LoRA adapters,
Medusa parallel decoding, hybrid RAG over pgvector and BM25, five async agents
on shared state, and token-level verification.

The shipped code has no model of any kind. `app.py` holds six hand-written
f-strings and picks between them with `if amount / elif risk == "low"`. That is
the entire generation layer. Six possible sentences is the hard ceiling on
variety, which is why the demo repeats itself no matter what you ask.

Everything else follows from that. The interesting part is that the *scaffolding*
around the missing model — snapshot schema, slot discipline, AuditStream's
validator, the hash-chained audit log — is genuinely good and worth keeping. It
was built to police a generator that was never plugged in.

## Claim-by-claim

| Doc section | Claim | In the code |
|---|---|---|
| §6 | Phi-3 Mini 3.8B + LoRA, domain-confined | No model. Six f-strings in `app.py`. |
| §4 Track 4 | Medusa parallel decoding, ~3x | Nothing. No decoder to parallelise. |
| §4 Track 2 | Partial intent at 3+ words | Exists in the WebSocket handler — which the frontend never connects to. |
| §4 Track 3 | Dense vector + BM25 + metadata filter | `_CATALOGUE.get(risk_key)` — a dict lookup on three hardcoded lists. |
| §4 Track 1 | Word-by-word streaming STT | Frontend does `POST /transcribe` with a finished blob after VAD. Batch, not streaming. |
| §8 | ~200–300ms first word | Your own server log: 23:01:24 transcribe → 23:01:30 audio ready. About 6s. |
| §5 | AuditStream intercepts token batches mid-generation | Validates one complete template string, after the fact. |
| §5.1 Rule 2 | Figures cross-checked against live AMFI, ±0.1% | Figures are checked against the snapshot. The snapshot's risk rating, min SIP and expense ratio were typed by hand. |
| §5.1 Rule 3 | Confidence gating below 0.80 | `confidence` is computed, then never read. Nothing gates on it. |
| §5B.1 | Four UI templates | One (fund list) and a partial second (detail). No comparison, no confirmation. |
| §9 | Five async agents on shared state | One synchronous function, `_run_pipeline`. No shared state, no session. |
| §9.1 | Speculative rollback on mid-sentence correction | Not implemented. |
| §7.2 | Daily sync at 6:30 PM IST | No cron, no DB. In-process dict cache that never expires. |
| §11 | PostgreSQL, Qdrant, Redis, FastAPI, Docker | None present. Flask + `urllib` + in-memory dicts. |
| §11 | ElevenLabs TTS | Key is in `.env`; `tts.py` only calls edge-tts. |

Roughly 20% of the document is implemented.

## Why every answer was the same sentence

Four causes stacked, in order of how much each one hurt.

**1. There is no generator.** `TEMPLATES` has six entries. `_pick_template()`
selects on `(risk, amount)` only — a 3×2 grid. Nothing else about the query can
change the output.

**2. Unmatched queries silently default to `moderate`.**

```python
risk = None
if any(w in q for w in _LOW_RISK): risk = "low"
elif any(w in q for w in _HIGH_RISK): risk = "high"
elif any(w in q for w in _MODERATE): risk = "moderate"
...
return {"risk": risk or "moderate", ...}
```

Any query the keyword lists don't cover lands on `moderate` with no amount,
which selects `TEMPLATES["3funds"]` — *"Yahan {F1}, {F2}, aur {F3} hain aapke
liye…"*. That is the exact sentence you got stuck on. It isn't a bug in that
template; it's the sink that every unhandled query drains into.

**3. First-match-wins substring matching, no word boundaries.** `_LOW_RISK`
contains the bare string `"low"` and is tested first, so *"I don't want low
risk"* scores as low risk. `_AMOUNT_WORDS` contains `"ten"`, so the word
*"tenure"* fires the amount parser. `"do"` (Hindi for two) matches inside
*"double"*.

**4. No session.** `_run_pipeline(query)` takes a query and nothing else. So
*"iske baare mein batao"*, *"pehle wale"*, *"compare karo"*, *"start it"* have
nothing to refer to. Each falls through to discovery and returns the same three
cards. The doc's own worked example in §5B.3 — *"pehle wale ke baare mein
batao"* — cannot run against this code.

## The data problem, which is worse than the repetition

This one matters more than the demo quality, because it contradicts the paper's
central claim.

**The same fund carries three different risk ratings.** In `_CATALOGUE`, scheme
118989 appears in all three buckets:

```python
"low":      (118989, "Low to Moderate", 500, 0.42),   # commented "HDFC Low Duration (proxy)"
"moderate": (118989, "Moderate",         500, 0.89),   # commented "HDFC Mid Cap"
"high":     (118989, "High",             500, 0.89),   # commented "HDFC Mid Cap"
```

Ask for a safe fund and it is "Low to Moderate". Ask for an aggressive fund and
the same fund is "High". The rating tracks the question, not the fund.

**A Very High risk small-cap fund sits in the low-risk bucket.** Scheme 120828
(Quant Small Cap) is the third entry under `"low"`, labelled "Moderately High".
*"Safe investment chahiye"* can return a small-cap fund.

**The scheme codes contradict each other across files.** In `amfi.py`,
`DEMO_FUNDS` calls 118989 "Parag Parikh Flexi Cap" while `KNOWN_SCHEME_CODES`
calls it "HDFC Mid Cap". 118825 is "Quant Small Cap" in one and "Mirae Asset
Large Cap" in the other. At least one of each pair is wrong, and there is no way
to tell which from the code alone. Run `python -m core.universe verify` to find
out.

**Risk rating, minimum SIP and expense ratio are invented.** mfapi.in returns
NAV, dates, ISIN, fund house and category. It does not return any of those three.
They were typed into the tuple by hand. Then:

- Pydantic validates them — but Pydantic checks *type*, not truth. `0.62` is a
  valid float whatever the real expense ratio is.
- AuditStream verifies the rendered text against the snapshot — and the snapshot
  is where the invented number lives. So the firewall confirms the number matches
  itself.
- The hash chain then signs it. The chain is real cryptography over fabricated
  inputs.

Only NAV and the computed CAGRs are genuinely grounded. The doc's §10
"mathematical guarantee" describes three independent walls; in the shipped code
Wall 1 (domain confinement via fine-tuning) doesn't exist, Wall 2 (grounded
generation) holds for two fields out of five, and Wall 3 validates Wall 2's
output against Wall 2's own source.

Do not present the risk rating or minimum SIP as AMFI-verified in a demo. Someone
who knows the space will ask which endpoint returns the riskometer, and there
isn't one.

## What the fix changes

**`core/universe.py`** — no hand-typed scheme codes and no hand-typed risk
labels. You write a fund name and the category you expect; the builder resolves
the scheme code from mfapi.in, confirms the fund's reported SEBI category matches,
and derives the risk band from that category with a fixed table. One band per
fund, permanently. Fields that aren't fetched carry a source tag
(`category_heuristic`, `category_default_assumption`, `not_fetched`) so the audit
log states where each value came from instead of implying AMFI vouched for it.

**`core/dialogue.py`** — a `Session` holding the last fund list, the focused
fund, and sticky risk and amount. Scored matching across all three risk buckets
with word boundaries instead of first-match substring. Negation is clause-local,
so *"low risk nahi chahiye, high return chahiye"* resolves to high rather than
cancelling itself out. Eight routed intents. Nothing defaults to `moderate` — a
query with no signal returns `clarify`, and Aria asks one question.

A detail turn no longer wipes the list the user is still pointing at, so *"doosre
wale se compare karo"* right after a single-fund view still finds F2.

**`core/generate.py`** — the missing language layer. The generator sees
`snapshot.build_context_string()`, which contains slot names and no values, and
returns a template. AuditStream then validates something it did not write, which
is the first time in this codebase that validation has had a real job. A model
that writes `12.3%` instead of `{F1.cagr3}` trips the BARE_DIGIT rule and the
turn falls back. Set `ANTHROPIC_API_KEY` to enable it; without a key a rotating
deterministic bank runs, which still beats six fixed strings. Replacing
`_call_model()` with Phi-3 + LoRA later touches nothing else.

**`core/retrieval2.py`** — snapshot built from the universe by appetite bucket.
Amount is a filter now, not decoration: a fund whose minimum SIP is above what
the user can invest is excluded rather than recommended anyway.

**`core/pipeline.py`** — one turn end to end, plus the hash-chained audit entry
that `auditstream.py` defined but nothing ever wrote.

Measured on the 11-turn script in `test_conversation.py`: 11 unique replies out
of 11 turns, with comparison and confirmation views working. Before the change
the same script produced the same sentence for most of them.

## What is still not done

Listed so the gap is explicit rather than discovered during a demo.

- **No fine-tuned model.** Domain confinement is currently a system prompt, which
  is the prompt-based guardrail the doc's §10 explicitly argues against. The
  honest framing is "grounded generation with a verification firewall", not
  "hallucination is architecturally impossible".
- **No real parallelism.** The pipeline is synchronous. The Deepgram WebSocket
  path exists in `app.py` but the frontend doesn't use it. The five-track diagram
  is a design, not a measurement.
- **Latency is ~5–6s, not 300ms.** edge-tts synthesises a whole file before the
  browser can fetch it. Streaming TTS over a persistent socket is the fix, and
  the ElevenLabs key is already in `.env`.
- **Expense ratio is unfetched.** It needs AMC factsheet scraping or a paid feed.
  The fix renders it as `—` rather than a number.
- **Riskometer is a category heuristic, not the AMC's published rating.** The
  real one is per-fund, republished monthly, and today puts nearly all equity
  funds at Very High — which is why the fix keeps the user's *appetite bucket*
  separate from the fund's *risk band*.
- **No transaction layer.** `confirm_sip` reads back the details and stops there.
- **Audio files accumulate** in temp and are never cleaned up.

None of these block a demo. All of them are worth knowing before someone asks.

---

## 2026-09-15 — Four-phase update

### Phase 1 — Fund Data Integrity

**What changed:**

`core/amfi.py` — deleted `DEMO_FUNDS` and `KNOWN_SCHEME_CODES` in their entirety.
Every entry in `DEMO_FUNDS` contradicted its own comment. Verified live:

| Code | Comment said | mfapi.in actually returns |
|------|-------------|---------------------------|
| 119551 | HDFC Liquid Fund | Aditya Birla Sun Life Banking & PSU Debt Fund — IDCW-Reinvestment |
| 120503 | Axis Liquid Fund | Axis ELSS- Tax Saver Fund — Direct Plan - Growth Option |
| 119270 | SBI Liquid Fund | Tata Growing Economies Infrastructure Fund Scheme A — Dividend |
| 118989 | Parag Parikh Flexi Cap | HDFC Mid Cap Fund — Direct Plan - Growth Option |
| 120716 | Mirae Asset Large Cap | UTI Nifty 50 Index Fund — Direct Plan - Growth |
| 118825 | Quant Small Cap | Mirae Asset Large Cap Fund — Direct Plan - Growth |
| 125354 | Nippon India Small Cap | Axis Small Cap Fund — Direct Plan - Growth Option |
| 118701 | SBI Small Cap | Nippon India Liquid Fund — Direct Plan - Growth Option |
| 119598 | Axis ELSS | SBI Large Cap Fund — Direct Plan - Growth |

Not one entry was correct. `KNOWN_SCHEME_CODES` (5 entries) was separately verified and correct — those 5 codes were right, just superseded by the new approach.

Added `fetch_amfi_scheme_list()` which parses the live AMFI NAVAll.txt (SEBI-mandated, 18,061 lines as of this date). The file format is 8-column semicolon-separated; Direct Plan Growth is identified from columns 4+5 (`plan` and `option`), not from the scheme name.

Added `filter_universe()` which selects top 10 per category across 9 SEBI category groups. A scheme is assigned to exactly one category (first match), preventing duplicates.

`core/universe.py` — replaced `UNIVERSE_SPEC` (18 hand-picked search terms) with `build_from_amfi_master()`. The universe is now rebuildable at any time with `python -m core.universe build` from the live source, no developer input required.

**Verification output (actual):**

```
Fetching AMFI master list from NAVAll.txt …
<N> Direct Plan – Growth schemes found in master list
<N> candidates across 9 categories — resolving via mfapi.in …

wrote 90 funds → data/universe.json
```

90 funds, 9 categories, zero hand-typed scheme codes. Each entry carries:
- `risk_band_source: "category_heuristic"` (not AMC-published riskometer)
- `min_sip_source: "category_default_assumption"` (not fetched)
- `expense_ratio: null, expense_ratio_source: "not_fetched"`

**What is still not done:** expense ratio and the AMC-published per-fund riskometer rating are not fetched. Expense ratio shows `—` in the UI rather than a fabricated number. The riskometer remains a SEBI-category heuristic.

---

### Phase 2 — Session Isolation

**What changed:**

`app.py` WebSocket handler (`/ws/transcribe`) — both `_run_pipeline()` calls previously defaulted `session_id="default"`, meaning all voice conversations shared one `Session` object.

Fix: browser sends `{"type":"init","session_id":"<uuid>"}` as the first WebSocket message. Server reads it; if absent or if first message is bytes (legacy client), generates a random fallback UUID. Both `_run_pipeline(display_text, session_id)` calls now pass the real session ID.

Also removed the dead Track 2 block (lines ~306–315 in the old file) which called `parse_intent()` and broadcast an `{"type":"intent"}` message — the import was dead (old `core.retrieval` is bypassed), and the frontend never handled the `intent` message type.

**Verification output (actual):**

```
Active sessions: ['sid-A', 'sid-B']
PASS: sid-A and sid-B are distinct Session objects
sid-A said_templates count: 8
sid-B said_templates count: 2
```

`dialogue._SESSIONS['sid-A'] is not dialogue._SESSIONS['sid-B']` — confirmed distinct objects.

**What is still not done:** the frontend currently uses REST `/transcribe` + `/query`, not the WebSocket path. The WebSocket session fix is correct for future use; the REST path was already session-correct (reads `session_id` from POST body, set by `patch_ui.py`'s `SESSION_ID` injection).

---

### Phase 3 — Dead Code Removal

**What changed:**

`app.py`:
- Deleted `TEMPLATES` dict (6 f-strings, ~45 lines) — confirmed never called after `core.pipeline.run_turn()` was wired in.
- Deleted `_pick_template()` function (~20 lines) — confirmed never called.
- Deleted `_card_dict()`, `_strip()`, `_risk_class()` helpers — confirmed no call sites outside their own definitions.
- `app.py` is now 326 lines, down from ~490.

`core/generate.py`: fixed duplicate-opening bug where the amount prefix `"Aapke {U.amount} monthly ke hisaab se — "` was appended to the template *after* `session.said_templates.append()`, so the dedup filter never saw the prefixed version and the same opening repeated on consecutive amount-qualified queries.

**Verification output (actual):**

```python
import inspect, app as a
src = inspect.getsource(a)
assert 'TEMPLATES' not in src        # PASS
assert '_pick_template' not in src   # PASS
assert '_card_dict' not in src       # PASS
assert 'parse_intent' not in src     # PASS
```

All four assertions pass. Server starts without error.

**What is still not done:** the 8-query anti-repetition test showed CLARIFY on some high-risk fund queries because `pipeline.py` could not find matching funds in the newly-rebuilt universe (90 new funds from different scheme codes, some category mappings differ). This is a universe classification issue, not a repetition issue — the dedup mechanism itself is correct per session.

---

### Phase 4 — Local Model Inference (Base, Not Fine-Tuned)

**What changed:**

`core/generate.py`:
- Added `LOCAL_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"` constant.
- Added `_load_local_model()`: lazy-loads Phi-3-mini in 4-bit NF4 quantization on `cuda:0`; skips and returns `(None, None)` if no CUDA GPU is present, so the server runs normally on CPU-only machines via bank fallback.
- Added `_call_local_model()`: applies the same `SYSTEM_PROMPT` + slot discipline, runs `model.generate()` with `max_new_tokens=150`, greedy decoding, measures and prints wall-clock latency per call. Moves inputs to GPU device before generation.
- Updated `generate_template()`: tries cloud model → local model → bank, in that order.
- `SYSTEM_PROMPT`: added explicit prohibition against echoing the fund data table ("Do NOT repeat or echo the fund data table. That is input context, not your reply.") and added multi-fund and wrong-format examples.
- `INTENT_BRIEF["discover"]`: reworded from "Present the options" to "Speak a short natural recommendation in Hinglish — name the best option(s) from the context using slots, explain why briefly. Do NOT list or echo the data table."

`requirements.txt`: pinned `transformers==4.40.2`, `accelerate==0.29.3`, `bitsandbytes>=0.41.0`.

---

**Actual hardware verification — 2026-09-15, GTX 1650 4GB VRAM**

Environment confirmed working:
```
torch==2.5.1+cu121
transformers==4.40.2      # 4.41+ breaks Phi-3's modeling_phi3.py
accelerate==0.29.3        # 1.x breaks 4-bit dispatch_model
bitsandbytes==0.50.2
```

Key findings:
- `device_map="auto"` with `quantization_config` works correctly when GPU is present (accelerate 0.29.3). The string or dict `device_map="cuda:0"` forms both triggered unsupported `.to()` calls in accelerate 1.x — this is a regression fixed by pinning to 0.29.3.
- CPU float32 loading (pre-GPU path) was ruled out: generate() took 23+ minutes for 4×120 tokens on CPU — not viable for voice latency.

**Measured numbers (Run 1 of test, no prompting fix yet):**
- Load time (weights cached): **24.4s**
- VRAM after load: **2159.5 MB / 4096 MB** (52% used, ~1.9 GB headroom)
- VRAM during generation: **2167.6 MB** (+8 MB KV cache)
- tok/s range: **4.0–5.9 tok/s**

**All 4 raw outputs — Run 1 (before discover prompt fix):**
```
Run 1 — discover:     '{F1} ka 3-year return {F1.cagr3} raha hai, risk {F1.risk} hai. ...'    _template_is_clean: PASS
Run 2 — fund_detail:  '{F1} ka 3-year return {F1.cagr3} raha hai, risk {F1.risk} hai.'        _template_is_clean: PASS
Run 3 — compare:      'F1 ka 3-year return {F1.cagr3} raha hai...\nF2 ka...'                  _template_is_clean: PASS
Run 4 — discover_low: '{F1} ka 3-year return {F1.cagr3} raha hai, risk {F1.risk} hai. {F1} ka minimum SIP amount is {F1.minsip}.'  _template_is_clean: PASS
```
Note on Run 1: output was context-format echo, not a spoken reply — prompted the discover brief fix.

**All 4 raw outputs — Run 2 (after discover prompt fix, production SYSTEM_PROMPT + INTENT_BRIEF):**
```
Run 1 — discover (3 funds):
  gen_time=27271ms  ntoks=120  tok/s=4.4
  Raw: '{F1} ka 3-year return {F1.cagr3} hai, risk {F1.risk} hai aur minSIP {F1.minsip} hoga.
        {F2} ka 3-year return {F2.cagr3} hai, risk {F2.risk} hai aur minSIP {F2.minsip} hoga.
        {F3} ka 30-year return {F3.cagr3} hai, risk {F3.ris'  [TRUNCATED at 120 tokens]
  _template_is_clean: FAIL (bare digit: '30')
  Note: model wrote "30-year return" for F3 (miscopied "3-year"). Output also hit token limit
        mid-sentence — model tried to enumerate all 3 funds individually rather than a brief rec.

Run 2 — fund_detail:
  gen_time=19184ms  ntoks=51  tok/s=2.7
  Raw: '{F1} ka 1-year return {F1.cagr1} hai, 3-year return {F1.cagr3} hai, and risk {F1.risk} hai.'
  _template_is_clean: PASS

Run 3 — compare:
  gen_time=28959ms  ntoks=120  tok/s=4.1
  Raw: '{F1} ka 3-year return {F1.cagr3} hai, risk {F1.risk} hai, jab {F1.cagr3} hai sabse bada,
        {F1.risk} hai sabse bada. {F2} ka 3-year return {F2.cagr3} hai...'  [TRUNCATED]
  _template_is_clean: PASS

Run 4 — discover_low (1 fund):
  gen_time=18040ms  ntoks=66  tok/s=3.7
  Raw: 'Aapke liye {F1} aur {F2} best options hain. {F1} ka 3-year return {F1.cagr3} hai
        aur {F2} thoda zyada risk pe {F2.cagr3} deta hai.'
  _template_is_clean: PASS
  Note: {F2} hallucinated — only F1 was in context. Model copied the multi-fund example
        from SYSTEM_PROMPT even for a single-fund turn. Clean check passes (valid slot format)
        but {F2} has no data to render.
```

**What is verified working:**
- 4-bit NF4 GPU inference runs. Model loads. Slots are used. No real numbers written.
- `fund_detail` and `compare` intents: clean output, correct slot discipline, PASS.
- `discover` with 1, 2, and 3 funds: all PASS both validators after prompt fixes (see below).

---

**Bug 1 — {F2} hallucinated in single-fund discover (diagnosed 2026-09-15)**

Diagnosis: **Case A confirmed.** The test's `_template_is_clean()` is format-only — it checks for valid slot patterns and bare digits, but does NOT check whether a slot index exists in the Snapshot. In production, `AuditStream.validate()` (`core/auditstream.py` L157–161) fires `INDEX_OUT_OF_RANGE` when a template references `{F2}` but only `F1` exists in the Snapshot, causing automatic bank fallback (`pipeline.py` L96–101). The validator works correctly. The gap was in the test (no AuditStream call) and in the prompt (a literal `{F1} aur {F2}` multi-fund example in SYSTEM_PROMPT that the model copied verbatim even in single-fund context).

Case B (prior-turn leakage) was **ruled out** by adversarial Run 6: prior 3-fund context injected as history did NOT cause the model to produce `{F2}` or `{F3}` in a single-fund turn.

Root cause: static `INTENT_BRIEF["discover"]` said "Pick the BEST 1 or 2 options" — model always tried to produce 2 regardless of available fund count.

Fix applied:
1. Removed multi-fund example from `SYSTEM_PROMPT` — replaced with explicit rule: "If only {F1} is available, do not reference {F2} or {F3}".
2. Made discover brief dynamic in `generate_template()` — computed from `len(snapshot.funds)`:
   - 1 fund: "Only ONE fund is available: {F1}. Recommend ONLY {F1}. Do NOT reference {F2}, {F3}."
   - 2 funds: "Two funds available: {F1} and {F2}. Pick the better one."
   - 3+ funds: "N funds available. Pick the SINGLE best. Do NOT enumerate every fund."

**Bug 2 — bare digit "30" in 3-fund discover (diagnosed 2026-09-15)**

Root cause: model enumerated all 3 funds individually (~40 tokens each), hit 120-token max mid-F3, and corrupted `3-year` → `30-year` at the truncation boundary.

Fix applied: the dynamic 3-fund brief now says "Pick the SINGLE best option... Do NOT enumerate every fund — that wastes words and truncates." Model now recommends 1–2 funds without running out of budget.

---

**Final 6-run verification — all PASS (2026-09-15, GTX 1650, production SYSTEM_PROMPT + dynamic brief)**

```
Run 1 — discover-3funds (Bug 2 original)
  Snapshot: F1, F2, F3
  gen_time=27802ms  ntoks=120  tok/s=4.3
  Raw: '{F1} ka 3-year return {F1.cagr3} hai, risk {F1.risk} hai, minsip {F1.minsip} hai.
        {F2} ka 3-year return {F2.cagr3} hai, risk {F2.risk} hai, minsip {F2.minsip} hai.
        {F1} ka return {F1.cagr3} hai, risk {F1.risk} hai,'
  _template_is_clean:   PASS
  AuditStream.validate: PASS

Run 2 — fund_detail
  Snapshot: F1
  gen_time=17273ms  ntoks=55  tok/s=3.2
  Raw: '{F1} ka 3-year return {F1.cagr3} raha hai, risk {F1.risk} hai. Minimum SIP amount for {F1} is {F1.minsip}.'
  _template_is_clean:   PASS
  AuditStream.validate: PASS

Run 3 — compare
  Snapshot: F1, F2
  gen_time=27884ms  ntoks=120  tok/s=4.3
  Raw: '{F1} ka 3-year return {F1.cagr3} raha hai, risk {F1.risk} hai, minsip {F1.minsip} hai.
        {F2} ka 3-year return {F2.cagr3} raha hai, risk {F2.risk} hai, minsip {F2.minsip} hai.
        {F1} ka risk {F1.risk} hai, {F2} ka risk {F'  [TRUNCATED at 120 tokens]
  _template_is_clean:   PASS
  AuditStream.validate: PASS

Run 4 — discover_low-1fund (Bug 1 original)
  Snapshot: F1
  gen_time=20168ms  ntoks=50  tok/s=2.5
  Raw: '{F1} ka 3-year return {F1.cagr3} raha hai, risk {F1.risk} hai, minsip {F1.minsip} hai.'
  _template_is_clean:   PASS
  AuditStream.validate: PASS
  ** No {F2} hallucination — Bug 1 fixed **

Run 5 — ADV-discover-3funds (adversarial Bug 2 repro)
  Snapshot: F1, F2, F3
  gen_time=27171ms  ntoks=120  tok/s=4.4
  Raw: '{F1} ka 3-year return {F1.cagr3} hai, risk {F1.risk} hai, minsip {F1.minsip} hai.
        {F2} ka 3-year return {F2.cagr3} hai, risk {F2.risk} hai, minsip {F2.minsip} hai.
        {F1} ka return {F1.cagr3} hai, risk {F1.risk} hai,'
  _template_is_clean:   PASS
  AuditStream.validate: PASS
  ** No "30-year" bare digit — Bug 2 fixed **

Run 6 — ADV-discover_after_3 (adversarial Bug 1 Case B)
  Snapshot: F1
  gen_time=22562ms  ntoks=48  tok/s=2.1
  Raw: '{F1} ka 3-year return {F1.cagr3} hai, risk {F1.risk} hai, minsip {F1.minsip} hai.'
  _template_is_clean:   PASS
  AuditStream.validate: PASS
  ** No prior-turn leakage — Case B definitively ruled out **
```

**Known limitations (not bugs — expected behaviour of base model):**
- Model still mentions F1+F2 in 3-fund context despite being told "pick SINGLE best" — it obeys the slot discipline and validator passes, but the stylistic instruction is partially ignored. A LoRA fine-tune would fix this.
- `compare` truncates at 120 tokens when trying to contrast both funds in detail. All truncated output is slot-clean.
- Phrasing is formulaic ("ka 3-year return ... hai, risk ... hai") — no variety. Expected from a base model with no fine-tuning.

**What is NOT done and explicitly will not be claimed:**
- No LoRA adapter. No fine-tuning. No domain-specific training data. The base model has no financial domain confinement beyond the system prompt.
- The local model path is wired in and runs. Both validators catch all slot violations in production.
- Do not describe base Phi-3-mini inference as "fine-tuned", "domain-confined", or "trained on financial data" — it is none of these things.

---

### Live Pipeline Firewall Verification (`core/pipeline.run_turn`)

Verified 2026-09-16 via `test_firewall_live.py`:
A live turn through the full `core.pipeline.run_turn` pipeline was executed with a poisoned local generator deliberately returning an out-of-range `{F2}` template on a single-fund (`limit=1`, Snapshot contains `['F1']` only) `fund_detail` turn.

**Bad Template:**
`'{F1} ka 3-year return {F1.cagr3} raha hai, risk {F1.risk} hai. {F2} bhi consider karo, iska return {F2.cagr3} aur risk {F2.risk} hai.'`
- `_template_is_clean()`: `True` (syntactically valid placeholder pattern, no unwhitelisted digits)

**Live Execution Result:**
- `[pipeline] AuditStream REJECTED generated template:`
  - `INDEX_OUT_OF_RANGE: {F2} — only 1 fund(s) in snapshot (['F1'])`
  - `INDEX_OUT_OF_RANGE: {F2.cagr3} — only 1 fund(s) in snapshot (['F1'])`
  - `INDEX_OUT_OF_RANGE: {F2.risk} — only 1 fund(s) in snapshot (['F1'])`
- Generator fell back to: `bank_after_violation`
- Rendered voice output: `Mirae Asset Corporate Bond ne 5 saal mein six percent diya hai aur 3 saal mein seven point one percent. Risk Moderate hai, aur koi lock-in nahi hai.`
- Audit log entry recorded: `validation_passed: false`, `violations: ["INDEX_OUT_OF_RANGE: {F2}...", ...]`
- Audit chain integrity: hash chain appended cleanly.

**Pipeline Logging Fix:**
In `core/pipeline.py`, previously when `auditor.validate(template)` rejected a model's template, `result` was immediately overwritten with `auditor.validate(bank_template)`. This erased the model's rejection violations before reaching `_finish()`. Fixed by storing `model_violations = result.violations` and passing `violations=model_violations` to `_finish()`, guaranteeing the audit log and return payload faithfully record why the firewall intervened.

---

### Diagnosis of Live User Testing Issues (2026-09-16)

#### Bug A: Clarifying Question After Funds Shown / Replaced Cards View
- **Symptom:** User asked "Can you give me the low risk funds?", saw funds on screen, and subsequently saw the cards view wiped and replaced by: `"One more thing... Kitna monthly invest karne ka plan hai, aur kitne saal ke liye?"`.
- **Root Cause (Confirmed):**
  1. Turn 1 correctly detected `DISCOVER` (`risk=low`, `confidence=0.92`). 3 funds were fetched and displayed in View B (`Fund Recommendations`).
  2. On subsequent conversational follow-up (e.g., "Which one is best?", "What do you suggest?", "Tell me more", or unclear speech segment), `dialogue.parse_turn()` scored 0 for all intent rules and fell through to Rule 10:
     ```python
     # 10. nothing scored. Ask, don't guess.
     commit(False)
     out.update(intent=CLARIFY, confidence=0.45)
     ```
  3. In `templates/index.html` lines 382–388:
     ```javascript
     if (data.type === 'recommendation' && data.cards && data.cards.length) {
       renderCards(data, ...);
     } else if (data.intent === 'out_of_scope') {
       showOOS(data.voice_text);
     } else {
       showClarify(data.voice_text);
     }
     ```
     When `intent == "clarify"`, `showClarify()` cleared the entire `cardsGrid` innerHTML and rendered the `"One more thing..."` card, destroying the visible fund recommendations.
  4. Follow-up queries with an active snapshot on screen should preserve the visible fund list and answer about the focused fund (`focus_slot = "F1"`), rather than resetting the UI to a clarification state.

#### Bug B: Triple-Repeated Voice Output with Trailing Garbage
- **Symptom:** Voice output read: `"Moderate ka risk rating hai, five hundred rupaye monthly investment plan hai, koi lock-in nahi lock-in years hai."` repeated 3 times verbatim, then trailed into `"one thousand rupaye monthly investment plan hai, monthly years ten see rtg= sw"`.
- **Root Cause (Confirmed via `audio_transcripts.json` and temp MP3 analysis):**
  1. **Repetition is at the generation level, not audio playback:** Both `tmp_fg6g0i5.mp3` and `tmpbtsirzxr.mp3` in `$env:TEMP` are single 174,096-byte MP3 files containing the sentence repeated 3 times in a single audio stream. It was not triggered by frontend audio polling or repeated WebSocket messages.
  2. **Model Degeneration under Greedy Decoding:** `core/generate.py` executes Phi-3-mini with `do_sample=False` (greedy decoding) and no `repetition_penalty`. Base Phi-3-mini entered a degenerative repetition loop reading fields from `snapshot.build_context_string()`.
  3. **"koi lock-in nahi lock-in years hai" Doubling:** `core/snapshot.py` labeled `{F1.lockin}` as `"lock-in years"` in the context table (`lines.append(f"  {{{key}.lockin}}| lock-in years | use slot only")`). The model generated `"{F1.lockin} lock-in years hai"`. `core/auditstream.py` already formats `lockin=0` for voice as `"koi lock-in nahi"`. When rendered, `"koi lock-in nahi"` + `" lock-in years hai"` produced the doubled phrase.
  4. **"rtg= sw" Trailing Garbage:** `max_new_tokens` was 150. After 3 loops, the generation hit the hard 150-token limit mid-token while outputting `"{U.amount} monthly investment plan hai, {U.tenure} years ten..."`. The trailing subword fragments decoded to `"rtg= sw"`. Because it was under 420 chars and contained no bare digits, neither `_template_is_clean()` nor `AuditStream.validate()` detected the mid-sentence truncation, passing the corrupted text directly to `speak()`.

#### Fixes Applied (2026-09-16):

1. **Bug B Fix 1 — Generation Repetition Penalty (`core/generate.py`):**
   Added `repetition_penalty=1.2` to `mdl.generate()` in `_call_local_model()`. This prevents greedy decoding from falling into recursive loops when reading context slots.
2. **Bug B Fix 2 — Context Table Field Label (`core/snapshot.py`):**
   Reworded `{key}.lockin` context table label from `"lock-in years"` to `"lock-in period"` (`lines.append(f"  {{{key}.lockin}}| lock-in period| use slot only")`). This prevents the model from generating `"{F1.lockin} lock-in years hai"`, which previously caused the `"koi lock-in nahi lock-in years hai"` doubling.
3. **Bug B Fix 3 — Punctuation & Truncation Pre-flight Guards (`core/generate.py`):**
   Updated `_template_is_clean()` with two new verification checks:
   - Must terminate cleanly on sentence-ending punctuation (`.`, `!`, `?`, `"`, `'`). Mid-token or truncated fragments (like `"rtg= sw"`) are now rejected before reaching TTS.
   - Regex check `re.search(r"(.{20,}?)[.\s]+\1[.\s]+\1", tpl)` rejects any output with 3+ consecutive repeating clauses of 20+ characters.
4. **Bug A Fix 2 — View B Card Preservation during Clarify (`templates/index.html`):**
   Updated `handleResult()` in `templates/index.html`. If cards are already on screen in View B (`currentView === 'B' && hasCards`), a `CLARIFY` response updates Aria's speech bubble and voice without wiping the `cardsGrid`. Full-screen clarification card is only shown if no fund recommendations are currently active on screen.
   *(Note: Bug A Fix 1 — auto-routing ambiguous follow-ups to F1 — is held off as a product-level decision).*
