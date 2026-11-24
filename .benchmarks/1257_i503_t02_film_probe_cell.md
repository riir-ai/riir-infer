# Bench 1257 — plan 503 T0.2: the fixed-coeff FiLM probe cell (folded 27B, matvec-wrap seam)

**Verdict: MEASUREMENT CURIOSITY (the pre-registered T0.2 kill language).** The
seam engages (48/48 layers' census rows moved) but the geometry shift at fixed
coefficients is 1e-3-class r_ent from a γ±0.038 / β±0.008 perturbation — the
folded GDN state geometry is ROBUST to input-side fixed-coeff FiLM. Phase 2
does NOT launch on census movement alone.

## Posture

- Model `Ternary-Bonsai-2-27B-PQ2_0.gguf` (64 layers / 48 GDN / n_embd 5120 /
  folded=true), corpus `../riir-train/data/chat_probe`, n=512, cap_every=64
  (8 capture positions), CPU ternary (4090 box, release build).
- Build: riir-infer develop `b6f1954` + `1b8e20d` (the T0.2 matvec-wrap hook +
  the bin wiring), features `lattice_census,bonsai2_hadamard`.
- Arm: `--film-seed 0x50E1C5` → layer 0 (first GDN), resident-plane address
  keyed on its in_proj_qkv/in_proj_z, γ±0.038 β±0.008 (LCG→sigmoid-bounded,
  deterministic), Rβ from the rotation's sign vector (n_embd 5120).

## Engagement proof (rides the driver, `b6f1954`)

A fresh UNARMED pos-1 forward vs a fresh ARMED pos-1 forward on the armed
layer's state, to_bits-compared: **DIFFERENT** — the seam engaged. An
armed-but-unchanged census would have refused the run loud.

## Read-out (armed vs the bench-1249 baseline, same box/corpus/posture)

- mean r_ent (all layers × points): **3.3437** vs baseline **3.3442** (−0.0005,
  −0.015% relative).
- Per-layer (mean over the 8 capture points; baseline
  `i503_lattice_census_baseline.json`, armed
  `i503_lattice_census_t02_film_armed.json` in this directory):
  - **48/48 layers have at least one r_ent row moved** at the report's
    6-decimal precision — the injection propagates through the whole stack.
  - The ARMED layer 0 itself: −0.000010 mean r_ent (its own state barely
    moves — the input-side perturbation washes out in the state update).
  - Downstream: 1e-4..2e-3 class, both signs, no monotone decay. Largest:
    layer 23 −0.00199 r_ent, layer 33 −0.00193, layer 45 +0.00236 r_stable.
- Digests: armed `21648cf2bee68a0af5a6a94d0cca968655edb3d20ac7f4c64af0676502153889`
  ≠ baseline `255a2395…` (the bench-1249 pin) — the runs genuinely differ.
- Census pass wall: 708.0 s (CPU ternary) + 67.2 s load + 2 engagement
  forwards. Box state: 4090 Windows workstation, CPU-only cell (no GPU arm —
  the census lane is deliberately CPU ternary per the sink_audit family
  shape), no concurrent heavy jobs at run time.

## Adjudication (the pre-registered T0.2 falsifier, plan 503 L28)

"if fixed-coeff conditioning moves the census but the plan has no path to
accuracy evidence, the mechanism is recorded as a measurement curiosity —
Phase 2 does not launch on census movement alone."

- The census MOVED (the probe is honest — a broken seam would read unchanged).
- The movement is SMALL (1e-3-class from a ±5%-class γ) — the folded GDN state
  geometry is robust to fixed-coeff conditioning at probe amplitude.
- Phase 2 (trained lattice coefficients) is NOT launched by this evidence.
  Its premise (trained coefficients can be larger/aligned where fixed ones
  are not) stays untested and owner-gated as before. The lane's T0 phase is
  COMPLETE: T0.1 census + baseline (bench 1249), T0.2 seam + probe cell (this
  record), T0.3 design record (plan doc). Phase 1+ stays unclaimed.
- Re-open trigger (unchanged from the plan): a Phase-2-class trained-coefficient
  cell with its own accuracy path — never a census row alone.

## Addendum (same lease, 2026-11-24T09:0xZ local): the query-gain capture arm landed — g1 rows real

riir-infer `496e90e`: `LatticeQueryGainCapture` (the `sink_audit_q` mirror) —
the driver arms the SNAPSHOT positions; two cfg write-sites (ternary layer
body + dense main body, both `is_linear`) copy the transient `q_normed` into
the slot for pos+1 right after the GDN layer returns, so G1 pairs the top
singular vector with the query that produced it. 12 lib tests green (new
pairing fixture). The census bin arms the capture unconditionally: the report
flips to `query_armed=true`, `g1=na` rows GONE.

Armed-cell rerun (same seed 0x50E1C5, same posture — the g1 column joins the
comparison above apples-to-apples): 18,432 rows, ZERO g1=na, all values in
[0, d_k=128]. **g1 mean 1.105, median 0.195, max 115.8** — heavily
right-skewed: most heads' states are weakly query-aligned, a minority
strongly aligned; armed layer 0's own mean 0.741 (same class as the rest).
14 zero rows (zero queries / exact orthogonality). r_ent/r_stable halves
byte-consistent with the 1257 run (mean 3.3437 — the capture is
observation-only, the forward untouched). Digest
`76ad5c8783dfc0baab63fa2651fa783db1341e263c4fc899bc1e55b70f2725ff` (differs
from 1257's — the g1 column populated). Census pass 1254.5 s + 157.2 s load
(this box also carried the gitsync sweep's disk load during the pass).
Artifact: `i503_lattice_census_t02_film_armed_g1.json` in this directory.

**T0 era residue: NONE.** T0.1's three closed forms are complete on the real
checkpoint (r_ent / r_stable bench 1249, g1 this addendum); T0.2's probe cell
+ verdict stand (measurement curiosity); T0.3's design record stands. The
unarmed-baseline g1 rerun (completing the BASELINE row's g1 half) is the one
optional follow-up — the armed-vs-baseline g1 delta reads only after it.
