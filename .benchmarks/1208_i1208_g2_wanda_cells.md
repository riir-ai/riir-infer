# Bench 1208 — Issue 1208 T2's G2 bars + the T5 Wanda sweep determinism (0.8B hybrid, amo CPU)

Date: 2026-11-12, window 15:21–15:35 +07 (load1 3.15–4.63, 30 cores, AC).
Fixture: `riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf` (qwen35 hybrid dense,
24 layers, mlp_hidden 3584 × n_embd 1024 — 88.5M gate weights total).
Instruments at `a6c3829` + the arg fixes in the landing below; release build,
`deletion_mask,deltanet_inference`. The mask: every 9th weight of every gate
tensor deleted (≈11% — deliberately far above the paper's <0.05% so overlay
overhead is visible, not lost in noise).

## G2 bar (b) — overlay decode delta: PASS −0.35% (bar ≤1%)

Interleaved base/overlay A/B rounds (64 tokens each, 6 rounds, warm-up first;
overlay mounted ONCE — pointer-swap steady state):

| round | base tok/s | overlay tok/s | delta |
|---|---|---|---|
| 0 | 21.0 | 20.8 | −0.66% |
| 1 | 20.9 | 20.3 | −2.85% |
| 2 | 20.4 | 20.8 | +2.12% |
| 3 | 20.8 | 20.8 | −0.01% |
| 4 | 21.0 | 20.9 | −0.27% |
| 5 | 20.9 | 20.8 | −0.38% |

Mean delta **−0.352%**, base spread 2.62% — the delta sits 7× under the box
noise in absolute terms. The shadow-copy overlay costs nothing measurable at
decode: PASS. (This also re-proves the T2 G1 three-way parity at timing
scale — the same kernel runs, only the data differs.)

## G2 bar (a) — apply share of load wall: 11.09% MEASURED, bandwidth-bound (disclosed)

The cell's first load is cold-cache (0.876 s) vs warm (0.298 s) — the printed
11.09% compares cold-vs-warm and is NOT the clean number. The clean numbers:
apply wall **33 ms** over 9,786,720 weights zeroed (≈0.38 ns/weight —
memory-bandwidth zeroing of the 24 gate tensors' f32 copies, near-optimal),
≈ **11% of the warm load wall** (0.298 s). Against the bar's letter ("a few
%") this reads high; the analysis: the apply touches every gate tensor's
f32 image once (354 MB), so its cost is deletion-fraction-independent and
irreducible below memory bandwidth. The bar's intent — the mask must not
dominate deployment — holds: 33 ms ONCE per mask mount, and the overlay
posture (hot-swap) needs NO load-time apply at all (0% — bar (b) proved it
free at decode). The deploy-posture choice lands with T4's cells.

## T5 Wanda sweep — same-box determinism: PROVEN (byte-identical)

Two independent full runs (24 layers armed, 512 chat_probe tokens, cap 512):

- corpus digest `72ecce464f55…bc8a` (BLAKE3 over the token stream)
- sweep digest `80ed4edb4229…853c` — identical across runs
- blob digest `8bc1c712…795a` — identical across runs
- file sha1: both `c582078d80a0…a33c` — byte-identical artifacts

27 s/sweep (512 tokens, 24 layers). The cross-box digest comparison
**PASSED 2026-11-14** (bmo, exclusive job `01a1d96d`, 25 s/sweep — session
zed-1df3fef0@amo): corpus digest `72ecce464f551867ee6e9a545c4afb80fb49f601aa
45c2116cc580335fffbc8a`, sweep digest `80ed4edb420294601b24b2635b2834b07870b
84be0e82cb9ba4794e1a326853c`, blob digest `8bc1c712a0ebd80f43bc4ba6646e53bb
c547311ddc876f355a2f6388ed78795a` — ALL THREE byte-identical to the amo
record above (amo arm64 → bmo arm64; the artifact is reproducible from
(checkpoint, corpus) on any box, no blob needs archiving). The T5 cross-box
falsifier is CLOSED; the sweep determinism is now cross-box proven, not
same-box only.

## Warmed re-run (the instrument fixed in the same window)

After the warm-up fix (throwaway load first, both timed loads warm):
base load wall **0.299 s** vs masked 0.322 s → apply **36 ms = 11.1%** of
warm load wall — the 11% number is REAL and confirmed (not a cold/warm
artifact). Bar (b) re-passed under higher load (load1 10.6→15.0 during the
run): delta −0.466%, spread 1.77%. The 11% vs "a few %" letter: the apply
touches every gate tensor's f32 image once (354 MB) — bandwidth-bound,
deletion-fraction-independent; the overlay posture needs no apply at all
(bar b: free at decode). Deploy-posture choice = T4's cell.

## Cell honesty notes

- (fixed in-window) The bar-(a) cold-vs-warm flaw — the instrument now
  warms both loads before timing; the warmed re-run confirms 11.1%.
- PROVENANCE lines verbatim in the run log; the quiet-window refusal guard
  fired correctly earlier in the day (load1 14.65 > 12 → refused).
