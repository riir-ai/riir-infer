# Issue 1190 — the i1188 T3 ranked sensitivity cell (rig job): the 16-family Δ-NLL ranking on Bonsai-2-27B-PQ2_0

**Status:** OPEN — filed 2026-11-09 (zed-2d47a6cb@amo, right after the i1188 T0+T1 landing `45778ee`). Pre-registered protocol; the rig run queues behind the i1117 exclusive job on bmo (the rig law: never stack a measurement cell on a measuring rig). Execution vehicle: `agent-mail rig run bmo --exclusive` OR 4090 (the model must transfer there — the i1177 chunked-scp lesson applies).
**Claimed:** zed-c21902cf@amo, 1765484700, expires 2026-11-14T20:35:00Z
plan 631 files the job; the compute parks on the bmo rig queue behind i1117 (never stacked).
(Status line above: the original filer's lease expired 2026-11-11T09:30Z with zero rows landed — a normal re-claim.)

## What runs (pre-registered — no re-tuning after the first row)

The T3 cell of riir-infer Issue 1188: ONE ablation run per per-layer family on
`Ternary-Bonsai-2-27B-PQ2_0.gguf` (sha256-pinned below), the candidate posture
Q8_0 quant-resident, everything else at the incumbent. The tool is
`si_zero_shot_ppl --family-ablate` (landed `45778ee`, plumbing PROVEN on the
amo fixture cell — ffn_gate, pooled Δ-NLL −5.171e-4 nats/token, 0/128 argmax
flips, 64 tensors 2.12→8.50 bpw).

**The 16 families (v1 surface; `token_embd`/`output` refuse loud — bare
`TernaryGroupWeights` globals, recorded scope decision in 1188):**
`attn_q, attn_k, attn_v, attn_out, attn_gate, gdn_qkv, gdn_alpha, gdn_beta,
gdn_out, ffn_gate, ffn_up, ffn_down, gdn_conv, gdn_dt, gdn_a, norms`.

**Pinned inputs:**
- Model: `../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` — sha256
  `e00082f7…` (the i1177-pinned file; verify at job start: `shasum -a 256`,
  refuse on mismatch — never measure a different quantization run, the 1177
  dbirks lesson).
- Corpus: `/tmp/i1188_corpus.txt` (28× the market prose paragraph, 29 036
  bytes, sha256 `a6998c740ab43d21eaa127e699d1aff042cba41b4df8aea46e7f05b40712fcbc`)
  — ONE frozen corpus across every family and every re-run; regenerate
  verbatim from the committed recipe if missing (the paragraph is in the rig
  job script, byte-exact).
- Shape: `--max-length 2048 --min-length 128 --score-window 512` (prefix 2048,
  512 scored positions — the amo fixture used 256/128 for plumbing only).

**Per-family row (the report fields the tool already prints):** pooled Δ-NLL
nats/token (golden oracle), mean |Δ|, max |Δ|, argmax flips /512, the fp32-
and f64-cliff band rows, the engagement-proof per-tensor table (tensor count +
bpw shift), pass times.

## The gates

- **Engagement:** every family's report carries `tensors > 0` (16/16 present)
  and the per-tensor rows name the right tensors (family slug in every
  `blk.N.` row). A family with zero rows is a REFUSAL, never a ranked 0.
- **Determinism:** re-run ONE family (`ffn_down`) at the end; its pooled
  Δ-NLL must be to_bits-identical to the first run (the tool is deterministic
  by construction — no RNG anywhere; this gate proves the box didn't inject
  variance).
- **Sanity direction:** Q8 ≥ Q2_0 precision ⇒ every PROJ/GateProj family's
  pooled Δ-NLL is expected ≤ ~0 (a large POSITIVE pooled Δ = the candidate
  HURT = investigate before ranking; the only families that may read ≈0 are
  the vector round-trips).
- **Box state:** the rig job's PROVENANCE line (quiet box, load, RAM) printed
  beside every row; `--exclusive` on bmo (no concurrent heavy job), run
  `caffeinate -is` wrapped.

## The ranked table (T2's output, owner-gated posture line)

Sort the 16 rows by |pooled Δ-NLL| descending — the families where ternary
leaves the most NLL on the table rank first. Beside the table: per-family
keep/demote LINE under the lossy-surface promotion rule (per-family
conditional retention — never aggregate perplexity alone). **The posture
decision is owner-gated; this issue files the table and never flips a
default.**

Runtime estimate: ~16 families × (load 6.4 s + pass 1 ~115 s + requant ~6 s +
pass 2 115–350 s family-dependent) ≈ 1.5–2.5 h on bmo-class silicon at prefix
2048.

## Repro (the exact rig job body)

```sh
FEATURES="scale_invariant_attn,bonsai2_hadamard,golden_nll"
cargo build --release --features $FEATURES --bin si_zero_shot_ppl
for f in ffn_gate ffn_up ffn_down gdn_qkv gdn_out attn_q attn_k attn_v attn_out \
         attn_gate gdn_alpha gdn_beta gdn_conv gdn_dt gdn_a norms; do
  caffeinate -is ./target/release/si_zero_shot_ppl \
    --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --corpus /tmp/i1188_corpus.txt \
    --family-ablate $f --family-posture q8 \
    --max-length 2048 --min-length 128 --score-window 512 \
    > .i1190_out/ablate_$f.log 2>&1
done
# determinism gate:
caffeinate -is ./target/release/si_zero_shot_ppl ... --family-ablate ffn_down ... \
  > .i1190_out/ablate_ffn_down_rerun.log 2>&1
# compare pooled Δ-NLL lines to_bits
```

⛔ Scope guards (carried from 1188): NOT a new quantizer; the posture decision
is owner-gated; citation as "riir-infer issue 1190", never a bare number
across repos.

## bmo run failure 2026-11-12 (rig job a45099bc, logged by zed-061faf21@amo housekeeping)

The first bmo exclusive run FAILED at exit 101 during staging: `failed to select a version for katgpt-core` — dependency resolution could not resolve the conflict (staged sandbox missing/mismatched sibling checkout). Log: /Users/katopz/.agent-mail/jobs/logs/01a1cc2f-6fad-7e51-bfb3-7297a45099bc.log on bmo. The lease-holder's inbox (zed-c21902cf@amo) is session-less (boss housekeeping sweep 2026-11-12); before re-queueing, fix the sandbox's katgpt-core resolution — the t2b staging pattern (SB_STAGED_OK sibling layout check, /Users/katopz/riir932/sb-amo/) resolves the same sibling deps and is the known-good staging shape.
