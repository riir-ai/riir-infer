# Issue 1233: the gdn_rec chunked-form prefill kernel — the Bench-1233 named residue gets its lane

**Status:** OPEN — filed 2026-11-17 from [`.benchmarks/1233_i1167_gdnrec_price_cell.md`](../.benchmarks/1233_i1167_gdnrec_price_cell.md)'s CONFIRMATORY close (planes=8 stands at every serving shape; the 6.9% residue is kernel WORK, not a planes re-tune). Scope owner: unclaimed; the lane prices at ~6.9% of prefill at the promoted posture (Bench 1227 re-rank).

## Why

Bench 1233 (the i1167 engagement+price cell) re-scored the MR planes ladder with planes=16/32 AND the serving chunk p=512: the DEFAULT planes=8 stands at every serving shape — the ladder axis is exhausted. The remaining gdn_rec ground (6.9% of prefill wall at the promoted posture) needs the **chunked intra-chunk kernel form**: the incumbent recurrence walks the chunk sequentially per token; a chunked form parallelizes within the chunk (the 1146 dist-prefill lane's chunked-class kernel work — the form Bench 1233 priced as "its own lane, not a rung"). The 6.9% share is the ceiling: any candidate must clear the bars below or the lane folds CONFIRMATORY again.

## Scope

1. **Design first (read-only):** the chunked GDN recurrence for Metal (cubecl) — intra-chunk parallel form over the serving chunk p=512, the state-journal delta-roll shape respected (`gdn_delta_roll` stays opt-in, byte-identical off), no change to the MR planes dispatch (`RIIR_PREFILL_REC_MR` ladder untouched — planes=8 stands unless this kernel's own ladder says otherwise).
2. **G1 bit-identity (mandatory, zero tolerance):** the chunked kernel must reproduce the incumbent recurrence EXACTLY at the B1 case set — the same law Bench 1233's ladder arm carried (`mr_is_bit_identical_to_multi_token`, now covering 16/32 automatically). A lossy chunked form is a DIFFERENT lane (it would need the f16b-class retention protocol, not this issue).
3. **Price cell (pre-registered):** the G2 isolated-kernel probe (`test_rec_mr_timing_probe`) gains the chunked arm, scored at the serving chunk p=512 + the E1 ladder shapes p=128/2048; interleaved median-of-9, release posture, bench_preflight PROVENANCE per the F3 standing rule. **Bar: ≥3% kernel-isolated median at p=512** graduates to the promotion lane (paired e2e A/B over pp512/pp4096/pp32k, the i1167 E-shape); under 3% → the lane closes CONFIRMATORY with the chunked form measured-out (the 6.9% residue is then priced as sequential-bound, recorded, done).
4. **Promotion (only on a pass):** the 1146/1225 lane discipline — demote-env first (`RIIR_PREFILL_GDN_CHUNKED` PROMOTE-ONLY per the env-switch law), GOAT gates, delegated-owner verdict, then the default flip inside the dist lane. The pre-registered negative prior stands: CUDA r8 arms LOST to r4w4 on sm_89 (Bench 896) — Metal's scheduler may differ; this cell answers it.

## Guards

- Engagement proof FIRST (the Issue-1124 lesson): prove the chunked arm is reached in the measured posture before quoting any number.
- The prefill f16-B twin (default-ON in the dist lane) composes: the price cell runs at the promoted posture, not a stripped one.
- Never touch the decode resident lanes (issue 1181/1217 own those seams).

Sibling context: [.issues/1167_metal_prefill_followup_lane_b_c.md](1167_metal_prefill_followup_lane_b_c.md) (the parent lane; its rung ground is RESOLVED — this issue is the named follow-on, not a rung).
