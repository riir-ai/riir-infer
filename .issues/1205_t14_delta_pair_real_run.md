# Issue 1205 — Issue 022 T1.4 residual: the REAL parent-vs-quantized ΔS pair run (the removed issue's deferred checkbox)

**Status:** OPEN — the residual recorded at riir-infer `7c5b5ef`'s removal note ("T1.4's REAL parent-vs-quantized ΔS pair run stays deferred until a quantized laya-class checkpoint + the T1.2 GDN capture half exist; the module `src/twt/delta.rs` + its localization gate are landed; only the real-pair run waits"). **Both prerequisites now exist**: the T1.2 laya capture half landed long since (`Encoder::forward_capture` + `examples/twt_laya_profile`), and a quantized laya-class checkpoint NOW exists on this box (amo) — the english q8 storage artifact (`~/.cache/riir-reflex/laya/english/derived/model.q8.safetensors`, converted 2026-11-11 by `laya-quant8`, sidecar BLAKE3 verified, 53.1% — the D2a posture Bench 0047 adopted).

**Claimed:** amo-idle-i1197-p2, 1794415195, expires 2026-11-13T16:39:55Z

## The run (read-only diagnostic — never a serving path)

- **Parent arm:** english F16 → S matrices over the frozen calibration corpus.
- **Quantized arm:** english q8 (the derived artifact, env-selected through the sidecar-verified resolver) → S matrices over the SAME corpus, SAME stride — the Q8 payloads widen per-tensor through the identical `widen_q8_0` arithmetic the serving posture resolves with, so this IS the storage-tier numerics (Bench 0047's posture), not an in-memory fake-quant.
- **ΔS:** `twt::delta::delta(S_parent, S_q)` → the localization report (top entries, max, RMS). The localization property is the gate: WHERE the quantization damage sits — never a collapse adjudication (that was T5.2's job, closed).

## Instrument extension (the only code change)

`examples/twt_laya_profile` loads `model.safetensors` directly today — it cannot see the q8 artifact. Extend it to resolve through `q8_artifact::resolve_weights_posture` (the same env + sidecar-verify path every consumer uses; an env typo fails loud, never falls back), and stamp `weights_path` / `weights_posture` / `weights_blake3` into the artifact JSON next to `corpus_blake3`.

## Corpus disclosure (pre-run)

The m3-local corpus `.raw/twt/audition_calib.txt` (BLAKE3 `ea2773689e1a5a3dd6f28b95948aff5d45543058cb7b3e0f5811e1665824c680`, the T5 GOAT corpus) was never committed and does not exist on amo or bmo; the 4090's copy (`E:\git\riir-infer\.raw\twt\audition_calib.txt`, 289,910 bytes, 500 lines, SHA256 `a247ad26…`) hashes BLAKE3 `9f088e7b…` — a DIFFERENT byte stream than the m3 original (same size class, different content lineage; likely a divergent early copy — the T5.4 4090 session already recorded "the M3-local corpus does not exist on this box" and made its instrument box-portable instead). **The pair run needs ONE consistent corpus, not the m3 original** — every claim is within-run (parent vs quantized on identical bytes), the artifact records the corpus BLAKE3 it actually used, and NO number here is comparable to T5's m3 rows. Disclosed, never silently pooled.

## Tasks

- [ ] T1 — the instrument extension (variant resolution + artifact provenance), `cargo clippy -p riir-infer-laya --all-targets --features laya-riir -- -D warnings` clean.
- [ ] T2 — both arms execute on amo (CPU posture; the capture driver is calibration-only), artifacts in `.raw/twt/`, corpus + weights BLAKE3s recorded.
- [ ] T3 — the ΔS report + the bench record (`twt_delta_pair` class, next `.benchmarks/` number): top localizations, max, RMS, the corpus/weights provenance + the corpus-divergence disclosure.
- [ ] T4 — HISTORY.md residual row (the 7c5b5ef note's debt closed), issue closed + removed.
