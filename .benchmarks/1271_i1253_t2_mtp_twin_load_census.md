# Bench 1271 — Issue 1253 T2: the `-mtp` GGUF load census (typed outcome + the loader's MTP-twin class)

**Status:** COMPLETE — all arms green (2026-11-26, session zed-c133e475@4090).
**Issue:** riir-infer `.issues/1253` T2 (the 4090 lane; T3/T4 stay with riir-ai 1263's GPU-leg lane).
**Box:** shikuwa (4090, i7-13700K, 31.8 GB RAM, AC) — CPU-only cell; release load 119.67 s.
**File of record:** `Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf` — PrismML `Ternary-Bonsai-2.5-27B-GGUF`
(Apache-2.0), 7,492,338,944 bytes, sha256 `ededc9fc0a8a6a8dd30914d107b2d2ce7674d789f1d05a1a481f240609aa6c23`
(verified at download; restore = HF resolve URL + the sha).

## The header census (the real twin)

| field | value |
|---|---|
| `qwen35.block_count` | 65 (64 main + 1 MTP block) |
| `qwen35.nextn_predict_layers` | **1 — DECLARED** |
| tensor count | 866 = 851 (main, byte-count identical to the plain 2.5 file) + **15 at `blk.64`** |
| `blk.64` contents | 8 projections quantized **Q4_0** (type 2: attn_q/k/v/output, ffn_gate/up/down, `nextn.eh_proj`) + 7 F32 (norms + `nextn.enorm/hnorm/shared_head_norm`) |
| open-time type check | clean — `GgmlType::from_id(2) = Q4_0` is known (the dspark/BF16 lesson does not recur) |

The draft head is a **separate component with its own quantization** (Q4_0) riding the
league-format main stack — the T5 lane's consumption target, never a main-stack layer.

## The loader's typed outcome (what T2 asked for)

The unscoped main-stack load of the twin now lands in a NAMED class:

- **DECLARED (the real twin)** — loads; `n_main = 64`; the draft block never enters the
  main stack; ONE loud disclosure line names the count, the block range, the declaration,
  and the consuming lane:
  `[gguf_loader] MTP-twin file: 15 draft-head tensor(s) at block indices >= 64 are present
  but NOT consumed by this main-stack load (qwen35.nextn_predict_layers = 1); the draft
  head stays for the MTP lane (Issue 1253 T5).`
  **Proof cell:** `real_mtp_twin_full_load_proof` (release, `--ignored`) — 64 layers,
  G1 bit-plane invariants hold, rotation parses, 119.67 s wall. Never a silent partial load.
- **UNDECLARED** — `.nextn.*`-named tensors present with no `nextn_predict_layers`:
  **refuses loud** (the Issue-742 T9.2 garbage class — the trailing block would be typed
  and consumed as main-stack layers). Gate `mtp_undeclared_refuses_loud`.
- **UNCOVERED** — a declaration that does not cover the highest trailing block: refuses
  loud (synthetic-gate covered by the census predicate; the declared arm pins the pass).
- **CLEAN** — no trailing block, no nextn names: census empty, load byte-identical to the
  pre-change path (no false positive; the whole existing suite is the regression fence).

## Gates + posture

- `tests/mtp_twin_load_gates.rs` (NEW, `required-features = ["deltanet_ternary_inference"]`):
  3/3 green — declared loads (census fields pinned: declared_nextn 1, n_main 4, trailing 3,
  nextn_names 2, covered), undeclared refuses (refusal names the `nextn.*` class + the
  declaration), clean empty (covered, n_main intact).
- `tests/mtp_twin_real_file.rs` (NEW): header census 1/1 green on the real twin
  (declared 1 / n_main 64 / trailing 15 / nextn_names 4 / covered); the full-load proof arm
  `--ignored` at release, GREEN 119.67 s.
- Regression: `bonsai2_rotation_load` 11/11, `issue028_disaggregated_handoff` 9/9 (+2
  ignored = the 6.7 GB PQ2_0 arm, unchanged), lib 404/0 (+bonsai2_hadamard), 404/0
  (ternary-only), 310/0 (default).
- Clippy `-D warnings` clean ×3 postures (default / `deltanet_inference` /
  `deltanet_ternary_inference` + `bonsai2_hadamard`, `--all-targets`); the all-targets
  warning rows are byte-identical to the pre-change baseline (pre-existing, not this lane's).

## Design notes

- The census is a **header + tensor-table read, never a weight read** (`mtp_twin_census`).
- The undeclared class is detected by the **`nextn` NAME COMPONENT** (`.nextn.*` names with
  no declaration), not by block index alone: an undeclared trailing block with no nextn
  names is indistinguishable from a legal larger model (and IS legal) — the garbage risk
  is the draft head the config path does not know about.
- The disclosure lives on the load path (one eprintln, ONCE per load); the typed surface
  (`MtpTwinCensus` + `covered()` + `refusal()`) is the programmatic/assertable half.
- API surface: additive only — `MtpTwinCensus`, `mtp_twin_census`, one gate call inside the
  existing scoped loader. No return-type change; callers (disaggregated, dist lane) are
  untouched.
