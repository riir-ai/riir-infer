# Issue 1249 — GDN erase-stability gate: two-cell pin (classical scalar-β bound + MemT channel-wise law)

**Status:** OPEN — hardening + adoption gate (prevention-with-proof-path; the shipped lane is contractive by construction). Companion distill: `../.research/041_MemT_Memory_Foundation_Model_Distill.md` (2-round Claude verdict AGREE, session `18b5a3da-1ab0-4916-83f1-d1f208681ea8`).

**Claimed:** zed-glm5-flash@m3, 1795518691, expires 2026-11-26T19:00:00Z

## Why

Two stability facts about rank-1 erase operators in gated-delta-rule state updates are true but unpinned in this repo:

- **Cell (a) — classical DeltaNet bound (prior art: Yang et al., GDN 2412.06464 lineage; NOT MemT's).** Eigenvalues of `I − βkkᵀ` are `{1 (×d−1), 1 − β‖k‖²}` ⇒ non-expansive iff **0 ≤ β‖k‖² ≤ 2**. Our shipped lane satisfies this by construction (scalar per-head β, `deltanet/forward.rs` `expand_l2_normalized_heads_into` + sigmoid β + L2-normalized k ⇒ β‖k‖² < 1), but the bound itself is nowhere asserted, and our own GPU fixture empirically hit the violation class: `crates/riir-infer-gpu/src/cudarc_kernels/deltanet.rs` L4955-4959 — raw fixture drove β·‖k‖² to 5..30 (eigenvalues −4..−29, state → ~1.5e7) because it bypassed the production `expand_and_l2_normalize`. A regression (someone removing the normalize, widening β's range, or feeding un-normalized keys) would reproduce that blow-up with no test red.
- **Cell (b) — MemT channel-wise law (arXiv:2610.26825 App. C; their delta on NVIDIA GDN-2 2605.22791).** A channel-wise gate b on unit-norm k makes the erase `A = I − kᵀ(b⊙k)` **non-normal**: sup‖A‖ → **2/√3 ≈ 1.1547** (peak at ρ = Σbᵢkᵢ² = 1/3; measured verification sweep this session: random 60k draws reached 1.1493; the symmetrized form `U = √β⊙k` measured exactly 1.000 — non-expansive by construction). The per-step worst-case contraction guarantee ‖AD‖ ≤ (2/√3)·d̄ < 1 is SUFFICIENT, not necessary, and couples to the forget gate: it requires **d̄ < √3/2 ≈ 0.866/token** (worst-case-guarantee half-life ≈ 5 tokens) — an unsymmetrized channel-wise erase and a long-half-life forget gate cannot both hold their design intent. **This is the pre-admission gate for the live GDN-2 league watch** (riir-refine `.research/253` gdn2 watch-note rider added same day).

## Tasks

- [ ] Cell (a): add `riir-infer/tests/gdn_erase_stability_gate.rs` — assert on the shipped scalar-β step (`gated_deltanet_step` / the chunked lane's state update) that per-step state norm is non-increasing under production postures (L2-normalized k, β ∈ (0,1), decay ≤ 1), including a sinusoid + random-walk input regime; assert the REFUSED posture (β·‖k‖² > 2, un-normalized k) is detected by the same predicate — formalizing the L4955 fixture class as a named test, citing DeltaNet as prior art.
- [ ] Cell (b): in the same file — the MemT law sweep on a **deterministic grid covering the analytic peak ρ = 1/3** (unit-norm k, b ∈ (0,1)^d, d ∈ {2, 4, 16}, ρ grid points incl. 1/3 ± small offsets — never random draws alone): assert measured sup‖I − kᵀ(b⊙k)‖ ∈ [1.14, 1.155] (approaches 2/√3 ≈ 1.1547) and measured sup‖I − uᵀu‖ ≤ 1+ε for u = √β⊙k over the same grid. **Falsification arm**: if the unsymmetrized sweep cannot reach ≥ 1.14, the law's premise fails in our parameterization — drop cell (b) and record the negative in the issue before closing.
- [ ] Run the gate on amo or bmo (m3 BUSY 2026-11-24; pure-CPU test, `cargo test -p riir-infer-core --test gdn_erase_stability_gate` — no GPU, no features).
- [ ] On close: record the verdict + numbers in `.research/041` (one line) and remove this file (ref hash in the note).

## Gate bars (pre-registered)

- Cell (a): state-norm ratio ≤ 1 + 1e-6 per step across the production-posture regime; the refused-posture arm must FAIL the predicate (proving the gate fires).
- Cell (b): unsymmetrized sup ∈ [1.14, 1.155]; symmetrized sup ≤ 1 + 1e-6. Both on the deterministic grid; seed-pinned PRNG only for the random-input regime of cell (a) (Issue-809 law).
