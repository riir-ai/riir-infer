# Issue 1210: rope.rs module-header doc describes a pairing the code does not use

**Status:** open — filed 2026-11-12; one-line doc fix. Found by the lens PoC (katgpt-rs Issue 999, T2a) while pinning `rope_vjp` against the forward.
**Claimed:** amo-zed-1210@amo (amo), 1794499469, expires 2026-11-14T16:04:29+00:00

- [ ] Fix the `src/rope.rs` module-header doc comment: it says "For each pair (`q[2i]`, `q[2i+1]`)" (adjacent/interleaved pairing), but EVERY rotation path in the file — `apply_rope_heads_precomputed`, `apply_inverse_rope_heads_precomputed`, `apply_rope_values`, `apply_rope_with_freq`, `apply_partial_rope_with_freq`, and the RoVE inverse (Plan 557 T3.B) — uses the **rotate-half pairing `(vec[i], vec[i+half])`** (HuggingFace `rotate_half` convention; line-256 comment states it correctly). Rewrite the header example to the rotate-half form so nobody ports the wrong pairing (the lens PoC's `lens::ops::rope_vjp` follows the CODE and round-trip-tests against it; a future reader trusting the header would break both).

**Evidence:** header lines ~5-13 vs `apply_rope_heads_precomputed` (line 383+: pairs `vec[i]` with `vec[i + half]`) and line 256's own "matching HuggingFace's `rotate_half`" note.
