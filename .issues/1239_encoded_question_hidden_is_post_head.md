# Issue 1239 — `EncodedQuestion.hidden` holds the post-head residual, not the documented encoder output

**Status:** IN PROGRESS — T1 verdict (a) FIX (Claude verdict, katgpt-rs-93: the field name, its doc and all four consumers assume the encoder output; re-documenting would bless a silent semantic drift).
**Claimed:** katgpt-rs-93@amo, 1795110000, expires 2026-11-22T00:00:00Z

## The finding

`RiirAgent::encode_question` (`crates/riir-infer-laya/src/laya/riir/agent.rs`) documents `hidden` as "the encoder's final hidden states, PRE-type-emb (the type-embedding add is the head's own first op)". The code runs `self.head.layers_forward_rows(.., &mut h, ..)` (which writes `h` IN PLACE: type-emb add plus both head layers' residual adds) and only THEN does `download_into(&h, &mut h_host)`. So `hidden` is the post-head residual stream.

**Measured:** against the encoder-only seam `encoder_hidden` (Issue 1237 T-R0, held patch), the max |Δ| over the first 32 forwards of the tetris v4 suite on english is **1281.6**. `marker_rows` is unaffected (it is gathered inside the head), and so is the serving forward.

## Who reads `.hidden` believing it is the encoder output

- riir-reflex `examples/pair_free_probe.rs` (mean-pooled "encoder" states)
- riir-reflex `examples/rho_suite_probe.rs`
- riir-infer `crates/riir-infer-laya/examples/dump_encoder_states.rs`
- riir-infer `crates/riir-infer-laya/tests/prefix_state_coupling.rs` (state-span row comparisons)

Any number those produced describes the head's residual, not the encoder.

## Tasks

- [ ] **T1 — decide the contract** (Claude verdict): (a) fix the code (download `h` BEFORE the head runs, matching the doc and every consumer's assumption), or (b) keep the bytes and re-document the field as post-head. (a) is the expected verdict (four consumers assume the encoder output, and the field name says so). Under (a), re-check each consumer's recorded numbers and say in its record whether they move.
- [ ] **T2 — pin it:** a test asserting `encode_question(..).hidden == encoder-only hidden` (bit-identical on CPU) so the class cannot regress.
