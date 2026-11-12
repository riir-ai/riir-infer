//! Frozen-weight adapter artifacts (Issue 1208): the deployment half of
//! post-training adapters that write no new weight values.
//!
//! One overlay class lives here so far: the deletion mask
//! ([`mask_codec`]). Hot-swap dispatch is an extension of the existing
//! overlay seam (T2), never a parallel system.

#[cfg(feature = "deltanet_inference")]
pub mod f32_apply;
pub mod mask_codec;
pub mod q2_apply;
