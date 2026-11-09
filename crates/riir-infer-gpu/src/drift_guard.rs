//! The online drift guard with exact-path fallback for the dense-f16
//! matmul2d arm (riir-infer issue 1182 — the XERJ int8-guard adoption).
//!
//! # The gap class
//!
//! The promoted serving posture (issue 1176) compiles the dense-f16
//! matmul2d kernels with `relaxed_precision = true` — gated offline
//! (per-family retention at promotion, bench-side FNV/argmax tripwires)
//! and at load time (BLAKE3 sidecars), but nothing watches the arm at
//! RUNTIME. A lossy arm that passed its offline gate can still misbehave
//! on an unforeseen input family, silently, until the next offline
//! re-measure. This module is the serving-lane counterpart of the
//! retention gate: sample a bounded fraction of dispatches through the
//! exact (`relaxed = false`) twin, sigmoid-threshold the divergence, and
//! demote a breached family at runtime to the exact path with a counted
//! fallback line.
//!
//! # Scope (issue 1182 T4 — read before wiring anywhere else)
//!
//! This is a SERVING-lane device. It samples PREFILL `gemm()` dispatches
//! of the dense-f16 arm only (the half-B F16ACT serving posture is the
//! arming target; the mixed-path-only posture declines with a logged
//! line). The 20 Hz game hot path trades tail latency for a per-query
//! cross-check — game-side consumers must keep this OFF (the feature is
//! opt-in and never joins `production`; the env is promote-only), and
//! this module must NEVER be wired into a game tick path. In riir-infer
//! itself (upstream of every game) the guard is legal anywhere the arm
//! serves.
//!
//! # The verify-against-exact spectrum (issue 1182's genus note)
//!
//! Speculative decoding verifies EVERY step against the exact path
//! (lossless by construction, `katgpt-speculative`). This guard samples a
//! bounded stride and demotes the breached family (lossy retained,
//! overhead bounded) — a different coverage/cost point on the same
//! spectrum.
//!
//! # Threshold law (issue 1182 T3 — sigmoid, never softmax)
//!
//! The sampled divergence is the relative RMS over a bounded strided
//! readback of both outputs:
//!
//! `d = ‖a − b‖₂ / (‖b‖₂ + EPS)` with `a` = production (relaxed) output,
//! `b` = exact output.
//!
//! The breach predicate is a SIGMOID projection
//! `σ((d − DIV_CENTER) / DIV_SCALE) ≥ 0.5` — monotone in `d`, bounded,
//! equivalent to `d ≥ DIV_CENTER` at the 0.5 cut but carried in the
//! sigmoid form so calibrated soft thresholds ride the same law later.
//! Constants are PINNED (the env-switch law: an opt-in feature's knobs
//! are promote-only literals; a calibration cell may re-pin them, never
//! a runtime env).
//!
//! # Family axis
//!
//! The dispatch seam sees only `(w, m, n, p)` — no weight name — so the
//! PoC families are SHAPE BUCKETS keyed `(m << 32) | n`. That is a
//! disclosed approximation of the tensor-family axis (issue 1188's
//! census families): all same-shape weights share a bucket (per-layer
//! ffn/gates collapse together across layers). A named-family wire would
//! need `QuantWeight` to carry metadata — a wider refactor the GOAT cell
//! can justify or refuse.
//!
//! # Determinism
//!
//! Sampling is a FIXED-STRIDE counter (no RNG) — the same dispatch
//! sequence samples the same dispatches on every run. The readback is a
//! bounded strided element subset (`MAX_SAMPLE_ELEMS`). Latching is
//! permanent for the process (runtime demote, counted once, loudly).
//!
//! # Cost posture
//!
//! Disarmed: the seam declines before touching anything (one
//! `OnceLock<bool>` read) — byte-identical to the incumbent path. Armed:
//! 1/`SAMPLE_STRIDE` of dispatches pay a second (exact) matmul + one
//! bounded readback sync; latched families pay the exact kernel forever
//! (the fallback IS the demotion). The state map is a
//! `std::sync::Mutex<HashMap>` — armed-path only, disclosed; the unarmed
//! path never touches it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Sample one dispatch in this many prefill gemms (fixed stride, the
/// determinism law — no RNG).
pub const SAMPLE_STRIDE: u64 = 64;

/// Sigmoid center on the relative-RMS divergence (pinned; the T3 law).
/// The relaxed f16 chain's noise floor on well-conditioned inputs is
/// ~1e-4-class; 3e-3 sits above it with ~30x margin while remaining ~2
/// orders under a wrong-answer-class drift. The T2 cell measures the real
/// floor and may re-pin WITH the cell's record (a calibrated constant
/// change, never a runtime env).
pub const DIV_CENTER: f32 = 3.0e-3;

/// Sigmoid scale (pinned; the T3 law).
pub const DIV_SCALE: f32 = 1.0e-3;

/// Relative-RMS epsilon (pinned): keeps the ratio finite on an all-zero
/// exact output.
pub const DIV_EPS: f32 = 1.0e-12;

/// Bounded strided readback: at most this many elements are compared per
/// sampled dispatch (the tail-latency bound — a full (p·m)-element read
/// would sync multi-MB scratch rows).
pub const MAX_SAMPLE_ELEMS: usize = 4096;

/// `RIIR_DRIFT_GUARD` — PROMOTE-ONLY (the env-switch law): arms only on
/// the exact literal `1`; unset/anything-else is disarmed. A demote
/// literal here would be a no-op (the feature is opt-in; there is no
/// default-on posture to demote — that flip is T5's GOAT call).
pub fn drift_guard_armed() -> bool {
    static ARMED: OnceLock<bool> = OnceLock::new();
    *ARMED.get_or_init(|| std::env::var("RIIR_DRIFT_GUARD").as_deref() == Ok("1"))
}

/// `RIIR_DRIFT_GUARD_INJECT` — the T2 tripwire posture (debug-only):
/// perturbs the SAMPLED production readback by a synthetic relative delta
/// before thresholding, so the latch/sampler machinery is provable
/// end-to-end without a physically diverging kernel. Arms only on the
/// exact literal `1`; the injection is COUNTED and named in the latch
/// line (an injected latch is a machinery proof, never a numerics
/// finding).
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
fn inject_armed() -> bool {
    static INJECT: OnceLock<bool> = OnceLock::new();
    *INJECT.get_or_init(|| std::env::var("RIIR_DRIFT_GUARD_INJECT").as_deref() == Ok("1"))
}

/// The injected synthetic relative delta (half an order of magnitude over
/// `DIV_CENTER` — the latch must fire).
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
const INJECT_DELTA: f32 = 1.0e-2;

/// Overflow-safe logistic. `|x| ≥ 40` saturates without calling `exp`.
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = -x;
        if z < -40.0 {
            return 1.0;
        }
        1.0 / (1.0 + z.exp())
    } else {
        if x < -40.0 {
            return 0.0;
        }
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// The breach predicate: sigmoid-thresholded divergence (T3 law). NaN
/// breaches — fail-closed (the house loud-zero law): a NaN divergence
/// means one output is garbage, exactly the failure class this guard
/// exists to catch; the family demotes to the exact arm rather than
/// passing as clean.
pub fn breaches(div: f32) -> bool {
    if div.is_nan() {
        return true;
    }
    sigmoid((div - DIV_CENTER) / DIV_SCALE) >= 0.5
}

/// Family key: the shape bucket `(m << 32) | n` (the module doc's
/// disclosed approximation).
pub fn family_key(m: usize, n: usize) -> u64 {
    ((m as u64) << 32) | (n as u64)
}

/// The relative-RMS divergence over two equally strided samples.
///
/// Sequential fold (the Issue-940 law — this feeds a runtime demote
/// decision, and a platform-lane SIMD reduction would fork the threshold
/// across backends).
pub fn rel_rms_divergence(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    let mut norm = 0.0f32;
    let n = a.len().min(b.len());
    for i in 0..n {
        let d = a[i] - b[i];
        acc += d * d;
        norm += b[i] * b[i];
    }
    (acc / (norm + DIV_EPS)).sqrt()
}

/// The bounded strided view of `v`, materialized into `out` (cleared
/// first): at most [`MAX_SAMPLE_ELEMS`] elements, evenly strided, first
/// element kept.
pub fn sample_strided(v: &[f32], out: &mut Vec<f32>) {
    out.clear();
    if v.len() <= MAX_SAMPLE_ELEMS {
        out.extend_from_slice(v);
        return;
    }
    let stride = v.len().div_ceil(MAX_SAMPLE_ELEMS);
    let mut i = 0;
    while i < v.len() {
        out.push(v[i]);
        i += stride;
    }
}

/// Per-family (shape-bucket) guard state.
#[derive(Debug, Default, Clone)]
pub struct FamilyStats {
    pub samples: u64,
    pub breaches: u64,
    pub fallbacks: u64,
    pub latched: bool,
    pub max_div: f32,
    /// Samples that carried the T2 synthetic injection (disclosed in the
    /// latch line so an injected proof is never misread as numerics).
    pub injected: u64,
}

/// The process-wide guard. Created once; the unarmed path never touches
/// the family map.
pub struct DriftGuard {
    stride_counter: AtomicU64,
    families: Mutex<HashMap<u64, FamilyStats>>,
    total_samples: AtomicU64,
    total_fallbacks: AtomicU64,
    exact_installed: AtomicBool,
}

impl DriftGuard {
    fn new() -> Self {
        Self {
            stride_counter: AtomicU64::new(0),
            families: Mutex::new(HashMap::new()),
            total_samples: AtomicU64::new(0),
            total_fallbacks: AtomicU64::new(0),
            exact_installed: AtomicBool::new(false),
        }
    }

    /// A fresh, isolated guard instance for test cells (doc-hidden: NOT a
    /// production surface — the process-wide [`guard`] is the real one).
    #[doc(hidden)]
    pub fn new_for_test() -> Self {
        Self::new()
    }

    /// Fixed-stride sampler: true when THIS dispatch is the stride's
    /// (dispatch 0, then every [`SAMPLE_STRIDE`]th).
    pub fn should_sample(&self) -> bool {
        let n = self.stride_counter.fetch_add(1, Ordering::Relaxed);
        n.is_multiple_of(SAMPLE_STRIDE)
    }

    /// Record one sampled divergence; returns whether it breaches (the
    /// caller then latches). `injected` marks the T2 synthetic tripwire.
    pub fn note_sample(&self, family: u64, div: f32, injected: bool) -> bool {
        let hit = {
            let mut map = self.families.lock().expect("drift-guard families lock");
            let st = map.entry(family).or_default();
            st.samples += 1;
            st.injected += u64::from(injected);
            if div > st.max_div {
                st.max_div = div;
            }
            let hit = breaches(div);
            if hit {
                st.breaches += 1;
            }
            hit
        };
        self.total_samples.fetch_add(1, Ordering::Relaxed);
        hit
    }

    /// Demote the family for the process lifetime: route to the exact
    /// arm, count, one loud line. Idempotent.
    pub fn latch(&self, family: u64, div: f32) {
        let (samples, injected) = {
            let mut map = self.families.lock().expect("drift-guard families lock");
            let st = map.entry(family).or_default();
            if st.latched {
                return;
            }
            st.latched = true;
            (st.samples, st.injected)
        };
        let mismatch = if div.is_infinite() {
            " (readback mismatch)"
        } else {
            ""
        };
        eprintln!(
            "[drift-guard] family {family:#x} LATCHED after {samples} sample(s) \
             (rel-RMS div {div:.3e} >= {DIV_CENTER:.1e}{mismatch}; injected={injected}) — \
             routed to the exact arm; counted fallbacks follow"
        );
    }

    /// Whether the family is latched (the fallback route).
    pub fn is_latched(&self, family: u64) -> bool {
        self.families
            .lock()
            .expect("drift-guard families lock")
            .get(&family)
            .is_some_and(|st| st.latched)
    }

    /// Count one exact-path fallback dispatch.
    pub fn note_fallback(&self, family: u64) {
        {
            let mut map = self.families.lock().expect("drift-guard families lock");
            map.entry(family).or_default().fallbacks += 1;
        }
        self.total_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// One-line report (tests + benches; never printed per dispatch).
    pub fn summary(&self) -> String {
        let map = self.families.lock().expect("drift-guard families lock");
        let mut rows: Vec<String> = map
            .iter()
            .map(|(k, st)| {
                format!(
                    "{k:#x}: samples={} breaches={} latched={} fallbacks={} \
                     max_div={:.3e} injected={}",
                    st.samples, st.breaches, st.latched, st.fallbacks, st.max_div, st.injected
                )
            })
            .collect();
        rows.sort();
        format!(
            "samples={} fallbacks={} exact_installed={} families[{}]",
            self.total_samples.load(Ordering::Relaxed),
            self.total_fallbacks.load(Ordering::Relaxed),
            self.exact_installed.load(Ordering::Relaxed),
            rows.join(" · ")
        )
    }
}

static GUARD: OnceLock<DriftGuard> = OnceLock::new();

/// The process-wide guard.
pub fn guard() -> &'static DriftGuard {
    GUARD.get_or_init(DriftGuard::new)
}

#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
/// pub so the forward's load site can install the twin through the
/// crate path; the unarmed decline lives in [`install_exact_mm`].
pub mod metal {
    use super::{
        DriftGuard, INJECT_DELTA, MAX_SAMPLE_ELEMS, breaches, drift_guard_armed, family_key, guard,
        inject_armed, rel_rms_divergence, sample_strided,
    };
    use crate::cubecl_runtime::ActiveRuntime;
    use crate::gemm_dense_f16_metal_zero_copy::MetalDenseF16ZeroCopyGemm;
    use cubecl::client::ComputeClient;
    use cubecl::server::Handle;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex, OnceLock};
    use wgpu::{Device as WgpuDevice, Queue as WgpuQueue};

    /// The exact twin: the SAME MSL source compiled `relaxed = false`
    /// (the byte-exact pre-1176 incumbent posture). Installed once at
    /// forward load when the guard env is armed; `None` = declined.
    static EXACT_MM: OnceLock<Option<Arc<MetalDenseF16ZeroCopyGemm>>> = OnceLock::new();

    /// Scratch handles for the sampled exact run, per byte-size (the
    /// dense module's `output_cache` shape — a handful of distinct
    /// prefill shapes live at once).
    static SCRATCH: OnceLock<Mutex<HashMap<usize, Handle>>> = OnceLock::new();

    /// Install the exact twin (call once at the half-B forward-load site,
    /// where the device/queue Arcs + the SAME MSL source are in scope).
    /// A second install is an idempotent no-op (the OnceLock law).
    pub fn install_exact_mm(
        dev: Arc<WgpuDevice>,
        queue: Arc<WgpuQueue>,
        msl_source: &'static str,
    ) -> Result<(), String> {
        let exact = MetalDenseF16ZeroCopyGemm::new_with_gemm_source(dev, queue, false, msl_source)?;
        if EXACT_MM.set(Some(Arc::new(exact))).is_err() {
            return Err("drift-guard exact twin installed twice".into());
        }
        guard().exact_installed.store(true, Ordering::Relaxed);
        eprintln!(
            "[drift-guard] exact twin installed (relaxed=false, same MSL) — \
             sampling stride {} · center {:.1e} · scale {:.1e}",
            super::SAMPLE_STRIDE,
            super::DIV_CENTER,
            super::DIV_SCALE
        );
        Ok(())
    }

    /// Whether the exact twin is installed (the arming disclosure).
    pub fn exact_mm_installed() -> bool {
        EXACT_MM.get().is_some_and(|o| o.is_some())
    }

    /// One-line-per-process skip-reason disclosure (the failure taxonomy
    /// is first-class — a silent skip would read as "guard idle" in a
    /// cell log when the sampler never measured). Env-gated:
    /// `RIIR_DRIFT_GUARD_DEBUG=1`.
    fn debug_skip(reason: &str) {
        static DONE: OnceLock<()> = OnceLock::new();
        static DEBUG: OnceLock<bool> = OnceLock::new();
        if *DEBUG.get_or_init(|| std::env::var("RIIR_DRIFT_GUARD_DEBUG").as_deref() == Ok("1")) {
            if DONE.set(()).is_ok() {
                eprintln!("[drift-guard] first skip: {reason}");
            }
        }
    }

    /// The guard-aware prefill gemm dispatch. `false` = not handled —
    /// the caller MUST fall through to the incumbent path
    /// byte-identically (the unarmed fast path declines before touching
    /// anything).
    pub fn dispatch_prefill_with(
        client: &ComputeClient<ActiveRuntime>,
        mm_prod: &Arc<MetalDenseF16ZeroCopyGemm>,
        w: &Handle,
        m: usize,
        n: usize,
        input: &Handle,
        output: &Handle,
        p: usize,
    ) -> bool {
        let g: &DriftGuard = guard();
        let family = family_key(m, n);

        // Latched family → the exact path IS the fallback (lossy
        // retained everywhere else; this family is demoted for the
        // process lifetime).
        if g.is_latched(family) {
            if let Some(exact) = EXACT_MM.get().and_then(|o| o.as_ref())
                && exact.dispatch(client, w, m, n, input, output, p).is_ok()
            {
                g.note_fallback(family);
                return true;
            }
            // Exact twin missing or refused: fall through to the
            // incumbent rather than silently dropping the dispatch.
            debug_skip("latched family but exact twin missing/refused — incumbent served");
            return false;
        }

        // Armed + sampled: production FIRST (the forward's output keeps
        // the lossy result — the guard measures, never replaces, until a
        // latch), then the exact twin into guard scratch, then a bounded
        // strided compare.
        if g.should_sample() {
            if mm_prod.dispatch(client, w, m, n, input, output, p).is_err() {
                debug_skip("production dispatch errored — incumbent path reports it");
                return false; // incumbent path reports the failure its way
            }
            let Some(exact) = EXACT_MM.get().and_then(|o| o.as_ref()) else {
                debug_skip("sampled but exact twin not installed");
                return true; // sampled, no twin: measurement skipped, result stands
            };
            let bytes = p * m * std::mem::size_of::<f32>();
            let scratch = {
                let map = SCRATCH.get_or_init(|| Mutex::new(HashMap::new()));
                let mut map = map.lock().expect("drift-guard scratch lock");
                map.entry(bytes)
                    .or_insert_with(|| client.create_from_slice(&vec![0u8; bytes]))
                    .clone()
            };
            if exact.dispatch(client, w, m, n, input, &scratch, p).is_err() {
                debug_skip("exact-twin dispatch errored — measurement skipped");
                return true; // measurement failed, lossy result stands
            }
            // In-order shared queue (the dense module's law): production
            // committed before the exact twin's command buffer, so
            // waiting the twin's last buffer settles both.
            exact.wait_last();
            let (prod, exac) = match (
                client.read_one(output.clone()),
                client.read_one(scratch.clone()),
            ) {
                (Ok(p), Ok(e)) if !p.is_empty() && p.len() == e.len() => (p, e),
                _ => {
                    debug_skip("readback mismatch/empty — never latched blind");
                    return true; // readback mismatch: disclosed, never latched blind
                }
            };
            let prod_f: &[f32] = bytemuck::cast_slice(&prod);
            let exac_f: &[f32] = bytemuck::cast_slice(&exac);
            let mut a = Vec::with_capacity(MAX_SAMPLE_ELEMS);
            let mut b = Vec::with_capacity(MAX_SAMPLE_ELEMS);
            sample_strided(prod_f, &mut a);
            sample_strided(exac_f, &mut b);
            let mut div = rel_rms_divergence(&a, &b);
            let injected = inject_armed();
            if injected {
                // T2 tripwire posture: synthetic drift, counted, named.
                div = INJECT_DELTA;
            }
            if breaches(div) {
                g.note_sample(family, div, injected);
                g.latch(family, div);
            } else {
                g.note_sample(family, div, injected);
            }
            return true;
        }

        // Armed, unsampled: the production dispatch IS the incumbent.
        mm_prod.dispatch(client, w, m, n, input, output, p).is_ok()
    }

    /// The seam the prefill `gemm()` F16 arm calls (cfg-gated at the
    /// call site; WITHOUT the feature the site keeps the incumbent lines
    /// untouched). Returns `false` = fall through to the incumbent path.
    pub fn dispatch_prefill(
        client: &ComputeClient<ActiveRuntime>,
        w: &Handle,
        m: usize,
        n: usize,
        input: &Handle,
        output: &Handle,
        p: usize,
    ) -> bool {
        if !drift_guard_armed() {
            return false;
        }
        let Some(mm) = crate::gemm_dense_f16_metal_zero_copy::gemm_mm_for_posture() else {
            return false;
        };
        dispatch_prefill_with(client, mm, w, m, n, input, output, p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_is_bounded_and_monotone() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(-100.0), 0.0); // saturation branch
        assert_eq!(sigmoid(100.0), 1.0); // saturation branch (f32 folds 1-e^-100 to 1.0)
        // Strictly increasing where f32 can distinguish; |x| ≥ 17 hits
        // the f32 resolution floor (1.0 + e^-17 folds to 1.0) — there the
        // law is non-decreasing + saturated, which the endpoint equality
        // above pins.
        let mut prev = -1.0;
        for i in -12..=12 {
            let v = sigmoid(i as f32);
            assert!(v > prev, "not monotone at {i}");
            prev = v;
        }
        // Non-decreasing across the whole probed range.
        let mut prev = 0.0;
        for i in -40..=40 {
            let v = sigmoid(i as f32);
            assert!(v >= prev, "decreasing at {i}");
            prev = v;
        }
    }

    #[test]
    fn breach_cut_sits_at_div_center() {
        assert!(!breaches(DIV_CENTER * 0.5));
        assert!(!breaches(DIV_CENTER - 1e-9));
        assert!(breaches(DIV_CENTER));
        assert!(breaches(DIV_CENTER * 10.0));
        // NaN breaches — fail-closed (a garbage output demotes to the
        // exact arm, never passes as clean).
        assert!(breaches(f32::NAN));
    }

    #[test]
    fn rel_rms_is_exact_on_identical_and_scales() {
        let b = vec![1.0f32, -2.0, 3.0, 0.0];
        assert_eq!(rel_rms_divergence(&b, &b), 0.0);
        let a = vec![1.1f32, -2.2, 3.3, 0.0];
        let d = rel_rms_divergence(&a, &b);
        // d = sqrt((0.01+0.04+0.09)/(1+4+9)) = sqrt(0.14/14) ≈ 0.1
        assert!((d - 0.1).abs() < 1e-6, "got {d}");
        // ε guards the all-zero exact output.
        let zero = vec![0.0f32; 4];
        assert!(rel_rms_divergence(&a, &zero).is_finite());
    }

    #[test]
    fn strided_sample_bounds_the_readback() {
        let v: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
        let mut out = Vec::new();
        sample_strided(&v, &mut out);
        assert!(out.len() <= MAX_SAMPLE_ELEMS);
        assert_eq!(out[0], 0.0);
        // Even stride: consecutive picks differ by the stride.
        let stride = 10_000f32 / MAX_SAMPLE_ELEMS as f32;
        assert_eq!(out[1] - out[0], stride.ceil());
        // Small slices ride through untouched.
        let small = vec![1.0f32, 2.0];
        sample_strided(&small, &mut out);
        assert_eq!(out, small);
    }

    #[test]
    fn clean_family_never_latches_injected_family_does() {
        let g = DriftGuard::new();
        let clean = family_key(64, 256);
        let dirty = family_key(128, 512);
        // 500 clean samples under the threshold: no latch, no breaches.
        for i in 0..500u64 {
            let div = DIV_CENTER * 0.1 * (1.0 + i as f32 * 1e-4);
            assert!(
                !g.note_sample(clean, div, false),
                "clean sample {i} breached"
            );
        }
        assert!(!g.is_latched(clean));
        // One synthetic-drift breach (the T2 tripwire magnitude) latches
        // immediately; the injected flag rides through to the summary.
        let trip = DIV_CENTER * 10.0;
        assert!(g.note_sample(dirty, trip, true));
        g.latch(dirty, trip);
        assert!(g.is_latched(dirty));
        g.note_fallback(dirty);
        g.note_fallback(dirty);
        let s = g.summary();
        assert!(s.contains("latched=true"));
        assert!(s.contains("injected=1"));
        // `latch` is idempotent (one loud line, one latch row).
        g.latch(dirty, trip);
        assert!(g.summary().contains(&s[..s.find("families[").unwrap()]));
    }

    #[test]
    fn fixed_stride_hits_the_expected_dispatches() {
        let g = DriftGuard::new();
        let hits: Vec<bool> = (0..256).map(|_| g.should_sample()).collect();
        assert!(hits[0]);
        assert!(!hits[1]);
        assert!(hits[SAMPLE_STRIDE as usize]);
        let expected = 256u64.div_ceil(SAMPLE_STRIDE);
        assert_eq!(
            hits.iter().filter(|h| **h).count() as u64,
            expected,
            "stride {SAMPLE_STRIDE} over 256 dispatches"
        );
    }

    #[test]
    fn family_key_stable_and_disjoint() {
        assert_eq!(family_key(64, 256), family_key(64, 256));
        assert_ne!(family_key(64, 256), family_key(256, 64));
        assert_ne!(family_key(64, 256), family_key(64, 512));
    }

    #[test]
    fn env_law_pins_the_constant_surface() {
        // The promote-only env law is asserted by the spelling in
        // `drift_guard_armed` (exact literal "1", no demote literal — an
        // opt-in feature has no default-on posture to demote). Runtime
        // env mutation in tests would poison a parallel suite
        // (env is process-global); this test pins the doc surface.
        assert_eq!(SAMPLE_STRIDE, 64);
        const {
            assert!(DIV_CENTER > 0.0);
            assert!(DIV_SCALE > 0.0);
            assert!(DIV_EPS > 0.0);
        }
        assert_eq!(MAX_SAMPLE_ELEMS, 4096);
    }
}
