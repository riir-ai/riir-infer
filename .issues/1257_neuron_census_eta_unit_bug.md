# Issue 1257 — neuron-census sweep eta prints SECONDS labeled "m" (minutes): a ~60× eta overstatement

**Status:** open — found 2026-11-26 while re-queuing the bmo CHAT sweep job (`1f8da9c4`, 24202/86016 done): the job log printed `[census] L6 col 2698/3584 · 24202/86016 · 0.37/s · eta 167586m` — read as 116 DAYS, spawning the wrong "this will never finish" alarm. The real remaining wall at the measured 0.37 cand/s is 62,014/0.37 = 167,605 **seconds** ≈ 46.6 h.
**Claimed:** zed (amo, this session), 1796317000, expires 2026-11-29T00:00:00+00:00

## The defect

`src/bin/neuron-census.rs` `run_sweep_cell` progress hook:

```rust
let rate = (done as f64) / (t1.elapsed().as_secs_f64()); // candidates per SECOND
let eta = if rate > 0.0 { remaining as f64 / rate } else { f64::NAN }; // => SECONDS
println!("... eta {:.0}m", eta); // labeled "m" — minutes
```

`eta` holds seconds; the format string labels it minutes. Every eta line in every sweep log since the bin landed is ~60× overstated. Downstream harm: queue/est decisions made off the log (today: the 22 h job est vs 65 h measured total, adjudicated by hand) read a number that is wrong by two orders of magnitude in the wrong direction.

## Fix

One line: print `eta / 60.0` (or keep seconds and label `s`). The dead `dt` (`let _ = dt;`) goes in the same commit. No behavior change beyond the printed number; the artifact/cursor/rows paths are untouched; resume compatibility unaffected (print-only).

## Guard

A tiny unit test pins the formatting arithmetic: remaining 62,014 at rate 0.37 → "46.6h" class (not "116 days"). The print is a format-string fix; the test pins the CONVERSION, not the string.
