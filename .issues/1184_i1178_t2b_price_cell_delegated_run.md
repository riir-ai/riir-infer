# Issue 1184 — run the i1178 T2b price cell (lane-sliced Q4 decode GEMV) on bmo — delegated execution vehicle

**Status:** OPEN — filed 2026-11-08 (session zed-m3-i1178t2b@m3, epoch 1794146120). Owner call: the amo rig session executes the bmo benchmark; the m3 session filed this issue and folded its 1178 lease. **UNCLAIMED — the executing session claims this issue (append `**Claimed:**` + push) before starting.**
Parent: riir-infer **Issue 1178 T2b** (the implementation `5674cef4` + engagement instrument `fd8ecac7` are already LANDED on origin/develop — nothing to write, this is a RUN-and-RECORD issue). Cross-links: 1179 (the beat arithmetic's serial half), Bench 1167 (the uzu bar), Bench 951 (validity question below), 1177 item 2 (f16 payload class — see finding).

## The posture finding (settled by code-reading + the GGUF header probe, 2026-11-08)

The league decode baseline **12.05 tok/s (Bench 1167) is the dense-f16 posture**: `production` implies `metal_tensor_gemm`, so on macOS `dense_armed` defaults ON, all Q4_K tensors load as **f16** (`from_gguf_f16`), and decode rides the plain `gemv_dense_f16` kernel. The packed Q4 route (where BOTH the T2a factored feature and the T2b lanesliced feature live) is only reachable with `RIIR_QWEN38_DENSE_MM2D=0` — plus `RIIR_QWEN38_ATTN_TENSOR=0` and `RIIR_QWEN38_DENSE_MM2D_F16ACT=0` (the boot REFUSES demoted-dense with those two still armed).

Real dims (GGUF header probe, `plan402_gguf_probe.py`): e=5120, m=17408, 65 blocks, vocab 248320. Byte math per decode step (FFN weights only, 64 layers × 3 × 5120 × 17408 = 17.1G params):

| posture | FFN bytes/step | wall (census ffn share) | effective BW | roofline class |
|---|---|---|---|---|
| league f16 (12.05) | 34.2 GB | 47.5 ms | ~720 GB/s | **~88% of ~819 GB/s** |
| packed Q4 (pre-1176 control: 8.61, Oct-29 build) | 9.6 GB | — | ~120-200 GB/s | ~15-25% (ALU-bound) |
| uzu 4-bit serial-equiv (R2, 0.6.17) | ~14.5 GB | ~26.3 ms/pass | ~551 GB/s | ~67% (their own record's figure) |

Consequences recorded as claims this cell must TEST:
1. **The 4.4× decode gap is a weight-byte-class gap** (3.6-3.8× bytes × ~1.18× roofline-efficiency), not a kernel-craft gap — our f16 kernels already extract MORE roofline than uzu's. The serial path to the uzu class runs through making the **packed Q4 GEMV bandwidth-bound** — exactly the lanesliced law (T2b) — not through f16 (byte-bound, ~15% headroom left).
2. **Bench 951 validity question (pre-registered discriminator):** nothing in its record sets the demote env, and at the league default BOTH its arms load f16 — the `q4k_factored_decode_gemv` feature is a compile-time no-op there, so its "FLAT, shape class doesn't transfer" verdict may be an unreachable-arm artifact. **Discriminator: the packedOFF leg of this run.** packedOFF ≈ 8.6-9 tok/s ⇒ 951 measured f16 (artifact confirmed — file the correction on 951 + 1178); packedOFF ≈ 12 tok/s ⇒ packed caught up post-Oct-29 (record which arms did it, 951 stands).

## Setup state (already done by m3, 2026-11-08)

- **bmo worktree exists**: `~/git/riir-infer-t2bwt`, created detached; RE-CHECKOUT to latest `origin/develop` before building (must be ≥ `fd8ecac7`): `cd ~/git/riir-infer && git fetch origin && cd ~/git/riir-infer-t2bwt && git checkout --detach origin/develop`.
- **Model**: `/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf` (16.81 GB, the Bench-1167 pin, blake3 `ce14c955…cc5bd06a`) — the bench bin's DEFAULT_GGUF.
- bmo reaches via the amo thunderbolt bridge (tailscale down): `ssh -J katopz@100.111.186.76 katopz@192.168.10.2`. Pre-fetch via ssh BEFORE the job ("no fetch in job context" — the i1167 lesson). `PATH=/opt/homebrew/bin` in any job context.
- **Do not disturb the sibling splash cell** (rig job `01a1bb77`, exclusive, GGUF download + serve). Queue this job on the exclusive lane AFTER it — `--exclusive` serializes; the script also carries an advisory sibling-process guard and the mds tripwire (the 15 GiB download's Spotlight index pass is the Bench-950 page-out class; NEVER record a page-out posture).

## The job (stage this script on bmo as /tmp/t2b_job.sh, run via `agent-mail rig run bmo --exclusive --needs metal --timeout 2h -- /bin/zsh /tmp/t2b_job.sh`)

```zsh
#!/bin/zsh
# i1178 T2b price cell — lane-sliced Q4 decode GEMV (impl 5674cef4 + instrument fd8ecac7)
# Three arms: packed OFF / packed ON (feature-diff, same tree) + league-f16 cross-check.
set -u
export PATH=/opt/homebrew/bin:$PATH
WT="$HOME/git/riir-infer-t2bwt"
LOGDIR=/tmp/i1178_t2b
mkdir -p "$LOGDIR"
echo "== i1178 T2b job start $(date -u +%FT%TZ) =="

# Advisory sibling-engine wait (splash cell may be draining). Up to 20 min, advisory only.
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  if pgrep -if 'splash' >/dev/null 2>&1 || pgrep -f 'uzu-engine|uzu serve|uzu27b.*serve' >/dev/null 2>&1; then
    echo "[guard] sibling engine process alive (leg $i); waiting 60s"; sleep 60
  else
    echo "[guard] no sibling engine process (leg $i)"; break
  fi
done

# mds tripwire (Bench-950 lesson): drain up to 5 min, hard-refuse exit 3.
mds_cpu() { ps -A -o %cpu,comm | grep -E 'mds_stores|/[m]ds$' | awk '{s+=$1} END {print s+0}'; }
for i in 1 2 3 4 5; do
  c=$(mds_cpu)
  if awk "BEGIN{exit !($c > 5)}"; then echo "[mds] ${c}% > 5%; draining 60s (leg $i)"; sleep 60; else break; fi
done
c=$(mds_cpu)
if awk "BEGIN{exit !($c > 5)}"; then echo "[mds] REFUSE exit 3: mds_stores at ${c}% after drain"; exit 3; fi
echo "[mds] clean at ${c}%"

cd "$WT" || exit 20
git --no-pager log --oneline -1
export CARGO_TARGET_DIR="$WT/target"

echo "== build OFF (league feature set) =="
cargo build --release -p riir-infer-gpu --features qwen38_wgpu,production --bin qwen38_wgpu_bench 2>&1 | tail -3
[ -x target/release/qwen38_wgpu_bench ] || { echo "BUILD OFF FAILED"; exit 10; }
cp target/release/qwen38_wgpu_bench /tmp/t2b_bench_OFF
echo "== build ON (+q4k_lanesliced_decode_gemv) =="
cargo build --release -p riir-infer-gpu --features qwen38_wgpu,production,q4k_lanesliced_decode_gemv --bin qwen38_wgpu_bench 2>&1 | tail -3
[ -x target/release/qwen38_wgpu_bench ] || { echo "BUILD ON FAILED"; exit 11; }
cp target/release/qwen38_wgpu_bench /tmp/t2b_bench_ON

BOX="uptime: $(uptime) | ac: $(pmset -g ps | head -1 | tr -s ' ')"

run_arm() {
  tag=$1; bin=$2; packed=$3
  if [ "$packed" = "1" ]; then
    ENVSET="RIIR_QWEN38_DENSE_MM2D=0 RIIR_QWEN38_ATTN_TENSOR=0 RIIR_QWEN38_DENSE_MM2D_F16ACT=0"
  else
    ENVSET=""
  fi
  echo "== arm $tag leg1 (pp1536 + decode512 + ctx1344 + walk64) env: $ENVSET =="
  env $ENVSET "$bin" --pp 1536 --decode 512 --decode-ctx 1344 --walk > "$LOGDIR/${tag}_leg1.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "ARM $tag LEG1 FAILED rc=$rc — log tail:"; tail -20 "$LOGDIR/${tag}_leg1.log"; exit 12; fi
  echo "== arm $tag leg2 (decode512 drift leg) =="
  env $ENVSET "$bin" --decode 512 > "$LOGDIR/${tag}_leg2.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "ARM $tag LEG2 FAILED rc=$rc — log tail:"; tail -20 "$LOGDIR/${tag}_leg2.log"; exit 13; fi
  echo "-- $tag posture lines:"
  grep -hE 'Issue 1150|Issue 1176|1178 T2b' "$LOGDIR/${tag}_leg1.log" | sort | uniq -c | sort -rn | head -6
  echo "-- $tag engagement: demoted=$(grep -c 'DEMOTED' "$LOGDIR/${tag}_leg1.log") armed=$(grep -c 'dense-f16 matmul2d arm armed' "$LOGDIR/${tag}_leg1.log") ls_twin=$(grep -c '1178 T2b' "$LOGDIR/${tag}_leg1.log")"
  echo "-- $tag cells:"
  grep -hE '\[decode|\[pp |\[decode-ctx|loaded in' "$LOGDIR/${tag}_leg1.log" "$LOGDIR/${tag}_leg2.log"
}

run_arm packedOFF /tmp/t2b_bench_OFF 1
run_arm packedON  /tmp/t2b_bench_ON  1
run_arm leagueOFF /tmp/t2b_bench_OFF 0

echo "== G1 walk pins (md5 of the [walk] ids line) =="
for f in "$LOGDIR"/packedOFF_leg1.log "$LOGDIR"/packedON_leg1.log; do
  echo "$f : $(grep '\[walk\] ids' "$f" | md5)"
done
diff <(grep '\[walk\] ids' "$LOGDIR/packedOFF_leg1.log") <(grep '\[walk\] ids' "$LOGDIR/packedON_leg1.log") >/dev/null 2>&1 \
  && echo "G1: walk ids IDENTICAL packedOFF==packedON" \
  || echo "G1: walk ids DIVERGE packedOFF!=packedON — count + disclose per the 940 law"

echo "== box: $BOX"
echo "== i1178 T2b job end $(date -u +%FT%TZ) =="
```

## Preregistered gates + decision rules (fixed BEFORE the run — never loosened after)

- **Engagement (Issue-1124 T1.5 law):** packed arms MUST print `[Issue 1176] … DEMOTED` exactly once per leg; packedON MUST print the `[1178 T2b] lane-sliced Q4 decode twin armed (m=…, n=…)` line once per Q4_K tensor (≈ hundreds of lines; grep -c > 0 is the pass bar — 0 = the arm never engaged, the run is VOID); leagueOFF MUST print the `[Issue 1150] … arm armed` line. A leg missing its posture line = VOID, re-run, never recorded.
- **G1 token identity:** the packedOFF == packedON greedy walk (64 tokens, default prompt — the Bench-950 fixture) must be id-identical (`[walk] ids` md5 equal). On divergence: count flips + disclose top-2 logit margins per the Issue-940 first-index/order law — and the feature CANNOT join production without a determinism story (the fused form `a·nib·x − b·x` vs the canonical `(A·nib − B)·x` is a different FP association; if it flips, the fix is re-associating the kernel, not absorbing the delta).
- **The 951 discriminator** (above): packedOFF's absolute decides which correction lands.
- **T2b price verdict:** packedON vs packedOFF on decode short + ctx1344. The lanesliced law's mechanism claim: lifting the Q4 GEMV off its ALU bound (~15-25% roofline) toward the bandwidth bound (f16 demonstrates ~88% is reachable on this GPU). Read effective BW per arm in the record (FFN bytes: 9.6 GB packed / 34.2 GB f16, over the ffn wall implied by the totals).
- **T3 production question (owner-gated):** packedON vs the LEAGUE f16 12.05 at the ≥1.5% e2e bar. Note carefully: a win here is NOT a one-row feature join — it is a **league-posture flip** (packed+lanesliced becomes the decode posture, displacing 1176's f16 default), which is an owner act with its own demote-check burden. Record the number; do NOT flip anything.
- **Load-cost disclosure:** packedON's `[bench] loaded in Ns` vs packedOFF's (the CPU repack, ~0.3-0.5 s/100 MB single-thread) + the ~1.44× quant-weight memory — both go in the record.
- **G3:** leagueOFF must reproduce ~12.05 (±drift band) — it is also the same-window f16 datum the record needs.

## Deliverables (same session as the run)

- [ ] Bench record at the next `.benchmarks` number (read `.benchmarks/.highwater` at write time; 955 expected) — cells, engagement proofs, md5 pins, box state (uptime/AC + the rig's QUIET status + job id), PROVENANCE lines, the 951 discriminator verdict, the posture finding confirmed/corrected.
- [ ] Issue 1178: T2b checkbox + status line update (the record number; the production verdict class; the 951 correction if it fired); fold the bench's lease row.
- [ ] If the discriminator confirmed the 951 artifact: append the correction note to `.benchmarks/951_…md` (the record's verdict was measured at an unreachable arm; its "shape class doesn't transfer" conclusion is unsupplemented) — the T2a routed-factored price cell is UNMEASURED on the packed route, noted for a future re-run, never silently re-claimed.
- [ ] Commit + push both (docs commits; `Session:` trailer with name + epoch).
- [ ] Cross-link the record on 1179 (the serial half the beat arithmetic consumes) and in the league doc's uzu block ONLY as WATCH intel (the scored opponents stay llama.cpp + vLLM; uzu/splash stay format-crossed/non-scoring).

## Boundary

riir-infer owns everything here. No training (the lanesliced repack is a deterministic load-time transform — modelless-clean). Weights never committed. bmo is a quiet runner — do not run the job from m3 (owner delegation), do not touch the sibling splash job's files or logs (`/tmp` dirs except this issue's own `/tmp/i1178_t2b`).
