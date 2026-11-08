# i1185 T2-run — the per-layer width census on Bonsai-2-27B (4090 cell).
#
# Preconditions (the queueing session's contract):
#   * E:\git\riir-infer is gitsynced to origin/develop >= 4e5f471
#     (sink_audit --width-eps + width_census.rs landed).
#   * The checkpoint lives at E:\git\riir-train\data\Ternary-Bonsai-2-27B-PQ2_0.gguf.
#   * GPU-exclusivity: this cell is OFFLINE measurement, not a latency bench —
#     it runs CPU-side (the ternary census + width census are CPU lanes), so a
#     busy GPU does not invalidate it. What DOES invalidate it: another heavy
#     CPU job (the width census is single-threaded CPU at the big window) —
#     the rig queue serializes us; do not run this script by hand beside one.
#
# Cells (pre-registered, issue 1185):
#   window 512, eps grid {0.1, 0.2, 0.3} — the paper's median-width posture is
#   eps=0.2; 0.1/0.3 bracket the sensitivity. Each cell writes its
#   width_census_report.json under .benchmarks/i1185_width_census\<eps>\ plus
#   the v1 sink report beside it (the BLAKE3 pin must reproduce across cells —
#   same window+corpus => same hash; that identity is checked cell-to-cell).
#
# FAIL-FAST engagement: after the first cell, parse the width report and
# refuse (exit 2) if any of: zero per_head rows, every width censored at the
# cap (a 6144-atom cap hit at n=512 is a wiring bug, not a measurement), or
# the v1 blake3 differs between cells.
#
# PowerShell 5.1-safe (no &&, no null-coalescing).

$ErrorActionPreference = "Stop"
Set-Location E:\git\riir-infer

$repo = "E:\git\riir-infer"
$model = "E:\git\riir-train\data\Ternary-Bonsai-2-27B-PQ2_0.gguf"
$outdir = "E:\git\riir-infer\.benchmarks\i1185_width_census"
$log = "$outdir\run.log"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

# Corpus: reuse the 947/948 census corpus if present; else the bin default's
# fallback below (chat_probe should exist on this box — it is a tracked data
# lane; if missing, FAIL, never silently substitute).
$corpus = "E:\git\riir-train\data\chat_probe"
if (-not (Test-Path $corpus)) {
    Write-Output "FAIL: census corpus missing at $corpus (947/948 lane)"
    exit 2
}

# Floor: the HEAD commit must carry the width lane (engagement proof).
$sha = git rev-parse --short=7 HEAD
Write-Output "== i1185 T2-run start $(Get-Date -Format o) HEAD=$sha ==" | Tee-Object -FilePath $log -Append
if (-not (Test-Path "$repo\src\deltanet\width_census.rs")) {
    Write-Output "FAIL: width_census.rs absent at HEAD $sha — gitsync did not land 4e5f471" | Tee-Object -FilePath $log -Append
    exit 2
}

# Build ONCE at the full feature set (ternary + folded-file hadamard).
$env:CARGO_NET_OFFLINE = "true"
cargo build --release --features sink_audit,deltanet_ternary_inference,bonsai2_hadamard --bin sink_audit
if ($LASTEXITCODE -ne 0) {
    Write-Output "FAIL: build" | Tee-Object -FilePath $log -Append
    exit 2
}
$bin = "$repo\target\release\sink_audit.exe"

$firstHash = $null
foreach ($eps in @("0.1", "0.2", "0.3")) {
    $cell = "$outdir\eps$eps"
    New-Item -ItemType Directory -Force -Path $cell | Out-Null
    Write-Output "== cell eps=$eps $(Get-Date -Format o) ==" | Tee-Object -FilePath $log -Append
    & $bin $model --corpus $corpus --n 512 --width-eps $eps --out "$cell\sink_audit.json" 2>&1 | Tee-Object -FilePath "$cell\stdout.log" -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Output "FAIL: cell eps=$eps run exit $LASTEXITCODE" | Tee-Object -FilePath $log -Append
        exit 2
    }
    $wreport = "$cell\width_census_report.json"
    if (-not (Test-Path $wreport)) {
        Write-Output "FAIL: cell eps=$eps produced no width report" | Tee-Object -FilePath $log -Append
        exit 2
    }
    # Engagement + sanity parse (the fail-fast arms).
    $raw = Get-Content $wreport -Raw
    $heads = ([regex]::Matches($raw, '"layer"')).Count
    $censored = ([regex]::Matches($raw, '"censored": true')).Count
    if ($heads -eq 0) {
        Write-Output "FAIL: eps=$eps width report has zero per_head rows" | Tee-Object -FilePath $log -Append
        exit 2
    }
    if ($censored -eq $heads) {
        Write-Output "FAIL: eps=$eps every head censored at n=512 — wiring bug, not a measurement" | Tee-Object -FilePath $log -Append
        exit 2
    }
    $m = [regex]::Match($raw, '"n_positions": (\d+)')
    if (-not $m.Success -or [int]$m.Groups[1].Value -ne 512) {
        Write-Output "FAIL: eps=$eps width report window mismatch" | Tee-Object -FilePath $log -Append
        exit 2
    }
    # The v1 BLAKE3 must be identical across cells (same window+corpus).
    $sinkRaw = Get-Content "$cell\sink_audit.json" -Raw
    $hm = [regex]::Match($sinkRaw, '"blake3": "([0-9a-f]{64})"')
    if (-not $hm.Success) {
        Write-Output "FAIL: eps=$eps sink report carries no blake3" | Tee-Object -FilePath $log -Append
        exit 2
    }
    $h = $hm.Groups[1].Value
    if ($null -eq $firstHash) {
        $firstHash = $h
    } else {
        if ($h -ne $firstHash) {
            Write-Output "FAIL: v1 blake3 drifted across cells ($h != $firstHash) — the width flag leaked into the canonical payload" | Tee-Object -FilePath $log -Append
            exit 2
        }
    }
    Write-Output "cell eps=$eps OK: heads=$heads censored=$censored v1blake3=$($h.Substring(0,12))" | Tee-Object -FilePath $log -Append
}

Write-Output "== i1185 T2-run DONE $(Get-Date -Format o) — cells in $outdir ==" | Tee-Object -FilePath $log -Append
Write-Output "I1185 T2 RUN DONE"
