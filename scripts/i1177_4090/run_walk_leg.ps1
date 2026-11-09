# Issue 1177 Item 1 — the 4090 CUDA walk leg (PowerShell).
# Waits for the model transfer to complete (size == the amo original's exact
# byte count), hashes the file, then runs the walk-parity bin.
# The expected size line + the sha256 print beside the walk output — the
# record's A1/A8 pin.
#
# Expected size: 16810714336 bytes (the amo original; the transfer is
# resume-looped on the amo side — scp full-file retries, so a size match
# implies a complete file).

$target = "F:/models/qwen3.8-27b-Q4_K_M.amo-copy.gguf"
$expected = 16810714336

Write-Output "[i1177-4090] waiting for model transfer (need $expected bytes)..."
$deadline = (Get-Date).AddHours(9)
while ($true) {
    $len = (Get-Item $target -ErrorAction SilentlyContinue).Length
    if ($len -eq $expected) { break }
    if ((Get-Date) -gt $deadline) {
        Write-Output "[i1177-4090] REFUSE: transfer did not complete within 9 h (last size $len)"
        exit 4
    }
    Start-Sleep -Seconds 60
}
Write-Output "[i1177-4090] size match: $len"

Write-Output "[i1177-4090] sha256 (may take ~1-2 min):"
$hash = (Get-FileHash -Path $target -Algorithm SHA256).Hash.ToLower()
Write-Output "model-sha256 $hash"

$env:QWEN38_GGUF = $target
Write-Output "[i1177-4090] running the walk (ctx 2048, 48 tokens, default prose prompt)..."
& E:\git\riir-infer\target\release\qwen38_cudarc_walk.exe --ctx 2048 --walk-tokens 48
Write-Output "[i1177-4090] walk exit rc=$LASTEXITCODE"
exit $LASTEXITCODE
