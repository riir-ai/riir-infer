# i1189 gate v9 - probe round: full unfiltered output + --nocapture so the
# cfg(test) eprintln probes surface. Same detached-worktree posture.
$ErrorActionPreference = "Continue"
Set-Location E:\git\riir-infer
Write-Output "=== i1189 gate v9 (probes) ==="
Write-Output ("MAIN HEAD: " + (git --no-pager log -1 --format='%H'))
git worktree add E:\git\i1189_gate HEAD 2>&1 | Select-Object -Last 1
Set-Location E:\git\i1189_gate
git apply C:\Users\katop\i1189_gate.patch 2>&1 | Select-Object -Last 3
if ($LASTEXITCODE -ne 0) { Write-Output "PATCH-APPLY-FAILED exit=$LASTEXITCODE"; Set-Location E:\git\riir-infer; git worktree remove --force E:\git\i1189_gate; exit 1 }
Write-Output "PATCH-APPLIED"
Write-Output "--- check (closed_form_write feature) ---"
cargo check -p riir-infer-core --features closed_form_write 2>&1
Write-Output ("check-feature exit=" + $LASTEXITCODE)
Write-Output "--- clippy (feature, lib, -D warnings) ---"
cargo clippy -p riir-infer-core --features closed_form_write --lib -- -D warnings 2>&1
Write-Output ("clippy-feature exit=" + $LASTEXITCODE)
Write-Output "--- tests (closed_form_write module, --nocapture) ---"
cargo test -p riir-infer-core --lib --features closed_form_write closed_form_write -- --nocapture 2>&1
Write-Output ("test-feature exit=" + $LASTEXITCODE)
Set-Location E:\git\riir-infer
git worktree remove --force E:\git\i1189_gate 2>&1 | Select-Object -Last 1
Write-Output ("worktree-removed exit=" + $LASTEXITCODE)
Write-Output "=== i1189 gate v9 DONE ==="
