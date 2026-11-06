"""Files written into the bundle for the run computer: ``setup/setup_runner.ps1``, ``RUNNER_README.md``, and the
``unpack`` step (restore packed arrays and verify every sha256 of MANIFEST.json).

No path of the thesis PC appears in these files; everything goes under ``C:\\uol_final`` and the located Drive root.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SETUP_PS1 = r'''<#
  uol.final run computer setup. Run once, in a normal PowerShell (no administrator rights needed):
      powershell -ExecutionPolicy Bypass -File "<Drive>\UOL_FINAL_RUN\setup\setup_runner.ps1" [-DriveRoot "<path to UOL_FINAL_RUN>"]

  1. Finds the shared folder UOL_FINAL_RUN (-DriveRoot, or <letter>:\My Drive, Mi unidad, .shortcut-targets-by-id\<id>).
  2. Checks that its results\ folder is writable (writes and deletes a test file). A second run computer (D85) passes
     -ResultsRoot UOL_RUN_B (a folder name beside UOL_FINAL_RUN, or a full path): its results\ is used instead.
  3. Waits until Drive has synced the whole bundle: every file of MANIFEST.json must exist with its sha256.
  4. Copies the bundle to C:\uol_final (every top-level file of MANIFEST too) and verifies every sha256 again.
  5. Creates C:\uol_final\venv (Python 3.12), installs code\requirements.lock, and torch (CUDA 12.8 build, sm_120).
  6. Unpacks the compressed arrays (python -m uol.final unpack) and checks their sha256.
  7. Checks the GPU (sm_120 supported, bf16 ok) and runs the test suite (pytest tests/final).
  Opens no ports, stores no credentials. Undo: delete C:\uol_final.
#>
param(
    [string]$DriveRoot,
    [string]$Root = "",
    [string]$ResultsRoot = "",
    [int]$SyncWaitMinutes = 240,
    [switch]$SkipTorch
)
$ErrorActionPreference = "Stop"
if (-not $Root) {
    $cfree = (Get-PSDrive C).Free
    $dd = Get-PSDrive D -ErrorAction SilentlyContinue
    if ($dd -and $dd.Free -gt $cfree) { $Root = "D:\uol_final" } else { $Root = "C:\uol_final" }
}
New-Item -ItemType Directory -Force "$Root\logs" | Out-Null
$log = "$Root\logs\setup.log"
$env:PYTHONDONTWRITEBYTECODE = "1"      # nothing is written inside code\ (every file there is in MANIFEST.json)
$env:HF_HUB_DISABLE_SYMLINKS = "1"      # no Developer Mode needed for the Hugging Face cache (as on the run computer)
function Say($m) { $line = "[{0}] {1}" -f (Get-Date -Format s), $m; Write-Host $line; Add-Content -Path $log -Value $line }

Say "== 1. Find the shared Drive folder UOL_FINAL_RUN"
$found = $null
if ($DriveRoot) {
    if ((Split-Path $DriveRoot -Leaf) -ne "UOL_FINAL_RUN" -and (Test-Path -LiteralPath (Join-Path $DriveRoot "UOL_FINAL_RUN"))) { $DriveRoot = Join-Path $DriveRoot "UOL_FINAL_RUN" }
    if (Test-Path -LiteralPath $DriveRoot) { $found = $DriveRoot }
} else {
    foreach ($d in (Get-PSDrive -PSProvider FileSystem)) {
        $cands = @()
        foreach ($sub in @("My Drive", "Mi unidad", "Mon Drive", "Meine Ablage")) { $cands += (Join-Path $d.Root (Join-Path $sub "UOL_FINAL_RUN")) }
        $sc = Join-Path $d.Root ".shortcut-targets-by-id"
        if (Test-Path -LiteralPath $sc) { foreach ($x in (Get-ChildItem -LiteralPath $sc -Directory -ErrorAction SilentlyContinue)) { $cands += (Join-Path $x.FullName "UOL_FINAL_RUN") } }
        foreach ($c in $cands) { if (Test-Path -LiteralPath (Join-Path $c "MANIFEST.json")) { $found = $c; break } }
        if ($found) { break }
    }
}
if (-not $found) { throw "UOL_FINAL_RUN not found. In Google Drive (web) open 'Shared with me', right-click UOL_FINAL_RUN, choose 'Organize > Add shortcut to My Drive', wait for Drive for desktop to sync, then rerun (or pass -DriveRoot '<full path>\UOL_FINAL_RUN')." }
Say "Drive folder: $found"
$resRoot = $found
if ($ResultsRoot) {
    if ($ResultsRoot -match '[\\/:]') { $resRoot = $ResultsRoot } else { $resRoot = Join-Path (Split-Path $found -Parent) $ResultsRoot }
}
$res = Join-Path $resRoot "results"

Say "== 2. Check write access to $res"
New-Item -ItemType Directory -Force $res | Out-Null
$t = Join-Path $res (".write_test_" + $PID)
# .NET, not Set-Content: on RUN-B's Drive (23 Oct) Set-Content failed for new files ("Unable to find the specified
# file", a 0-byte file left) while Python, [IO.File]::WriteAllText and Out-File, the writes the runner uses, worked
try {
    [IO.File]::WriteAllText($t, "ok")
    if ([IO.File]::ReadAllText($t) -ne "ok") { throw "read back differs" }
} catch {
    throw "$res is not writable: $($_.Exception.Message)"
} finally {
    if (Test-Path -LiteralPath $t) { Remove-Item -LiteralPath $t -Force -ErrorAction SilentlyContinue }
}
Say "$res is writable"

Say "== 3. Wait for Drive to sync the bundle (every MANIFEST sha256)"
$man = Get-Content -LiteralPath (Join-Path $found "MANIFEST.json") -Raw | ConvertFrom-Json
$names = @($man.files.PSObject.Properties.Name)
$deadline = (Get-Date).AddMinutes($SyncWaitMinutes)
while ($true) {
    $bad = @()
    foreach ($n in $names) {
        $p = Join-Path $found ($n -replace '/', '\')
        $want = $man.files.$n.sha256
        if (-not (Test-Path -LiteralPath $p)) { $bad += "missing $n"; continue }
        if ((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower() -ne $want) { $bad += "hash $n" }
    }
    if ($bad.Count -eq 0) { break }
    if ((Get-Date) -gt $deadline) { throw ("Drive copy still incomplete after $SyncWaitMinutes min: " + ($bad | Select-Object -First 10) -join "; ") }
    Say ("waiting for Drive sync: {0} of {1} files not ready (e.g. {2}); checking again in 5 min" -f $bad.Count, $names.Count, $bad[0])
    Start-Sleep -Seconds 300
}
Say ("all {0} files present with their sha256" -f $names.Count)

Say "== 4. Copy to $Root and verify"
foreach ($sub in @("code", "data", "setup", "jobs")) {
    robocopy (Join-Path $found $sub) (Join-Path $Root $sub) /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy $sub failed ($LASTEXITCODE)" }
}
$top = @("MANIFEST.json") + @($names | Where-Object { $_ -notmatch "/" })    # every top-level file of MANIFEST (23 Oct)
foreach ($f in $top) { if (Test-Path -LiteralPath (Join-Path $found $f)) { Copy-Item -LiteralPath (Join-Path $found $f) -Destination (Join-Path $Root $f) -Force } }
foreach ($n in $names) {
    $p = Join-Path $Root ($n -replace '/', '\')
    if ((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower() -ne $man.files.$n.sha256) { throw "local copy differs: $n" }
}
Say "local copy verified"

Say "== 5. Python 3.12 environment"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Python launcher 'py' not found. Install Python 3.12 (winget install -e --id Python.Python.3.12), then rerun." }
if (-not (Test-Path "$Root\venv\Scripts\python.exe")) { & py -3.12 -m venv "$Root\venv" }
$vpy = "$Root\venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip | Out-Null
& $vpy -m pip install -r "$Root\code\requirements.lock"
if ($LASTEXITCODE -ne 0) { throw ("pip install -r requirements.lock failed. If it failed building psutil 5.9.0, wrapt 1.14.1 or " +
    "kiwisolver 1.4.4 (no Python 3.12 wheels), the C++ build tools are missing: ask the user to approve " +
    "winget install Microsoft.VisualStudio.2022.BuildTools --override '--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended' " +
    "(as on the run computer, 21 Oct), then rerun setup. Never change the pins.") }
if (-not $SkipTorch) {
    # the lock pulls a CPU torch from PyPI; the run computer replaced it with this CUDA 12.8 build (21 Oct 2026). Only
    # when it is not already there: every sequence re-downloaded and reinstalled ~3 GB, and on a full disk (RUN-B,
    # 23 Oct, Errno 28) the reinstall itself failed. "torch==2.11.0" in the lock is met by 2.11.0+cu128, so pip -r keeps it.
    $tv = (& $vpy -c "import torch;print(torch.__version__)" 2>$null | Out-String).Trim()
    if ($tv -ne "2.11.0+cu128") {
        & $vpy -m pip install --force-reinstall --no-deps "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE -ne 0) { throw "torch 2.11.0+cu128 install failed" }
    } else {
        Say "torch 2.11.0+cu128 already installed; not reinstalled"
    }
    & $vpy -m pip install "setuptools<82"
}
# pyarrow 16.1.0 ships an old msvcp140.dll that breaks torch when pandas is imported first; the system runtime is used
# instead (the run computer's fix of 21 Oct 2026; no pin changes)
$pa = Join-Path (& $vpy -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim() "pyarrow\msvcp140.dll"
if (Test-Path -LiteralPath $pa) { Rename-Item -LiteralPath $pa -NewName "msvcp140.dll.operator_disabled" -Force; Say "pyarrow msvcp140.dll disabled" }
Say "== 5a. Check the pinned ML stack (imports and versions against requirements.lock)"
$chk = @"
import importlib, re, sys
from importlib.metadata import version
pins = dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s#]+)', open(r'$Root\code\requirements.lock', encoding='utf-8').read(), re.M))
bad = []
for dist, mod in (('transformers', 'transformers'), ('sentence-transformers', 'sentence_transformers'), ('peft', 'peft'),
                  ('accelerate', 'accelerate'), ('torch', 'torch')):
    try:
        importlib.import_module(mod)
        v = version(dist)
    except Exception as e:
        bad.append(f'{dist}: import failed ({type(e).__name__}: {e})'); continue
    want = pins.get(dist)
    print(f'{dist} {v}' + (f' (lock {want})' if want else ' (CUDA 12.8 build; not pinned)'))
    if want and v != want:
        bad.append(f'{dist}: installed {v}, lock {want}')
import torch
print('torch cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if bad:
    print('MISMATCH: ' + '; '.join(bad)); sys.exit(3)
"@
& $vpy -c $chk | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { throw "the ML stack does not match requirements.lock (see above); fix the environment, do not change the lock" }
$site = (& $vpy -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
Set-Content -Path (Join-Path $site "uol_final.pth") -Value "$Root\code\src" -Encoding ascii
& $vpy -m pip freeze | Out-File -FilePath "$Root\logs\pip_freeze.txt" -Encoding utf8
Say "environment ready: $vpy"
Say "== 5b. Codex CLI (native codex.exe, never the .cmd shim)"
$codex = (& $vpy -c "from uol.round3 import luna; print(luna.codex_bin())" 2>&1 | Out-String).Trim()
Say "codex: $codex  (model pinned: gpt-5.6-luna; the plan's codex_smoke job checks that it answers on this account)"

Say "== 6. Unpack compressed arrays (sha256 checked)"
$env:NLTK_DATA = "$Root\data\nltk_data"
& $vpy -m uol.final unpack --root $Root
if ($LASTEXITCODE -ne 0) { throw "unpack failed (see above)" }

Say "== 7. GPU check and tests"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | Tee-Object -FilePath $log -Append
if (-not $SkipTorch) {
    & $vpy -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available());print(torch.cuda.get_device_name(0));print('sm_120 supported:', 'sm_120' in torch.cuda.get_arch_list());x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16);print('bf16 ok:', bool((x@x).float().abs().mean()>0))" | Tee-Object -FilePath $log -Append
}
Push-Location "$Root\code"
$env:UOL_FINAL_DATA = $null
& $vpy -B -m pytest tests/final -q -p no:cacheprovider | Tee-Object -FilePath $log -Append
$rc = $LASTEXITCODE
Pop-Location
if ($rc -ne 0) { throw "tests failed (exit $rc)" }
Copy-Item -LiteralPath $log -Destination (Join-Path $res "setup.log") -Force
# one argument string with every path in double quotes: Start-Process joins an array without quoting, so a path
# with a space ('exchange\...') would split into two arguments
$runArgs = "-m uol.final overnight --plan `"$Root\jobs\plan.json`" --drive-root `"$found`""
if ($ResultsRoot) { $runArgs += " --results-root `"$resRoot`" --jobs-file `"$resRoot\ASSIGNMENTS.json`"" }
Say "Setup complete. Next:"
Say "  `$env:UOL_FINAL_LABELS='v11'; & '$vpy' -m uol.final --data '$Root\data' --store '$Root\run\final_20261021' g1 --route cached"
Say "  Start-Process -FilePath '$vpy' -ArgumentList '$runArgs' -RedirectStandardOutput '$Root\logs\overnight.log' -RedirectStandardError '$Root\logs\overnight.err' -WindowStyle Minimized"
'''

README = r'''# Running the final benchmark on the run computer

The bundle holds development data only (no test-set row; checked when it was built, at setup, and again when the
runner starts). Code, data and plan are fixed: the operator never changes what is computed.

## 1. Before you start (once)

1. Install **Google Drive for desktop** and sign in. The folder `UOL_FINAL_RUN` was shared with this account: in
   Google Drive on the web, open *Shared with me*, right-click `UOL_FINAL_RUN`, choose *Organize > Add shortcut to My
   Drive*. Wait until Drive for desktop shows it synced (it appears as `bundle\UOL_FINAL_RUN`, `Mi unidad`, or
   under `G:\.shortcut-targets-by-id\...`; the drive letter may differ).
2. Install **Python 3.12** (`winget install -e --id Python.Python.3.12`) and make sure `nvidia-smi` works.
3. Power: never sleep while plugged in (`powercfg /change standby-timeout-ac 0`, `powercfg /change monitor-timeout-ac 0`).
4. Free disk on C: about 60 GB (the arrays are unpacked locally).

## 2. Setup (about 30-60 minutes, mostly downloads)

`<Root>` below is `D:\uol_final` when D: has more free space than C:, otherwise `<Root>` (the setup script
chooses and prints it; pass `-Root` to force one).

    powershell -ExecutionPolicy Bypass -File "<Drive>\UOL_FINAL_RUN\setup\setup_runner.ps1"

Add `-DriveRoot "<full path>\UOL_FINAL_RUN"` if the folder is not found automatically. The script waits for Drive to
finish syncing (it checks every sha256 of `MANIFEST.json`), copies everything to `<Root>`, verifies again,
builds `<Root>\venv`, unpacks the arrays, checks the GPU (`sm_120 supported: True`, `bf16 ok: True`) and runs
the tests. A hash failure stops it: report the file names; do not repair data.

## 3. Gate G1, then the run

    $env:UOL_FINAL_LABELS = "v11"; <Root>\venv\Scripts\python.exe -m uol.final --data <Root>\data --store <Root>\run\final_20261021 g1 --route cached

It must print PASS (frozen development values within 1e-4: word matching .184, Qwen3-8B alone .274, rankers .4081
and .4721, LLM-stage hybrid .5558). The full G1 (with the CPU recompute of L* and both rankers) is the first job of the
overnight plan and blocks every other job if it fails (`results\G1_FAILED.md`).

Start the runner in the background so it survives the chat:

    Start-Process -FilePath '<Root>\venv\Scripts\python.exe' -ArgumentList '-m uol.final overnight --plan "<Root>\jobs\plan.json"' -RedirectStandardOutput '<Root>\logs\overnight.log' -RedirectStandardError '<Root>\logs\overnight.err' -WindowStyle Minimized

Add `--drive-root "<full path>\UOL_FINAL_RUN"` inside the argument string if the folder is not under My Drive. Keep
every path in the argument string in double quotes: Start-Process does not quote, so `exchange\...` would split.
Setup prints the exact line for this computer.

## 4. Watching

- `results\STATUS.json` on Drive (every minute): counts of done / running / failed / blocked / pending, the running
  jobs with their RAM, and a rough ETA. The same file is in `<Root>\run\STATUS.json`.
- Logs: `<Root>\logs\overnight.log` and one log per job in `<Root>\logs\jobs\` (also copied to
  `results\logs\` when the job ends).
- Each finished job's receipt is in `results\jobs\<job>.json`; the results store is mirrored to `results\store\`.

## 5. When something fails

- Environment failure (a missing pinned package, a path, a full disk, a Drive delay): fix the environment, then restart
  the runner with `--retry-failed`; finished jobs are skipped, failed ones run again.
- GPU out of memory: the runner retries once by itself with the fallback batch size listed in `plan.json`.
- Anything else: leave it failed; independent jobs continue; dependants are marked blocked.
- Never edit code, plan, data or configuration, and never rerun a finished job to get a different number.

## 6. End

The runner writes `results\RESULTS.md` (via `uol.final report`) and `results\MANIFEST_RESULTS.json` (sha256 of every
result file) when no job is running or pending. Then write `results\OPERATOR_LOG.md` (machine facts, start and end,
interventions, failed jobs with their error lines, wall time, GPU-hours, peak RAM).

Files outside MANIFEST.json by design (runtime outputs, never data to report): `<Root>\venv\`, `<Root>\work\`,
`<Root>\run\`, `<Root>\logs\`, and the `.npy` arrays that `unpack` restores next to their `.npy.zsh` (each listed in
MANIFEST.json as the `unpacked` file of its packed entry, with its sha256). Anything else outside MANIFEST.json: stop.

Not in this plan: E13 (new cases and TEST-R) runs from `jobs\plan_new_cases.json` after the freeze, with its own small
bundle.
'''


def write(out: Path) -> None:
    out = Path(out)
    (out / "setup").mkdir(parents=True, exist_ok=True)
    (out / "setup/setup_runner.ps1").write_text(SETUP_PS1.replace("\n", "\r\n"), encoding="utf-8")
    (out / "RUNNER_README.md").write_text(README, encoding="utf-8")


def unpack_all(root: Path) -> int:
    """Restore every packed array listed in MANIFEST.json next to it and check its sha256 (exit 0 = all good)."""
    from uol.final import packing
    root = Path(root)
    man = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = []
    n = 0
    for rel, rec in man["files"].items():
        if not rec.get("unpacked"):
            continue
        dst = root / rec["unpacked"]
        if dst.exists() and dst.stat().st_size == rec["unpacked_bytes"]:
            h = hashlib.sha256()
            with open(dst, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
            if h.hexdigest() == rec["unpacked_sha256"]:
                continue
        _, sha = packing.unpack(root / rel, dst)
        n += 1
        if sha != rec["unpacked_sha256"]:
            bad.append(rel)
            dst.unlink(missing_ok=True)
    print(json.dumps({"unpacked": n, "failed": bad}))
    return 1 if bad else 0
