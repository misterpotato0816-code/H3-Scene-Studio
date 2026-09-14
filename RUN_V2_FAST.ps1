$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# H3 Shorts Studio v2 entry: single FAST generation (Ref2VA Turbo 4-step).
# v1 launchers are untouched. Isolated ComfyUI on :8402, existing processes kept.
$VenvPy = 'E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe'
$Harness = Join-Path $PSScriptRoot 'backend\h3_v2\ab_run.py'

foreach($p in @($VenvPy,$Harness)) {
    if(-not (Test-Path -LiteralPath $p)) { throw "Required file not found: $p" }
}

Write-Host 'H3 Shorts Studio v2 FAST: Ref2VA Turbo 4-step (euler/simple/SigmaShift 12,3)'
Write-Host 'Same 576x1024 / 124f / seed 123456789 / gold prompt+ref as the v1 A/B.'
& $VenvPy $Harness --arm fast --port 8402
if($LASTEXITCODE -ne 0) { throw "v2 FAST generation failed with exit code $LASTEXITCODE" }
