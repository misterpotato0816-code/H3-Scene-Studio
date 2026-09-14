$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = 'E:\AI-Projects\H3\_hetero_test\final_gpu1_e2e'
$Runner = Join-Path $Root 'run_final_gpu1_e2e.py'
$Launch = Join-Path $Root 'run_final_gpu1_e2e_inner.ps1'

foreach($p in @($Runner,$Launch)) {
    if(-not (Test-Path -LiteralPath $p)) { throw "Required file not found: $p" }
}

& $Launch
if($LASTEXITCODE -ne 0) { throw "Final GPU1 E2E test failed with exit code $LASTEXITCODE" }
