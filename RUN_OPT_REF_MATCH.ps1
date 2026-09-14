$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = 'E:\AI-Projects\H3\_hetero_test\opt_ref_match'
$Runner = Join-Path $Root 'run_opt_ref_match.py'
$Launch = Join-Path $Root 'run_opt_ref_match_inner.ps1'
$Sheet = Join-Path $Root 'make_compare_sheet.py'

foreach($p in @($Runner,$Launch,$Sheet)) {
    if(-not (Test-Path -LiteralPath $p)) { throw "Required file not found: $p" }
}

& $Launch
if($LASTEXITCODE -ne 0) { throw "REF-MATCH candidate test failed with exit code $LASTEXITCODE" }
