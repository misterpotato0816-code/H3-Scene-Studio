$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Formal v1 entry point. Keep the measured implementation untouched and call it
# through this stable name so the verified graph and its SHA gates stay intact.
$VerifiedEntry = Join-Path $PSScriptRoot 'RUN_OPT_REF_MATCH.ps1'

if(-not (Test-Path -LiteralPath $VerifiedEntry)) {
    throw "Verified H3 heterogeneous v1 entry was not found: $VerifiedEntry"
}

Write-Host 'H3 heterogeneous GPU v1 (RTX 3060 conditioning -> RTX 4070 Ti DiT/VAE)'
Write-Host 'Fixed profile: 576x1024 / 124 frames / 8 steps / seed 123456789 / ref_image_size=match'
Write-Host 'This starts one full H3 video generation. Existing ComfyUI processes are not killed.'

& $VerifiedEntry

