$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Project = 'E:\AI-Projects\H3'
$Comfy = 'E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI'
$Python = Join-Path $Comfy '.venv\Scripts\python.exe'
$Main = Join-Path $Comfy 'main.py'
$Paths = Join-Path $Project '_hetero_test\hetero_paths.yaml'
$Runner = Join-Path $Root 'run_opt_ref_match.py'
$Sheet = Join-Path $Root 'make_compare_sheet.py'
$Log = Join-Path $Root 'comfy_opt.log'
$Err = Join-Path $Root 'comfy_opt.err'
$Db = Join-Path $Root 'opt.db'
$Port = 8402

foreach($p in @($Python,$Main,$Paths,$Runner,$Sheet)) {
    if(-not (Test-Path -LiteralPath $p)) { throw "Required file not found: $p" }
}

$listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if($listener) {
    throw "Port $Port is already in use. This test will not kill an existing process."
}

Remove-Item -LiteralPath $Log,$Err,$Db,"$Db-lock" -Force -ErrorAction SilentlyContinue

$oldPythonUtf8 = $env:PYTHONUTF8
$oldPythonIoEncoding = $env:PYTHONIOENCODING
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch {}

Write-Host '[1/4] Starting isolated ComfyUI...'
$args = @(
    '-X', 'utf8',
    $Main,
    '--port', "$Port",
    '--default-device', '0',
    '--reserve-vram', '1.0',
    '--extra-model-paths-config', $Paths,
    '--disable-all-custom-nodes',
    '--whitelist-custom-nodes', 'H3-GOLD-Conditioning-Cache', 'comfyui_layerstyle', 'comfyui-kjnodes-hetero-test',
    '--database-url', ('sqlite:///' + ($Db -replace '\\','/'))
)

$proc = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Comfy `
    -RedirectStandardOutput $Log -RedirectStandardError $Err -PassThru -NoNewWindow

try {
    Write-Host '[2/4] Running ONE candidate only: ref_image_size max -> match'
    Write-Host '      Everything else remains identical to the successful 308.37 s build.'
    & $Python -X utf8 $Runner
    if($LASTEXITCODE -ne 0) { throw "Runner failed with exit code $LASTEXITCODE" }

    Write-Host '[3/4] Building CURRENT-vs-CANDIDATE contact sheet...'
    & $Python -X utf8 $Sheet
    if($LASTEXITCODE -ne 0) {
        Write-Warning 'Candidate video succeeded, but contact-sheet creation failed. Compare the two MP4 files manually.'
    }

    Write-Host ''
    Write-Host '[4/4] Finished.'
    Write-Host ("Result:  " + (Join-Path $Root 'opt_result.json'))
    Write-Host ("Video:   " + (Join-Path $Comfy 'output\H3\20260814_OPT'))
    Write-Host ("Compare: " + (Join-Path $Root 'current_max_left_candidate_match_right.jpg'))
    Write-Host ''
    Write-Host 'QUALITY GATE: current MAX = left / candidate MATCH = right.'
    Write-Host 'If face/identity/motion/lips are even slightly worse, reject MATCH and keep the current build.'
}
finally {
    if($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $proc.Id -ErrorAction SilentlyContinue
    }
    $env:PYTHONUTF8 = $oldPythonUtf8
    $env:PYTHONIOENCODING = $oldPythonIoEncoding
}
