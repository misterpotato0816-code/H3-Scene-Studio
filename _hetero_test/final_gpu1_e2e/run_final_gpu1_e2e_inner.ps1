$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Project = 'E:\AI-Projects\H3'
$Comfy = 'E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI'
$Python = Join-Path $Comfy '.venv\Scripts\python.exe'
$Main = Join-Path $Comfy 'main.py'
$Paths = Join-Path $Project '_hetero_test\hetero_paths.yaml'
$Runner = Join-Path $Root 'run_final_gpu1_e2e.py'
$Log = Join-Path $Root 'comfy_final.log'
$Err = Join-Path $Root 'comfy_final.err'
$Db = Join-Path $Root 'final.db'
$Port = 8401

foreach($p in @($Python,$Main,$Paths,$Runner)) {
    if(-not (Test-Path -LiteralPath $p)) { throw "Required file not found: $p" }
}

$listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if($listener) {
    throw "Port $Port is already in use. This test will not kill an existing ComfyUI process."
}

Remove-Item -LiteralPath $Log,$Err,$Db,"$Db-lock" -Force -ErrorAction SilentlyContinue

# Force UTF-8 for both ComfyUI and the test runner. LayerStyle logs emoji; CP932 crashes on it.
$oldPythonUtf8 = $env:PYTHONUTF8
$oldPythonIoEncoding = $env:PYTHONIOENCODING
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch {}

Write-Host '[1/3] Starting isolated ComfyUI...'
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
    Write-Host '[2/3] Running LIVE heterogeneous H3: GPU1 TE -> GPU0 DiT/VAE...'
    & $Python -X utf8 $Runner
    if($LASTEXITCODE -ne 0) { throw "Runner failed with exit code $LASTEXITCODE" }

    Write-Host ''
    Write-Host '[3/3] Finished.'
    Write-Host ("Result: " + (Join-Path $Root 'final_result.json'))
    Write-Host ("Video:  " + (Join-Path $Comfy 'output\H3\20260814_FINAL'))
    Write-Host ("Log:    " + $Err)
}
finally {
    if($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $proc.Id -ErrorAction SilentlyContinue
    }
    $env:PYTHONUTF8 = $oldPythonUtf8
    $env:PYTHONIOENCODING = $oldPythonIoEncoding
}
