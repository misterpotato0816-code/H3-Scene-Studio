$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# H3 App launcher.
#
# Starts the local app server only. It never starts a generation by itself, and
# it never kills an existing ComfyUI: if the measured runners (port 8401/8402)
# or a Desktop instance are already running, they are left alone.
#
# ASCII ONLY - do not add non-ASCII text to this file.
# Windows PowerShell 5.1 reads a .ps1 without a BOM using the system ANSI code
# page, so UTF-8 Japanese here decodes as mojibake and breaks string terminators
# (ParserError). The file is also saved UTF-8 with BOM as a second line of
# defence. The application UI stays Japanese; only this launcher is English.

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $AppDir 'config.json'
$Server = Join-Path $AppDir 'server.py'
$PathsYaml = Join-Path $AppDir 'comfy_paths.yaml'
$ProjectRoot = Split-Path -Parent $AppDir
$BarrierNodes = Join-Path $ProjectRoot 'custom_nodes\H3-Device-Barrier'
$Manifest = Join-Path $ProjectRoot 'H3_HETERO_V1_MANIFEST.json'

Write-Host ''
Write-Host 'H3 Video Maker' -ForegroundColor Cyan
Write-Host '  Makes a talking vertical video from reference images and a Japanese instruction.'
Write-Host '  Default profile (576x1024 / 124 frames / 8 steps) takes about 3 min 40 sec per clip.'
Write-Host '  Any ComfyUI that is already running will NOT be stopped.'
Write-Host ''

# ---- preflight -----------------------------------------------------------
foreach($p in @($ConfigPath, $Server, $PathsYaml)) {
    if(-not (Test-Path -LiteralPath $p)) { throw "Required file not found: $p" }
}
if(-not (Test-Path -LiteralPath $Manifest)) {
    throw "v1 manifest not found: $Manifest"
}
if(-not (Test-Path -LiteralPath $BarrierNodes)) {
    throw "H3-Device-Barrier not found: $BarrierNodes -- without it the CUDA phase barrier is inactive and conditioning will OOM."
}

$cfg = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json

$ComfyDir = $cfg.comfy_dir
if(-not (Test-Path -LiteralPath $ComfyDir)) { throw "ComfyUI directory not found: $ComfyDir" }

$ComfyMain = Join-Path $ComfyDir 'main.py'
if(-not (Test-Path -LiteralPath $ComfyMain)) { throw "ComfyUI main.py not found: $ComfyMain" }

$Python = $cfg.comfy_python
if([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $ComfyDir '.venv\Scripts\python.exe'
}
if(-not (Test-Path -LiteralPath $Python)) { throw "ComfyUI python not found: $Python" }

$AppPort = [int]$cfg.app_port
$listener = Get-NetTCPConnection -State Listen -LocalPort $AppPort -ErrorAction SilentlyContinue
if($listener) {
    Write-Host "Port $AppPort is already in use." -ForegroundColor Yellow
    Write-Host 'The app may already be running. Open this in your browser:'
    Write-Host ("  http://127.0.0.1:{0}/" -f $AppPort)
    Write-Host 'If it is a different process, change app_port in config.json. Nothing is killed.'
    exit 1
}

# Model files are only a warning: the VLM part of the app still works without
# the 20 GiB H3 weights, and the app reports missing models per node at submit.
$SharedModels = 'E:\Comfy-Desktop\ComfyUI-Shared\models'
$modelChecks = @(
    @{ Path = (Join-Path $SharedModels ('diffusion_models\' + $cfg.models.unet)); Name = 'H3 model' },
    @{ Path = (Join-Path $SharedModels ('text_encoders\' + $cfg.models.clip));    Name = 'Text encoder' },
    @{ Path = (Join-Path $SharedModels ('vae\' + $cfg.models.vae_video));         Name = 'Video VAE' },
    @{ Path = (Join-Path $SharedModels ('vae\' + $cfg.models.vae_audio));         Name = 'Audio VAE' }
)
foreach($m in $modelChecks) {
    if(-not (Test-Path -LiteralPath $m.Path)) {
        Write-Warning ("{0} not found at the default location: {1}" -f $m.Name, $m.Path)
        Write-Warning '  (Fine if you keep it elsewhere. The app re-checks when generation starts.)'
    }
}

# ---- start ---------------------------------------------------------------
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch {}

$url = "http://127.0.0.1:$AppPort/"
Write-Host "Starting the app: $url" -ForegroundColor Green
Write-Host 'Press Ctrl+C in this window to stop it.'
Write-Host ''

# Open the browser once the server is listening, without blocking startup.
$opener = Start-Job -ScriptBlock {
    param($u, $p)
    for($i = 0; $i -lt 60; $i++) {
        if(Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue) {
            Start-Process $u
            return
        }
        Start-Sleep -Milliseconds 500
    }
} -ArgumentList $url, $AppPort

try {
    & $Python -X utf8 $Server
    if($LASTEXITCODE -ne 0) { throw "The app server exited with code $LASTEXITCODE." }
}
finally {
    Remove-Job -Job $opener -Force -ErrorAction SilentlyContinue
}
