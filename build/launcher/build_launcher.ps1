# build_launcher.ps1
#
# Reproducible build script for H3_Video_Maker_v2.exe.
#
# Compiles build\launcher\H3Launcher.cs with the .NET Framework C# compiler
# already present on this machine (no downloads, no NuGet, no SDK) and
# writes the resulting Windows GUI (WinForms, no console window) executable
# to the project root.
#
# ASCII only, matching the rule documented in app\RUN_H3_APP.ps1.

$ErrorActionPreference = "Stop"

$cscPath = "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"

if (-not (Test-Path $cscPath)) {
    Write-Error "csc.exe was not found at: $cscPath. Cannot build without the .NET Framework compiler."
    exit 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..\..")

$sourcePath = Join-Path $scriptDir "H3Launcher.cs"
if (-not (Test-Path $sourcePath)) {
    Write-Error "Source file not found: $sourcePath"
    exit 1
}

$outputPath = Join-Path $projectRoot "H3_Video_Maker_v2.exe"

Write-Host "Compiling $sourcePath"
Write-Host "Output: $outputPath"

& $cscPath /nologo /optimize+ /platform:anycpu /target:winexe /r:System.Windows.Forms.dll /r:System.Drawing.dll "/out:$outputPath" $sourcePath

if ($LASTEXITCODE -ne 0) {
    Write-Error "csc.exe failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

if (-not (Test-Path $outputPath)) {
    Write-Error "Build reported success but output file was not found: $outputPath"
    exit 1
}

$size = (Get-Item $outputPath).Length
Write-Host "Build succeeded."
Write-Host "Output path: $outputPath"
Write-Host "Size: $size bytes"
