[CmdletBinding()]
param(
    [string]$RepositoryRoot = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = Split-Path -Parent $PSScriptRoot
}

$root = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$null = & git -C $root rev-parse --git-dir 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Not a Git repository: $root"
}

$tracked = @(& git -C $root ls-files)
if ($LASTEXITCODE -ne 0) {
    throw 'git ls-files failed.'
}

$problems = New-Object System.Collections.Generic.List[string]
$forbiddenPathPatterns = @(
    '(?i)^app/config\.json$',
    '(?i)^app/comfy_paths\.yaml$',
    '(?i)^app/action_presets\.json$',
    '(?i)^backend/h3_v2/config\.json$',
    '(?i)^backend/h3_v2/paths_rnd\.yaml$',
    '(?i)^_hetero_test/hetero_paths\.yaml$',
    '(?i)(^|/)(H3_Media|projects|story_projects|character_library|audio_tests|_debug|benchmarks|_backup|_archive)(/|$)',
    '(?i)(^|/)(output|outputs|cache|temp|tmp|node_modules|__pycache__)(/|$)',
    '(?i)(^|/)\.brain(/|$)'
)

$forbiddenExtensions = @(
    '.mp4', '.mov', '.avi', '.mkv', '.webm',
    '.wav', '.mp3', '.flac', '.m4a', '.aac',
    '.safetensors', '.ckpt', '.pt', '.pth', '.gguf', '.onnx', '.engine',
    '.db', '.sqlite', '.sqlite3', '.log', '.err', '.jsonl',
    '.exe', '.lnk', '.pfx', '.p12', '.pem', '.key', '.zip', '.7z', '.rar'
)

$textExtensions = @(
    '.py', '.ps1', '.bat', '.cmd', '.cs', '.js', '.json', '.yaml', '.yml',
    '.toml', '.ini', '.cfg', '.md', '.txt', '.html', '.css', '.csv'
)

$contentRules = @(
    @{ Name = 'private key'; Pattern = '-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----' },
    @{ Name = 'GitHub token'; Pattern = '(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})' },
    @{ Name = 'OpenAI-style secret'; Pattern = '\bsk-[A-Za-z0-9_-]{20,}' },
    @{ Name = 'AWS access key'; Pattern = '\bAKIA[0-9A-Z]{16}\b' },
    @{ Name = 'Google API key'; Pattern = '\bAIza[0-9A-Za-z_-]{30,}' },
    @{ Name = 'Bearer credential'; Pattern = '(?i)authorization\s*[:=]\s*["'']?Bearer\s+[A-Za-z0-9._~-]{16,}' },
    @{ Name = 'credential assignment'; Pattern = '(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)\s*["'']?\s*[:=]\s*["''][^"'']{8,}["'']' },
    @{ Name = 'credential in URL'; Pattern = 'https?://[^\s/:]+:[^\s/@]+@' },
    @{ Name = 'email address'; Pattern = '(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b' },
    @{ Name = 'absolute Windows user path'; Pattern = '(?i)[A-Z]:\\Users\\(?!<)[^\\\s]+' }
)

foreach ($path in $tracked) {
    $normalized = $path.Replace('\', '/')
    foreach ($pattern in $forbiddenPathPatterns) {
        if ($normalized -match $pattern) {
            $problems.Add("forbidden path: $normalized")
            break
        }
    }

    $extension = [IO.Path]::GetExtension($normalized).ToLowerInvariant()
    if ($forbiddenExtensions -contains $extension) {
        $problems.Add("forbidden extension: $normalized")
    }

    $full = Join-Path $root $path
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
        $problems.Add("tracked file missing from worktree: $normalized")
        continue
    }

    $length = (Get-Item -LiteralPath $full).Length
    if ($length -gt 20MB) {
        $problems.Add("tracked file exceeds 20 MiB: $normalized")
    }

    if ($normalized -eq 'scripts/prepublish-check.ps1') {
        continue
    }
    if (($textExtensions -notcontains $extension) -or $length -gt 5MB) {
        continue
    }

    try {
        $content = [IO.File]::ReadAllText($full)
    }
    catch {
        $problems.Add("could not scan text file: $normalized")
        continue
    }
    foreach ($rule in $contentRules) {
        if ($content -match $rule.Pattern) {
            $problems.Add("$($rule.Name): $normalized")
        }
    }
}

$problems = @($problems | Sort-Object -Unique)
if ($problems.Count -gt 0) {
    Write-Host 'Pre-publish check FAILED. No secret values are printed.' -ForegroundColor Red
    $problems | ForEach-Object { Write-Host "  - $_" }
    exit 1
}

Write-Host ("Pre-publish check PASS: {0} tracked files scanned." -f $tracked.Count) -ForegroundColor Green
