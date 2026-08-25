param(
    [int]$Parallel = 16
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$downloadRoot = Join-Path $projectRoot "work\cache\prm800k_phase2_download"
$target = Join-Path $projectRoot "work\vendor\prm800k\prm800k\data\phase2_train.jsonl"
$url = "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/phase2_train.jsonl"
$expectedSha256 = "1110237feeb51d1bc200cb37b8f965cfdc1036eac7d506094049366fe7dc1089"
[int64]$expectedSize = 456135365

New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
$resolvedDownloadRoot = (Resolve-Path $downloadRoot).Path
$resolvedProjectRoot = (Resolve-Path $projectRoot).Path
if (-not $resolvedDownloadRoot.StartsWith($resolvedProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe download directory: $resolvedDownloadRoot"
}

[int64]$chunkSize = [math]::Ceiling($expectedSize / $Parallel)
$parts = @()
for ($index = 0; $index -lt $Parallel; $index++) {
    [int64]$start = $index * $chunkSize
    [int64]$end = [math]::Min($expectedSize - 1, ($index + 1) * $chunkSize - 1)
    $part = Join-Path $downloadRoot ("part_{0:D2}.bin" -f $index)
    $parts += [pscustomobject]@{ Path = $part; Start = $start; End = $end }
}

$pending = @(
    $parts | Where-Object {
        -not (Test-Path -LiteralPath $_.Path) -or
        (Get-Item -LiteralPath $_.Path).Length -ne ($_.End - $_.Start + 1)
    }
)
$curlArgs = @("--parallel", "--parallel-immediate", "--parallel-max", $Parallel.ToString())
for ($index = 0; $index -lt $pending.Count; $index++) {
    $part = $pending[$index]
    $curlArgs += @(
        "--fail", "--silent", "--show-error",
        "--retry", "20", "--retry-all-errors", "--connect-timeout", "20",
        "--speed-time", "30", "--speed-limit", "1024",
        "--range", "$($part.Start)-$($part.End)", "--output", $part.Path, $url
    )
    if ($index -lt $pending.Count - 1) {
        $curlArgs += "--next"
    }
}

if ($pending.Count -gt 0) {
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Parallel curl failed with exit code $LASTEXITCODE"
    }
}

foreach ($part in $parts) {
    [int64]$expectedPartSize = $part.End - $part.Start + 1
    [int64]$actualPartSize = (Get-Item -LiteralPath $part.Path).Length
    if ($actualPartSize -ne $expectedPartSize) {
        throw "Chunk size mismatch for $($part.Path): expected=$expectedPartSize actual=$actualPartSize"
    }
}

$complete = Join-Path $downloadRoot "phase2_train.jsonl.complete"
$output = [System.IO.File]::Open(
    $complete,
    [System.IO.FileMode]::Create,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
try {
    foreach ($part in $parts) {
        $input = [System.IO.File]::OpenRead($part.Path)
        try {
            $input.CopyTo($output)
        }
        finally {
            $input.Dispose()
        }
    }
}
finally {
    $output.Dispose()
}

if ((Get-Item -LiteralPath $complete).Length -ne $expectedSize) {
    throw "Combined file size mismatch"
}
$actualSha256 = (Get-FileHash -LiteralPath $complete -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw "Combined file SHA-256 mismatch: $actualSha256"
}

$resolvedTargetParent = (Resolve-Path (Split-Path -Parent $target)).Path
if (-not $resolvedTargetParent.StartsWith($resolvedProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe target directory: $resolvedTargetParent"
}
[System.IO.File]::Copy($complete, $target, $true)
Get-Item -LiteralPath $target | Select-Object FullName, Length, LastWriteTime
