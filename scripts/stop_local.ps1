[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$statePath = Join-Path $projectRoot "data\run\local-services.json"

function Get-DescendantProcessIds {
    param([int]$ParentId)

    $result = [System.Collections.Generic.List[int]]::new()
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        foreach ($descendantId in Get-DescendantProcessIds -ParentId ([int]$child.ProcessId)) {
            $result.Add($descendantId)
        }
        $result.Add([int]$child.ProcessId)
    }
    return $result
}

function Stop-RecordedTree {
    param(
        [string]$Name,
        [int]$ProcessId,
        [datetime]$ExpectedStartTime
    )

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Write-Host "$Name process $ProcessId is already stopped."
        return
    }

    if ([Math]::Abs(($process.StartTime - $ExpectedStartTime).TotalSeconds) -gt 2) {
        Write-Warning "$Name PID $ProcessId has been reused; it will not be stopped."
        return
    }

    $descendants = @(Get-DescendantProcessIds -ParentId $ProcessId)
    foreach ($descendantId in $descendants) {
        Stop-Process -Id $descendantId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped $Name process tree (PID $ProcessId)."
}

if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    Write-Host "No HardSec Scholar service state file was found; nothing to stop."
    exit 0
}

$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$recordedRoot = [System.IO.Path]::GetFullPath([string]$state.project_root)
if (-not [string]::Equals($recordedRoot, $projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "The service state belongs to another project directory and will not be used."
}

Stop-RecordedTree -Name "web" -ProcessId ([int]$state.web_pid) -ExpectedStartTime ([datetime]$state.web_started_at)
Stop-RecordedTree -Name "API" -ProcessId ([int]$state.api_pid) -ExpectedStartTime ([datetime]$state.api_started_at)
Remove-Item -LiteralPath $statePath -Force
Write-Host "HardSec Scholar local services are stopped."
