[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$webRoot = Join-Path $projectRoot "web"
$envPath = Join-Path $projectRoot ".env"
$runRoot = Join-Path $projectRoot "data\run"
$logRoot = Join-Path $projectRoot "logs"
$statePath = Join-Path $runRoot "local-services.json"

function Assert-PortAvailable {
    param([int]$Port)

    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $listener) {
        throw "Port $Port is already in use by PID $($listener.OwningProcess). Stop that service first."
    }
}

function Wait-HttpReady {
    param(
        [string]$Uri,
        [int]$TimeoutSeconds = 30
    )

    $handler = [System.Net.Http.HttpClientHandler]::new()
    $handler.UseProxy = $false
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(2)
    try {
        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        while ((Get-Date) -lt $deadline) {
            try {
                $response = $client.GetAsync($Uri).GetAwaiter().GetResult()
                if ([int]$response.StatusCode -ge 200 -and [int]$response.StatusCode -lt 500) {
                    $response.Dispose()
                    return
                }
                $response.Dispose()
            }
            catch {
                Start-Sleep -Milliseconds 500
            }
        }
    }
    finally {
        $client.Dispose()
        $handler.Dispose()
    }
    throw "Timed out waiting for $Uri. Check logs in $logRoot."
}

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

function Stop-StartedProcess {
    param([System.Diagnostics.Process]$Process)

    if ($null -ne $Process -and -not $Process.HasExited) {
        foreach ($descendantId in @(Get-DescendantProcessIds -ParentId $Process.Id)) {
            Stop-Process -Id $descendantId -Force -ErrorAction SilentlyContinue
        }
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python environment not found. Create .venv and install the project first; see README.md."
}
if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
    throw ".env not found. Copy .env.example to .env and fill in model credentials."
}
if (-not (Test-Path -LiteralPath (Join-Path $webRoot "node_modules") -PathType Container)) {
    throw "Frontend dependencies not found. Run 'npm ci' in the web directory first."
}
if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    throw "A local service state file already exists. Run scripts\stop_local.ps1 first."
}

Assert-PortAvailable -Port 8000
Assert-PortAvailable -Port 3000
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

$apiProcess = $null
$webProcess = $null
try {
    $apiProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @("-m", "uvicorn", "hardsec_scholar.api.app:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput (Join-Path $logRoot "api.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "api.stderr.log") `
        -WindowStyle Hidden `
        -PassThru

    $escapedWebRoot = $webRoot.Replace("'", "''")
    $webCommand = "Set-Location -LiteralPath '$escapedWebRoot'; & npm.cmd run dev"
    $webProcess = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-Command", $webCommand) `
        -WorkingDirectory $webRoot `
        -RedirectStandardOutput (Join-Path $logRoot "web.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "web.stderr.log") `
        -WindowStyle Hidden `
        -PassThru

    Wait-HttpReady -Uri "http://127.0.0.1:8000/api/health"
    Wait-HttpReady -Uri "http://localhost:3000"

    $state = [ordered]@{
        project_root = $projectRoot
        created_at = (Get-Date).ToString("o")
        api_pid = $apiProcess.Id
        api_started_at = $apiProcess.StartTime.ToString("o")
        web_pid = $webProcess.Id
        web_started_at = $webProcess.StartTime.ToString("o")
    }
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
}
catch {
    Stop-StartedProcess -Process $webProcess
    Stop-StartedProcess -Process $apiProcess
    throw
}

Write-Host "HardSec Scholar is ready."
Write-Host "Web:      http://localhost:3000"
Write-Host "API docs: http://127.0.0.1:8000/docs"
Write-Host "Logs:     $logRoot"
Write-Host "Stop:     .\scripts\stop_local.ps1"
