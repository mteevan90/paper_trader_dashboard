<#
Live tracker for the Phase 3 tuning run.

Parses models/studies/larger_universe_v1/phase3_progress.log as it grows
and prints a one-line-per-trial summary with: trial number, current IC,
running best IC + which trial set it, mean seconds per trial, and ETA.
Green = new best. Yellow = phase boundary. Cyan = convergence checkpoint.
Red = error. Dark gray = misc info lines.

Usage (from the repo root):
    powershell -ExecutionPolicy Bypass -File .\scripts\watch_phase3.ps1

Or with an explicit log path:
    powershell -ExecutionPolicy Bypass -File .\scripts\watch_phase3.ps1 -LogPath "..\custom\path.log"

Compatible with Windows PowerShell 5.1 and PowerShell 7+. Press Ctrl+C to stop.
#>
param(
    [string]$LogPath = "models\studies\larger_universe_v1\phase3_progress.log"
)

if (-not (Test-Path $LogPath)) {
    Write-Host "Log file not found: $LogPath" -ForegroundColor Red
    Write-Host "Pass a path with -LogPath, or wait — the runner creates it after a few seconds." -ForegroundColor DarkGray
    Write-Host "Polling for the file to appear..." -ForegroundColor DarkGray
    while (-not (Test-Path $LogPath)) {
        Start-Sleep -Seconds 2
    }
}

$LogPath = (Resolve-Path $LogPath).Path
Write-Host ""
Write-Host "Watching: $LogPath" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

# Per-model running state
$xgb  = @{ count = 0; total_s = 0.0; best_ic = [double]::NaN; best_trial = 0; total = 200 }
$enet = @{ count = 0; total_s = 0.0; best_ic = [double]::NaN; best_trial = 0; total = 100 }

# Regex patterns for the log lines we care about
$reXgbStart  = [regex]'=== XGBoost \(\d+ trials'
$reEnetStart = [regex]'=== ElasticNet \(\d+ trials'
$reXgb       = [regex]'XGB trial (\d+)/(\d+)\s+ic=(-?\d+\.\d+)\s+\(best so far (-?\d+\.\d+|nan)\)\s+(\d+\.\d+)s'
$reEnet      = [regex]'ENet trial (\d+)/(\d+)\s+ic=(-?\d+\.\d+)\s+\(best so far (-?\d+\.\d+|nan)\)\s+(\d+\.\d+)s'
$reConv      = [regex]'(XGB|ENet) convergence @ trial (\d+): running_best=(-?\d+\.\d+)'

function Format-Eta {
    param([double]$Seconds)
    if ($Seconds -ge 3600) {
        $h = [math]::Floor($Seconds / 3600)
        $m = [math]::Floor(($Seconds % 3600) / 60)
        return "{0}h{1:00}m" -f $h, $m
    }
    if ($Seconds -ge 60) {
        $m = [math]::Floor($Seconds / 60)
        $s = [math]::Floor($Seconds % 60)
        return "{0}m{1:00}s" -f $m, $s
    }
    return "{0:0}s" -f $Seconds
}

function Write-TrialLine {
    param([string]$Label, [hashtable]$St, [double]$Ic, [double]$Elapsed, [bool]$IsNewBest)
    $meanS = $St.total_s / [double]$St.count
    $remain = $St.total - $St.count
    $etaS = $meanS * [double]$remain
    $etaStr = Format-Eta -Seconds $etaS
    $color = if ($IsNewBest) { 'Green' } else { 'White' }
    $msg = "[{0} {1,3}/{2}]  ic={3,8:N4}  best={4,8:N4} (t{5,3})  mean={6,4:N0}s  ETA {7}" -f `
        $Label, $St.count, $St.total, $Ic, $St.best_ic, $St.best_trial, $meanS, $etaStr
    Write-Host $msg -ForegroundColor $color
}

Get-Content -Path $LogPath -Wait -Tail 100 | ForEach-Object {
    $line = $_
    if ([string]::IsNullOrWhiteSpace($line)) { return }

    if ($reXgbStart.IsMatch($line)) {
        Write-Host ""
        Write-Host "=== XGBoost phase started ===" -ForegroundColor Yellow
        return
    }
    if ($reEnetStart.IsMatch($line)) {
        Write-Host ""
        Write-Host "=== ElasticNet phase started ===" -ForegroundColor Yellow
        return
    }

    $m = $reXgb.Match($line)
    if ($m.Success) {
        $n = [int]$m.Groups[1].Value
        $tot = [int]$m.Groups[2].Value
        $ic = [double]$m.Groups[3].Value
        $elapsed = [double]$m.Groups[5].Value
        $xgb.count = $n
        $xgb.total = $tot
        $xgb.total_s = $xgb.total_s + $elapsed
        $isNewBest = $false
        if ([double]::IsNaN($xgb.best_ic) -or $ic -gt $xgb.best_ic) {
            $xgb.best_ic = $ic
            $xgb.best_trial = $n
            $isNewBest = $true
        }
        Write-TrialLine -Label "XGB " -St $xgb -Ic $ic -Elapsed $elapsed -IsNewBest $isNewBest
        return
    }

    $m = $reEnet.Match($line)
    if ($m.Success) {
        $n = [int]$m.Groups[1].Value
        $tot = [int]$m.Groups[2].Value
        $ic = [double]$m.Groups[3].Value
        $elapsed = [double]$m.Groups[5].Value
        $enet.count = $n
        $enet.total = $tot
        $enet.total_s = $enet.total_s + $elapsed
        $isNewBest = $false
        if ([double]::IsNaN($enet.best_ic) -or $ic -gt $enet.best_ic) {
            $enet.best_ic = $ic
            $enet.best_trial = $n
            $isNewBest = $true
        }
        Write-TrialLine -Label "ENet" -St $enet -Ic $ic -Elapsed $elapsed -IsNewBest $isNewBest
        return
    }

    $m = $reConv.Match($line)
    if ($m.Success) {
        Write-Host ("  >> {0} convergence @ trial {1}: running_best={2}" -f `
            $m.Groups[1].Value, $m.Groups[2].Value, $m.Groups[3].Value) -ForegroundColor Cyan
        return
    }

    if ($line -match 'ERROR|Traceback|Exception') {
        Write-Host "ERR: $line" -ForegroundColor Red
        return
    }
    if ($line -match 'WARNING') {
        Write-Host "WARN: $line" -ForegroundColor Yellow
        return
    }

    # Show only the meaningful info lines (best IC summary, phase done, fold breakdowns)
    if ($line -match 'best mean cross-sec IC|done in|fold \d+\s+n=|Phase 3') {
        Write-Host $line -ForegroundColor DarkGray
    }
}
