# PowerShell script for graph creation
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Change to script directory
Set-Location $PSScriptRoot
Set-Location ..

Write-Host "========================================"
Write-Host "Graph Creation Script"
Write-Host "========================================"
Write-Host ""

Write-Host "Creating graphs..."
$currentDir = Get-Location
$scriptPath = "$currentDir\ソース\create_graphs.py"
Write-Host "Script path: $scriptPath"
python $scriptPath

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to create graphs"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================"
Write-Host "Graph creation completed!"
Write-Host "========================================"
Write-Host ""
Read-Host "Press Enter to exit"

