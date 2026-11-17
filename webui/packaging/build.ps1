<#
  PingPong Studio 打包脚本
  =========================
  用法：
    cd webui
    .\packaging\build.ps1                              # 仅 exe 包（不含 ffmpeg，约 303 MB）
    .\packaging\build.ps1 -WithFFmpeg                  # 仅 exe 包（含 ffmpeg，约 736 MB）
    .\packaging\build.ps1 -WithFFmpeg -Installer       # exe 包 + 安装程序
    .\packaging\build.ps1 -Installer -SkipBuild        # 用现有 dist 只做安装程序

  产物：
    webui\dist\PingPongStudio\                          （onedir 目录）
    webui\dist\installer\PingPongStudio-0.0.1-setup.exe （-Installer 时）
#>
param(
    [switch]$WithFFmpeg,
    [switch]$Clean,
    [switch]$Installer,
    [switch]$SkipBuild,
    [string]$InnoPath = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot        # webui/
Set-Location $Root

# ---------------- 1. PyInstaller ----------------
if (-not $SkipBuild) {
    if ($WithFFmpeg) {
        $env:PPT_BUNDLE_FFMPEG = "1"
        Write-Host "[build] 内置 ffmpeg/ffprobe" -ForegroundColor Cyan
    } else {
        Remove-Item Env:\PPT_BUNDLE_FFMPEG -ErrorAction SilentlyContinue
        Write-Host "[build] 不内置 ffmpeg（目标机需自行安装）" -ForegroundColor Yellow
    }

    $pyiArgs = @(
        "packaging\app.spec",
        "--noconfirm",
        "--distpath", "dist",
        "--workpath", "build"
    )
    if ($Clean) { $pyiArgs += "--clean" }

    & ".\.venv\Scripts\pyinstaller.exe" @pyiArgs
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 失败 (exit $LASTEXITCODE)" }

    Copy-Item "packaging\使用说明.txt" "dist\PingPongStudio\使用说明.txt" -Force

    $dir = "dist\PingPongStudio"
    $size = (Get-ChildItem $dir -Recurse -File | Measure-Object -Property Length -Sum).Sum
    Write-Host ("[build] exe 包: {0:N1} MB / {1} 个文件" -f ($size / 1MB), (Get-ChildItem $dir -Recurse -File).Count) -ForegroundColor Green
}

# ---------------- 2. Inno Setup ----------------
if ($Installer) {
    $candidates = @()
    if ($InnoPath) { $candidates += $InnoPath }
    $candidates += @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe"
    )
    $iscc = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $iscc) {
        throw "找不到 Inno Setup 的 ISCC.exe，请先安装 Inno Setup 6（或传 -InnoPath）"
    }

    Write-Host "[build] 编译安装程序（大文件压缩，可能几分钟）..." -ForegroundColor Cyan
    & $iscc "packaging\app.iss"
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup 失败 (exit $LASTEXITCODE)" }

    $setup = Get-ChildItem "dist\installer\*.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    Write-Host ("[build] 安装包: {0} ({1:N1} MB)" -f $setup.FullName, ($setup.Length / 1MB)) -ForegroundColor Green
}
