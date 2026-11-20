<#
  TTComet 打包脚本
  =================
  用法：
    cd webui
    .\packaging\build.ps1                              # 仅 exe 包（不含 ffmpeg，约 303 MB）
    .\packaging\build.ps1 -WithFFmpeg                  # 仅 exe 包（含 ffmpeg，约 736 MB）
    .\packaging\build.ps1 -WithFFmpeg -Installer       # exe 包 + 安装程序
    .\packaging\build.ps1 -Installer -SkipBuild        # 用现有 dist 只做安装程序

  签名（可选）：
    .\packaging\build.ps1 -WithFFmpeg -Installer -Pfx C:\secure\cert.pfx -Password 123456
    .\packaging\build.ps1 -WithFFmpeg -Installer -Thumbprint ABCDEF...
    .\packaging\build.ps1 -WithFFmpeg -Installer -SelfSigned      # 仅内部测试用

  产物：
    webui\dist\TTComet\                                  （onedir 目录）
    webui\dist\installer\TTComet-0.2.1-setup.exe         （-Installer 时）
    webui\dist\SHA256SUMS.txt                            （始终生成，用于校验下载完整性）
#>
param(
    [switch]$WithFFmpeg,
    [switch]$Clean,
    [switch]$Installer,
    [switch]$SkipBuild,
    [string]$InnoPath = "",
    # 签名相关
    [string]$Pfx = "",
    [string]$Password = "",
    [string]$Thumbprint = "",
    [switch]$SelfSigned
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot        # webui/
Set-Location $Root

$wantSign = ($Pfx -ne "") -or ($Thumbprint -ne "") -or $SelfSigned

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

    Copy-Item "packaging\使用说明.txt" "dist\TTComet\使用说明.txt" -Force

    $dir = "dist\TTComet"
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
}

# ---------------- 3. 签名 ----------------
if ($wantSign) {
    # 只签不带私钥的可执行文件：exe + 安装包（DLL 太多，逐个签会很慢，且对 SmartScreen 意义不大）
    $signArgs = @{ Target = "dist\TTComet" }
    if ($Pfx) { $signArgs.Pfx = $Pfx }
    if ($Password) { $signArgs.Password = $Password }
    if ($Thumbprint) { $signArgs.Thumbprint = $Thumbprint }
    if ($SelfSigned) { $signArgs.SelfSigned = $true }
    & "$PSScriptRoot\sign.ps1" @signArgs
} else {
    Write-Host "[build] 未签名（未提供证书）。如需签名：-Pfx <证书> / -Thumbprint <指纹> / -SelfSigned" -ForegroundColor Yellow
}

# ---------------- 4. 校验值与产物汇总 ----------------
$sumPath = "dist\SHA256SUMS.txt"
$lines = @()
$setupExe = Get-ChildItem "dist\installer\*.exe" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($setupExe) {
    $h = (Get-FileHash $setupExe.FullName -Algorithm SHA256).Hash.ToLower()
    $lines += "$h  $($setupExe.Name)"
}
$exe = Get-Item "dist\TTComet\TTComet.exe" -ErrorAction SilentlyContinue
if ($exe) {
    $h = (Get-FileHash $exe.FullName -Algorithm SHA256).Hash.ToLower()
    $lines += "$h  TTComet.exe"
}
if ($lines.Count -gt 0) {
    $lines | Set-Content -Path $sumPath -Encoding ascii
    Write-Host "[build] 已写入校验值: $sumPath" -ForegroundColor Green
}

Write-Host ""
if ($setupExe) {
    Write-Host ("[build] 安装包: {0} ({1:N1} MB)" -f $setupExe.FullName, ($setupExe.Length / 1MB)) -ForegroundColor Green
}
if ($exe) {
    $sig = Get-AuthenticodeSignature $exe.FullName
    Write-Host ("[build] exe: {0} | 签名状态: {1}" -f $exe.FullName, $sig.Status) -ForegroundColor $(if ($sig.Status -eq 'Valid') { 'Green' } else { 'Yellow' })
}
