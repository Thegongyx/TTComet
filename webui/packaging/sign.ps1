<#
  TTComet 代码签名
  =================
  三种用法：

  1) 正式证书（PFX 文件，最常见）
       .\packaging\sign.ps1 -Target dist\TTComet -Pfx C:\secure\mycert.pfx -Password 123456

  2) 证书已安装在个人证书库（有指纹）
       .\packaging\sign.ps1 -Target dist\TTComet -Thumbprint ABCDEF...

  3) 自签名证书（内部测试用；外部用户仍会看到"未知发布者"）
       .\packaging\sign.ps1 -Target dist\TTComet -SelfSigned

  说明：
    - 只签 exe 文件（PyInstaller 的 TTComet.exe + Inno 的安装包）
    - 用 SHA256 + RFC3161 时间戳（证书过期后已签文件仍然有效）
    - 签完会逐个 verify，失败会明确报错
#>
param(
    [Parameter(Mandatory = $true)][string]$Target,   # 目录或单个 exe
    [string]$Pfx = "",
    [string]$Password = "",
    [string]$Thumbprint = "",
    [switch]$SelfSigned,
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [string]$SelfSignedSubject = "CN=TTComet Dev (self-signed), O=TTComet"
)

$ErrorActionPreference = "Stop"

# ---------------- 找 signtool ----------------
function Find-SignTool {
    $cands = @()
    $kits = "C:\Program Files (x86)\Windows Kits\10\bin"
    if (Test-Path $kits) {
        $cands += Get-ChildItem $kits -Directory |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "x64\signtool.exe" } |
            Where-Object { Test-Path $_ }
    }
    $cands += @(
        "C:\Program Files (x86)\Windows Kits\8.1\bin\x64\signtool.exe",
        "C:\Program Files\Windows Kits\10\bin\x64\signtool.exe"
    )
    $found = $cands | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $found) {
        $cmd = (Get-Command signtool.exe -ErrorAction SilentlyContinue)
        if ($cmd) { $found = $cmd.Source }
    }
    return $found
}

$signtool = Find-SignTool
if (-not $signtool) {
    throw "找不到 signtool.exe。请安装 Windows SDK（含 Signing Tools），或把它加入 PATH。"
}
Write-Host "[sign] signtool: $signtool" -ForegroundColor Cyan

# ---------------- 解析目标 ----------------
$files = @()
if (Test-Path $Target -PathType Container) {
    $files += Get-ChildItem $Target -Filter *.exe -File
    # 安装包在 dist\installer 下也一并签
    $inst = Join-Path (Split-Path $Target -Parent) "installer"
    if (Test-Path $inst) { $files += Get-ChildItem $inst -Filter *.exe -File }
} else {
    $files += Get-Item $Target
}
$files = $files | Sort-Object FullName -Unique
if (-not $files) { throw "没有找到可签名文件: $Target" }

# ---------------- 准备证书参数 ----------------
$certArgs = @()
$tempPfx = $null

if ($SelfSigned) {
    Write-Host "[sign] 生成自签名代码签名证书（有效期 3 年）..." -ForegroundColor Yellow
    $cert = New-SelfSignedCertificate `
        -Type CodeSigningCert `
        -Subject $SelfSignedSubject `
        -CertStoreLocation "Cert:\CurrentUser\My" `
        -KeyUsage DigitalSignature `
        -KeyAlgorithm RSA -KeyLength 3072 `
        -HashAlgorithm SHA256 `
        -NotAfter (Get-Date).AddYears(3)
    $certArgs = @("/sha1", $cert.Thumbprint)

    # 导出公钥证书，内部用户导入后才不会看到警告
    $certDir = Join-Path $PSScriptRoot "certs"
    New-Item -ItemType Directory -Force -Path $certDir | Out-Null
    $cerPath = Join-Path $certDir "TTComet-selfsigned.cer"
    Export-Certificate -Cert $cert -FilePath $cerPath -Type CERT | Out-Null
    Write-Host "[sign] 自签名证书: $($cert.Thumbprint)" -ForegroundColor Yellow
    Write-Host "[sign] 公钥已导出: $cerPath （内部用户双击安装到「受信任的根证书颁发机构」后不再提示）" -ForegroundColor Yellow
}
elseif ($Thumbprint) {
    $certArgs = @("/sha1", $Thumbprint)
    Write-Host "[sign] 使用证书库中的证书: $Thumbprint" -ForegroundColor Cyan
}
elseif ($Pfx) {
    if (-not (Test-Path $Pfx)) { throw "找不到 PFX 文件: $Pfx" }
    $certArgs = @("/f", (Resolve-Path $Pfx).Path)
    if ($Password) { $certArgs += @("/p", $Password) }
    Write-Host "[sign] 使用 PFX: $Pfx" -ForegroundColor Cyan
}
else {
    throw "必须指定 -Pfx / -Thumbprint / -SelfSigned 之一"
}

# ---------------- 逐个签名 ----------------
$ok = 0
foreach ($f in $files) {
    Write-Host "[sign] 签名 $($f.Name) ..." -ForegroundColor Cyan
    $args = @(
        "sign", "/fd", "SHA256", "/td", "SHA256", "/tr", $TimestampUrl,
        "/d", "TTComet - Ping-pong highlight editor",
        "/du", "https://github.com/Thegongyx/tt_ylov"
    ) + $certArgs + @($f.FullName)

    & $signtool @args
    if ($LASTEXITCODE -ne 0) {
        # 时间戳服务器偶发不可用时，退回不带时间戳再签一次
        Write-Host "[sign] 带时间戳失败，重试不带时间戳..." -ForegroundColor Yellow
        $args2 = @("sign", "/fd", "SHA256") + $certArgs + @($f.FullName)
        & $signtool @args2
        if ($LASTEXITCODE -ne 0) { throw "签名失败: $($f.Name)" }
    }

    # 校验
    #   - 正式证书：/pa 走系统信任链，失败即报错
    #   - 自签名：/pa 必然失败（证书不在受信任根里），只校验"确实带签名"
    if ($SelfSigned) {
        $sig = Get-AuthenticodeSignature $f.FullName
        if (-not $sig.SignerCertificate) { throw "签名写入失败: $($f.Name)" }
        Write-Host "[sign] ✅ $($f.Name) 已签名（自签名，外部用户仍会提示未知发布者）" -ForegroundColor Yellow
    } else {
        & $signtool verify /pa /v $f.FullName | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "签名校验失败: $($f.Name)" }
        Write-Host "[sign] ✅ $($f.Name) 已签名并通过信任链校验" -ForegroundColor Green
    }
    $ok++
}

Write-Host "[sign] 完成：$ok 个文件" -ForegroundColor Green
