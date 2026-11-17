; ============================================================
;  乒乓彗星 TTComet 安装包（Inno Setup 6）
; ============================================================
;  编译：
;      "C:\Users\<你>\AppData\Local\Programs\Inno Setup 6\ISCC.exe" packaging\app.iss
;  或：
;      .\packaging\build.ps1 -WithFFmpeg -Installer
;
;  产物：dist\installer\TTComet-0.0.2-setup.exe
; ============================================================

#define MyAppName "乒乓彗星"
#define MyAppVersion "0.1.1"
#define MyAppPublisher "Thegongyx"
#define MyAppExeName "TTComet.exe"
#define SrcDir "..\dist\TTComet"

[Setup]
AppId={{C41B7E52-9A3D-4F68-B0E7-2D5A8C41F390}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}

; 按用户安装，不需要管理员权限（避免 UAC 劝退）
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\Programs\TTComet
DisableProgramGroupPage=yes
AllowNoIcons=yes

; 输出
OutputDir=..\dist\installer
OutputBaseFilename=TTComet-{#MyAppVersion}-setup
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes

; 64 位
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; 安装/卸载时若程序在运行，提示关闭
CloseApplications=yes
CloseApplicationsFilter=*.exe,*.dll
RestartApplications=no

UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}
SetupIconFile=TTComet.ico

; 中文界面
[Languages]
Name: "chinese"; MessagesFile: "ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务："; Flags: unchecked

[Files]
; 整个 onedir 目录
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; 使用说明放在 exe 同级（构建脚本也会拷一份，这里再保险一次）
Source: "使用说明.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autoprograms}\{#MyAppName} 使用说明"; Filename: "{app}\使用说明.txt"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 只删程序目录；用户数据在 %LOCALAPPDATA%\TTComet\jobs，保留
Type: filesandordirs; Name: "{app}"
