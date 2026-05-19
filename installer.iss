; Sasayaki Inno Setup Script
; Inno Setup 6.x 以降が必要です

#define MyAppName "Sasayaki"
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#define MyAppPublisher "DouZin Inc."
#define MyAppExeName "start.bat"
#define MyAppIcon "sasayaki.ico"

[Setup]
AppId={{B8A3D2E1-4F5C-6789-ABCD-EF0123456789}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=SasayakiSetup ({#MyAppVersion})
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
SetupIconFile={#MyAppIcon}
UninstallDisplayIcon={app}\{#MyAppIcon}

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Files]
Source: "version.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "app.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "transcriber.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "diarize_worker.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "whisperx_worker.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "formatters.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "start.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "sasayaki.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIcon}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIcon}"

[Run]
Filename: "{app}\setup.bat"; Description: "セットアップを実行（Python/ffmpeg/依存パッケージのインストール）"; Flags: runascurrentuser waituntilterminated postinstall
Filename: "{app}\start.bat"; Description: "Sasayaki を起動する"; Flags: runascurrentuser nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\python\Lib"
Type: filesandordirs; Name: "{app}\python\Scripts"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\ffmpeg"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\history"
Type: filesandordirs; Name: "{app}\models"
