@echo off

set "BASE_DIR=%~dp0"
cd /d "%BASE_DIR%"

if not exist "%BASE_DIR%python\python.exe" (
    echo [Error] Python not found. Please run setup.bat first.
    pause
    exit /b 1
)

if exist "%BASE_DIR%ffmpeg\ffmpeg.exe" (
    set "PATH=%BASE_DIR%ffmpeg;%PATH%"
)

:: Ensure app directory is in Python module search path
for %%f in ("%BASE_DIR%python\python*._pth") do (
    findstr /x ".." "%%f" >nul 2>&1 || echo ..>>"%%f"
)

for /f "tokens=3 delims= " %%v in ('findstr /r "__version__" "%BASE_DIR%version.py"') do set "APP_VERSION=%%~v"

:: Try pythonw.exe (no console window) first, fallback to python.exe
if exist "%BASE_DIR%python\pythonw.exe" (
    start "" "%BASE_DIR%python\pythonw.exe" "%BASE_DIR%app.py"
    exit /b
)

:: Fallback: python.exe with console (minimized)
if not "%1"=="--minimized" (
    start /min "" cmd /k "%~f0" --minimized
    exit /b
)

echo Starting Sasayaki v%APP_VERSION% ...
"%BASE_DIR%python\python.exe" -u "%BASE_DIR%app.py" 2> "%BASE_DIR%crash.log"

if errorlevel 1 (
    echo.
    echo ============================================
    echo [Error] Sasayaki terminated unexpectedly
    echo ============================================
    echo.
    type "%BASE_DIR%crash.log"
    echo.
    echo Log file: %BASE_DIR%crash.log
    pause
)
