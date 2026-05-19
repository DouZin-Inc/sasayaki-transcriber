@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Sasayaki Setup
echo ============================================
echo.

set "BASE_DIR=%~dp0"
cd /d "%BASE_DIR%"

:: ============================================
:: 1. Python
:: ============================================
set "PYTHON_DIR=%BASE_DIR%python"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"

if exist "%PYTHON_EXE%" (
    echo [OK] Python already installed
    goto :check_ffmpeg
)

echo [1/4] Downloading Python...
set "PYTHON_VERSION=3.11.9"
set "PYTHON_ZIP=python-%PYTHON_VERSION%-embed-amd64.zip"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_ZIP%"

powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ZIP%'" 2>nul
if errorlevel 1 (
    echo [Error] Failed to download Python.
    pause
    exit /b 1
)

echo Extracting Python...
mkdir "%PYTHON_DIR%" 2>nul
powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
del "%PYTHON_ZIP%"

for %%f in ("%PYTHON_DIR%\python*._pth") do (
    powershell -Command "(Get-Content '%%f') -replace '#import site','import site' | Set-Content '%%f'"
    echo ..>>"%%f"
)

echo Installing pip...
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'"
"%PYTHON_EXE%" get-pip.py
del get-pip.py

echo [OK] Python %PYTHON_VERSION% installed

:: ============================================
:: 2. ffmpeg
:: ============================================
:check_ffmpeg
set "FFMPEG_DIR=%BASE_DIR%ffmpeg"
set "FFMPEG_EXE=%FFMPEG_DIR%\ffmpeg.exe"

if exist "%FFMPEG_EXE%" (
    echo [OK] ffmpeg already installed
    goto :install_deps
)

where ffmpeg >nul 2>&1
if not errorlevel 1 (
    echo [OK] Using system ffmpeg
    goto :install_deps
)

echo [2/4] Downloading ffmpeg...
set "FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
set "FFMPEG_ZIP=ffmpeg.zip"

powershell -Command "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%'" 2>nul
if errorlevel 1 (
    echo [Error] Failed to download ffmpeg.
    pause
    exit /b 1
)

echo Extracting ffmpeg...
powershell -Command "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%TEMP%\ffmpeg_extract' -Force"
mkdir "%FFMPEG_DIR%" 2>nul

for /d %%d in ("%TEMP%\ffmpeg_extract\ffmpeg-*") do (
    copy "%%d\bin\ffmpeg.exe" "%FFMPEG_DIR%\" >nul
    copy "%%d\bin\ffprobe.exe" "%FFMPEG_DIR%\" >nul
)

rd /s /q "%TEMP%\ffmpeg_extract" 2>nul
del "%FFMPEG_ZIP%"
echo [OK] ffmpeg installed

:: ============================================
:: 3. PyTorch
:: ============================================
:install_deps
set "PIP_EXE=%PYTHON_DIR%\Scripts\pip.exe"

echo [3/4] Installing dependencies...
"%PIP_EXE%" install -r "%BASE_DIR%requirements.txt"
if errorlevel 1 (
    echo [Error] Failed to install dependencies.
    pause
    exit /b 1
)

:: ============================================
:: 4. PyTorch CUDA (依存パッケージの後に上書き)
:: ============================================
echo [4/4] Installing PyTorch (CUDA)...
"%PIP_EXE%" install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps

echo.
echo ============================================
echo   Setup complete
echo ============================================
