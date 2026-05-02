@echo off
rem Upload GuiXi to PyPI (Windows)

echo === GuiXi PyPI Upload Script ===
echo.

rem Check for required tools
where python >nul 2>&1 || goto :error
where twine >nul 2>&1 || goto :no_twine
where build >nul 2>&1 || goto :no_build

rem Get version from pyproject.toml
for /f "tokens=2 delims===" %%i in ('findstr /C:"version" pyproject.toml') do set VERSION=%%i
set VERSION=%VERSION:"=%
set VERSION=%VERSION: =%
echo Building version: %VERSION%
echo.

rem Clean previous builds
echo Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist *.egg-info rmdir /s /q *.egg-info
echo.

rem Build the package
echo Building package...
python -m build
echo.

rem Check the package
echo Checking package...
python -m twine check dist/*
echo.

rem Upload to PyPI
echo Uploading to PyPI...
python -m twine upload dist/*

echo.
echo === Upload Complete ===
echo Package should be available at: https://pypi.org/project/guixi/%VERSION%/
goto :end

:error
echo Python is required but not installed.
exit /b 1

:no_twine
echo Twine is not installed. Run: pip install twine
exit /b 1

:no_build
echo Build is not installed. Run: pip install build
exit /b 1

:end
pause
