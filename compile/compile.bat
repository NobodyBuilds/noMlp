@echo off
rem Builds the noMlp static library from C++ and CUDA source files.
setlocal EnableExtensions EnableDelayedExpansion

rem Resolve project paths relative to this script.
set "ROOT=%~dp0.."
set "SRC_DIR=%ROOT%\src"
set "INCLUDE_DIR=%ROOT%\include"
set "LIB_DIR=%ROOT%\lib"
set "OUTPUT=%LIB_DIR%\noMlp.lib"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "SOURCES="

rem Initialize the MSVC x64 environment required by nvcc.
if not exist "%VCVARS%" (
    echo Visual Studio 2022 x64 build tools were not found.
    exit /b 1
)
call "%VCVARS%"
if errorlevel 1 (
    echo Failed to initialize the Visual Studio 2022 x64 compiler environment.
    exit /b 1
)

rem Collect only source files that exist.
for %%F in ("%SRC_DIR%\*.cpp" "%SRC_DIR%\*.cu") do (
    if exist "%%~fF" set "SOURCES=!SOURCES! "%%~fF""
)

rem Create the output directory and archive the sources with nvcc.
if not exist "%LIB_DIR%" mkdir "%LIB_DIR%"
if not exist "%LIB_DIR%" (
    echo Could not create "%LIB_DIR%".
    exit /b 1
)
if not defined SOURCES (
    echo No C++ or CUDA source files were found in "%SRC_DIR%".
    exit /b 1
)
nvcc -std=c++17 -rdc=true -Xcompiler "/std:c++17 /MD" -lib -I"%INCLUDE_DIR%" !SOURCES! -o "%OUTPUT%"
if errorlevel 1 (
    echo Compilation failed.
    exit /b 1
)

echo Built "%OUTPUT%"
exit /b 0
