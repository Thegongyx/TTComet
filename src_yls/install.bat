@echo off
echo Installing YOLO ML Backend dependencies...
call conda run -n yolo26 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r "%~dp0requirements.txt"
if %ERRORLEVEL% neq 0 (
    echo Installation FAILED.
    pause
    exit /b 1
)
echo Installation complete.
pause
