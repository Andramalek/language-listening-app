@echo off
echo ===== Windows Camera Bridge - Firewall Fix =====
echo.
echo This script will add firewall rules to allow connections to the Camera Bridge.
echo.

:: Check for administrative privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script requires administrative privileges.
    echo Please right-click on this file and select "Run as administrator".
    pause
    exit /b 1
)

echo Creating firewall rules...

:: Add rule for port 8080
echo Adding rule for port 8080...
netsh advfirewall firewall add rule name="Windows Camera Bridge 8080" dir=in action=allow protocol=TCP localport=8080

:: Allow Python through the firewall
echo Adding rule for Python...
where python >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('where python') do (
        echo Adding rule for: %%i
        netsh advfirewall firewall add rule name="Python - Camera Bridge" dir=in action=allow program="%%i" enable=yes
    )
)

echo.
echo Firewall rules have been added.
echo Please try connecting to the Camera Bridge again.
echo.
echo Remember to:
echo 1. Make sure the Camera Bridge is running
echo 2. Try both IP addresses: 192.168.0.32 and 172.18.16.1
echo.

pause 