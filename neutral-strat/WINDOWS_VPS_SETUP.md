# Windows VPS Setup Guide for Sigmatic Trading Bot

This guide helps you deploy the Sigmatic neutral pairs trading bot on a Windows Server 2022 VPS.

## Prerequisites

- Windows Server 2022 VPS with RDP access
- VPS IP address and administrator credentials
- Your trading bot code (this repository)

## Step 1: Connect to Your Windows VPS

1. Open **Remote Desktop Connection** on your local machine
2. Enter your VPS IP address: `188.119.102.184` (or your VPS IP)
3. Use your administrator username and password
4. Connect via RDP

## Step 2: Install Python 3.11

1. Open **PowerShell as Administrator**
2. Download Python 3.11:
   ```powershell
   Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe" -OutFile "python-3.11.7.exe"
   ```
3. Install Python (silent install with PATH):
   ```powershell
   .\python-3.11.7.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
   ```
4. Restart PowerShell and verify installation:
   ```powershell
   python --version
   pip --version
   ```

## Step 3: Download Your Trading Bot Code

### Option A: Using Git (Recommended)
1. Install Git for Windows:
   ```powershell
   winget install --id Git.Git -e --source winget
   ```
2. Clone your repository:
   ```powershell
   cd C:\
   git clone https://github.com/yourusername/sigmatic.git
   cd sigmatic\neutral-strat
   ```

### Option B: Direct Download
1. Download the ZIP file from your GitHub repository
2. Extract to `C:\sigmatic\neutral-strat\`
3. Navigate to the folder in PowerShell:
   ```powershell
   cd C:\sigmatic\neutral-strat
   ```

## Step 4: Install Python Dependencies

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install requirements:
   ```powershell
   pip install -r requirements.txt
   ```

## Step 5: Configure Your Trading Bot

1. Create your configuration file:
   ```powershell
   copy config\two_week_config.yaml config\windows_live_config.yaml
   ```

2. Edit the config file (use Notepad or your preferred editor):
   ```powershell
   notepad config\windows_live_config.yaml
   ```

3. Create environment file for API keys:
   ```powershell
   New-Item -Path ".env" -ItemType File
   notepad .env
   ```

4. Add your API credentials to `.env`:
   ```
   BINANCE_API_KEY=your_api_key_here
   BINANCE_SECRET_KEY=your_secret_key_here
   ```

## Step 6: Test Your Setup

1. Run a quick test:
   ```powershell
   python -c "import pandas, numpy, yaml; print('Dependencies OK')"
   ```

2. Test the trading bot (dry run):
   ```powershell
   python scripts\live_trader.py --config config\windows_live_config.yaml --dry-run
   ```

## Step 7: Create Windows Service (Optional)

To run your bot as a Windows service that starts automatically:

1. Install `pywin32`:
   ```powershell
   pip install pywin32
   ```

2. Create service script `sigmatic_service.py`:
   ```python
   import win32serviceutil
   import win32service
   import win32event
   import subprocess
   import os

   class SigmaticService(win32serviceutil.ServiceFramework):
       _svc_name_ = "SigmaticTrader"
       _svc_display_name_ = "Sigmatic Trading Bot"

       def __init__(self, args):
           win32serviceutil.ServiceFramework.__init__(self, args)
           self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)

       def SvcStop(self):
           self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
           win32event.SetEvent(self.hWaitStop)

       def SvcDoRun(self):
           os.chdir(r'C:\sigmatic\neutral-strat')
           subprocess.call([r'C:\sigmatic\neutral-strat\venv\Scripts\python.exe',
                           r'C:\sigmatic\neutral-strat\scripts\live_trader.py'])

   if __name__ == '__main__':
       win32serviceutil.HandleCommandLine(SigmaticService)
   ```

3. Install the service:
   ```powershell
   python sigmatic_service.py install
   python sigmatic_service.py start
   ```

## Step 8: Create Monitoring Scripts

### System Monitor (PowerShell)
Create `monitor.ps1`:
```powershell
# System monitoring script
$logFile = "C:\sigmatic\neutral-strat\logs\system_monitor.log"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Check disk space
$disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'"
$freeSpacePercent = [math]::Round(($disk.FreeSpace / $disk.Size) * 100, 2)

if ($freeSpacePercent -lt 20) {
    Add-Content $logFile "$timestamp - WARNING: Low disk space: $freeSpacePercent% free"
}

# Check if Python process is running
$pythonProcess = Get-Process -Name "python" -ErrorAction SilentlyContinue
if (-not $pythonProcess) {
    Add-Content $logFile "$timestamp - WARNING: Python trading bot not running"
}

Add-Content $logFile "$timestamp - System check completed"
```

### Schedule monitoring (Task Scheduler)
```powershell
# Create scheduled task for monitoring
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File C:\sigmatic\neutral-strat\monitor.ps1"
$trigger = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Minutes 5) -Once -At (Get-Date)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnDemand

Register-ScheduledTask -TaskName "SigmaticMonitor" -Action $action -Trigger $trigger -Principal $principal -Settings $settings
```

## Step 9: Start Your Trading Bot

### Manual Start
```powershell
cd C:\sigmatic\neutral-strat
.\venv\Scripts\Activate.ps1
python scripts\live_trader.py
```

### Background Start (using nssm - Non-Sucking Service Manager)
1. Download NSSM: https://nssm.cc/download
2. Install as service:
   ```powershell
   nssm install SigmaticBot C:\sigmatic\neutral-strat\venv\Scripts\python.exe
   nssm set SigmaticBot Arguments "C:\sigmatic\neutral-strat\scripts\live_trader.py"
   nssm set SigmaticBot AppDirectory "C:\sigmatic\neutral-strat"
   nssm start SigmaticBot
   ```

## Step 10: Monitor Your Bot

### Check logs:
```powershell
Get-Content C:\sigmatic\neutral-strat\logs\trading.log -Tail 20 -Wait
```

### Check service status:
```powershell
Get-Service SigmaticBot
```

### Performance monitoring:
- Open **Task Manager** → **Performance** tab
- Monitor CPU, Memory, and Network usage

## Security Considerations

1. **Firewall**: Enable Windows Defender Firewall
2. **Updates**: Keep Windows updated
3. **API Keys**: Never commit API keys to version control
4. **Access**: Limit RDP access to specific IP addresses
5. **Backup**: Regularly backup your configuration and logs

## Troubleshooting

### Common Issues:
1. **Python not found**: Restart PowerShell after Python installation
2. **Module not found**: Ensure virtual environment is activated
3. **API errors**: Check your `.env` file and API key permissions
4. **Service won't start**: Check Windows Event Viewer for error details

### Log Locations:
- Trading logs: `C:\sigmatic\neutral-strat\logs\`
- Windows Event Logs: Event Viewer → Windows Logs → Application
- Service logs: Event Viewer → Applications and Services Logs

## Next Steps

1. Monitor your bot's performance for the first few hours
2. Set up alerts for critical errors
3. Create regular backups of your configuration
4. Monitor your Binance account for trades

Your Sigmatic trading bot is now ready to run on Windows VPS!