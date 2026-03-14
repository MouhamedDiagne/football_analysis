"""
find_lock.py — trouve quel processus tient yolov9c.pt verrouillé
"""
import subprocess, sys

# Méthode 1 : WMI via Python
try:
    import wmi
    c = wmi.WMI()
    for p in c.Win32_Process():
        if 'python' in p.Name.lower():
            print(f"PID={p.ProcessId:6}  Name={p.Name:20}  CMD={str(p.CommandLine)[:80]}")
except ImportError:
    print("wmi non disponible, utilisation de PowerShell")

# Méthode 2 : PowerShell via subprocess
ps_script = """
$procs = Get-Process -Name python,python3.13 -ErrorAction SilentlyContinue
foreach ($p in $procs) {
    Write-Output ("PID=" + $p.Id + " Name=" + $p.Name)
}
"""
r = subprocess.run(
    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_script],
    capture_output=True, text=True, timeout=15
)
print("\nProcessus Python actifs :")
print(r.stdout)
if r.stderr:
    print("ERR:", r.stderr[:200])
