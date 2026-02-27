import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_script(path: Path):
    print(f"\n=== Running {path} ===")
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("[stderr]")
        print(result.stderr)
    if result.returncode != 0:
        print(f"Script {path.name} failed with code {result.returncode}")
    else:
        print(f"Completed {path.name}\n")


def main():
    scripts = [
        ROOT / 'rural_hadza' / 'xgb_ohe_rural_hadza.py',
        ROOT / 'gm' / 'xgb_ohe_gm.py',
        ROOT / 'age_18_24' / 'xgb_ohe_age_18_24.py',
    ]
    for script in scripts:
        run_script(script)


if __name__ == '__main__':
    main()
