"""Run the whole D2 pipeline:  python run_all.py"""
import subprocess
import sys

for script in ("step1_screen_baseline.py", "step2_train_zero_shot.py", "step3_reconstruct_and_evaluate.py"):
    print(f"\n=========== {script} ===========")
    if subprocess.call([sys.executable, script]) != 0:
        sys.exit(f"{script} failed - see the message above")
