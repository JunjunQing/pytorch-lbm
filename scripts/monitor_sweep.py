#!/usr/bin/env python3
"""Monitor the background sweep progress."""
import os
import time
from datetime import datetime

PROJECT_ROOT = '/home/qmingjun/projects/pytorch_lbm-main'
RESULTS_FILE = os.path.join(PROJECT_ROOT, 'results', 'thesis_sweep_results.json')
LOG_FILE = os.path.join(PROJECT_ROOT, 'results', 'sweep_background.log')
PENDING_LOG = os.path.join(PROJECT_ROOT, 'results', 'pending_sweep_log.txt')


def check_status():
    """Check current sweep status."""
    print("=" * 60)
    print(f"  SWEEP STATUS @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check if process is running
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    python_procs = [l for l in result.stdout.split('\n') if 'python3 scripts/run_pending' in l]
    if python_procs:
        print("  Status: RUNNING")
        for p in python_procs:
            print(f"  {p}")
    else:
        print("  Status: NOT RUNNING")

    # Check results
    if os.path.exists(RESULTS_FILE):
        import json
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        print(f"\n  Completed cases: {len(data)}")
        
        # Count by study
        studies = {}
        for d in data:
            label = d['label']
            if label.startswith('s3a'):
                studies['3a'] = studies.get('3a', 0) + 1
            elif label.startswith('s3b'):
                studies['3b'] = studies.get('3b', 0) + 1
            elif label.startswith('s5'):
                studies['5'] = studies.get('5', 0) + 1
            else:
                studies['other'] = studies.get('other', 0) + 1
        
        print(f"  Study 3a: {studies.get('3a', 0)} cases")
        print(f"  Study 3b: {studies.get('3b', 0)} cases")
        print(f"  Study 5: {studies.get('5', 0)} cases")
        print(f"  Other (0,1,2,7): {studies.get('other', 0)} cases")
    else:
        print("  Results file not found")

    # Check logs
    if os.path.exists(PENDING_LOG):
        print(f"\n  Recent log entries:")
        with open(PENDING_LOG) as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(f"    {line.rstrip()}")
    
    print("=" * 60)


if __name__ == '__main__':
    check_status()
