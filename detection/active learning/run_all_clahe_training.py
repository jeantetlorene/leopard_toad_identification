import os
import subprocess
import sys


def run_script(script_path):
    print(f"\n=======================================================")
    print(f"Executing: {script_path}")
    print(f"=======================================================\n")

    # Ensure we use the correct python interpreter from the active environment
    python_exe = sys.executable

    # Change working directory to the directory of the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

    result = subprocess.run([python_exe, script_path, "--mode", "both"], cwd=script_dir)
    if result.returncode != 0:
        print(f"Error executing {script_path}. Exiting.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    yolo_script = os.path.join(base_dir, "yolo_clahe", "run_yolo_clahe.py")
    rtdetr_script = os.path.join(base_dir, "rtdetr_clahe", "run_rtdetr_clahe.py")
    faster_rcnn_script = os.path.join(
        base_dir, "faster_rcnn_clahe", "run_faster_rcnn_clahe.py"
    )

    scripts_to_run = [yolo_script, rtdetr_script, faster_rcnn_script]

    for script in scripts_to_run:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Warning: Script not found: {script}")

    print("\nAll CLAHE training pipelines have completed.")
