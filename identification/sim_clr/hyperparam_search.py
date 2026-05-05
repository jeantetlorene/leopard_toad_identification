import os
import subprocess
import re
import sys
from itertools import product
from config import Config


def run_search():
    # Define search space
    # Temperature and Initial Learning Rate are the primary variables
    temps = [0.12, 0.5]
    lrs = [3e-4, 1e-4]

    # Search specific settings
    search_epochs = 20
    batch_size = 16

    data_dir = Config.DATA_DIR
    pretrained_backbone = Config.PRETRAINED_BACKBONE

    results = []

    for temp, lr in product(temps, lrs):
        print(f"\n{'=' * 60}")
        print(f" TESTING: Temperature={temp} | Initial LR={lr} ")
        print(f"{'=' * 60}")

        run_name = f"t{temp}_lr{lr}"
        weights_dir = os.path.join(
            Config.BASE_DATA_DIR, f"sim_clr/weights/search_{run_name}"
        )
        logs_dir = os.path.join(Config.BASE_DATA_DIR, f"sim_clr/logs/search_{run_name}")

        # 1. Train
        train_cmd = [
            sys.executable,
            "train.py",
            "--data_dir",
            data_dir,
            "--epochs",
            str(search_epochs),
            "--batch_size",
            str(batch_size),
            "--learning_rate",
            str(lr),
            "--temperature",
            str(temp),
            "--weights_dir",
            weights_dir,
            "--logs_dir",
            logs_dir,
            "--pretrained_backbone",
            pretrained_backbone,
        ]

        print(f"Running training: {' '.join(train_cmd)}")
        subprocess.run(train_cmd)

        # 2. Evaluate
        weights_path = os.path.join(weights_dir, "resnet50_backbone_final.pth")
        eval_cmd = [
            sys.executable,
            "evaluate_reid.py",
            "--data_dir",
            data_dir,
            "--weights_path",
            weights_path,
        ]

        print(f"Running evaluation: {' '.join(eval_cmd)}")
        try:
            eval_output = subprocess.check_output(
                eval_cmd, stderr=subprocess.STDOUT
            ).decode()
            print(eval_output)

            # Parse Top-1 Accuracy
            match = re.search(r"Top-1 Accuracy:\s+([\d.]+)", eval_output)
            top1 = float(match.group(1)) if match else 0.0

            results.append({"temp": temp, "lr": lr, "top1": top1})
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed for {run_name}: {e.output.decode()}")
            results.append({"temp": temp, "lr": lr, "top1": 0.0})

    # Summary Table
    print("\n" + "#" * 45)
    print("      SIMCLR HYPERPARAMETER SEARCH SUMMARY      ")
    print("#" * 45)
    print(f"{'Temperature':<12} | {'Initial LR':<12} | {'Top-1 Acc':<10}")
    print("-" * 43)
    for res in sorted(results, key=lambda x: x["top1"], reverse=True):
        print(f"{res['temp']:<12} | {res['lr']:<12} | {res['top1']:<10.4f}")
    print("#" * 45 + "\n")


if __name__ == "__main__":
    run_search()
