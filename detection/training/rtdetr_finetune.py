import os
import yaml
from ultralytics import RTDETR

dataset_dir = (
    "/home/Joshua/Downloads/leopard_toad_identification/dataset/detect_2/dataset_clahe"
)
data_yaml_path = os.path.join(dataset_dir, "dataset_clahe.yaml")

classes = ["Other_Amphibian", "Small_Mammal", "Western_Leopard_Toad"]

# Generate data.yaml if it doesn't exist
if not os.path.exists(data_yaml_path):
    yaml_content = {
        "path": dataset_dir,
        "train": "images",
        "val": "images",
        "nc": len(classes),
        "names": {i: name for i, name in enumerate(classes)},
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Successfully generated {data_yaml_path}")

# Load the best RT-DETR model from previous run
model_path = "/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/runs/detect/rtdetr_finetuning/weights/best.pt"
print(f"Loading RT-DETR model from {model_path}...")
model = RTDETR(model_path)


def main():
    print("Starting partial fine-tuning (freezing backbone)...")
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=32,
        device="0",
        project="/home/Joshua/Downloads/leopard_toad_identification/detection/runs/rtdetr_finetune_15",
        name="subset_finetune",
        exist_ok=True,
        freeze=15,  # Freeze the backbone layers for partial fine-tuning
        # Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )
    print("Fine-tuning complete! Saved to runs/rtdetr_finetune/subset_finetune")


if __name__ == "__main__":
    main()
