import os
import yaml
from ultralytics import YOLO

dataset_dir = "/home/Joshua/Downloads/leopard_toad_identification/dataset/dataset"
classes_txt_path = os.path.join(dataset_dir, "classes.txt")
data_yaml_path = os.path.join(dataset_dir, "data.yaml")

# 1. Read the classes from classes.txt
with open(classes_txt_path, "r") as f:
    classes = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loaded {len(classes)} classes: {classes}")

# 2. Automatically generate the data.yaml file
yaml_content = {
    "path": dataset_dir,
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(classes),
    "names": {i: name for i, name in enumerate(classes)},
}

with open(data_yaml_path, "w") as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"Successfully generated {data_yaml_path}")

# 3. Initialize the YOLO model
model = YOLO("yolo26m.pt")


# 4. Train the model
def main():
    print("Starting training...")
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        batch=32,
        device="0",
        project="/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/runs/train",
        name="yolo_model",
        exist_ok=True,
    )
    print("Training complete!")


if __name__ == "__main__":
    main()
