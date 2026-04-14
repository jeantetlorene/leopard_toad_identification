# %%
from ultralytics import YOLO

model = YOLO("/home/Joshua/Downloads/OIDv4_ToolKit/runs/detect/yolo_model/weights/best.pt")

# Prints the detailed layers table
model.info(detailed=True)

# %%
# Prints the exact PyTorch structure (nn.Sequential, exact channel sizes, etc.)
print(model.model)

# %%
