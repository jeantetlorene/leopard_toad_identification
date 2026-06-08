import time
import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from ultralytics import YOLO

def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
yolo_path = "/home/Joshua/Downloads/leopard_toad_identification/detection/runs/yolo_finetune/subset_finetune/weights/best.pt"
rtdetr_path = "/home/Joshua/Downloads/leopard_toad_identification/detection/runs/rtdetr_finetune/subset_finetune/weights/best.pt"

models = {
    "yolo": YOLO(yolo_path),
    "rtdetr": YOLO(rtdetr_path),
    "faster_rcnn": fasterrcnn_resnet50_fpn(weights=None, num_classes=4).to(device)
}
models["faster_rcnn"].eval()

dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

def count_params(m_name, m):
    if m_name in ["yolo", "rtdetr"]:
        return sum(p.numel() for p in m.model.parameters())
    return sum(p.numel() for p in m.parameters())

def time_inference(m_name, m, do_clahe):
    for _ in range(3):
        img = dummy_img.copy()
        if do_clahe: img = apply_clahe(img)
        if m_name in ["yolo", "rtdetr"]:
            m.predict(img, verbose=False, device=device)
        else:
            with torch.no_grad():
                m([TF.to_tensor(img).to(device)])
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        img = dummy_img.copy()
        if do_clahe: img = apply_clahe(img)
        if m_name in ["yolo", "rtdetr"]:
            m.predict(img, verbose=False, device=device)
        else:
            with torch.no_grad():
                m([TF.to_tensor(img).to(device)])
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.time() - start) / 10

for name, m in models.items():
    p = count_params(name, m)
    t_plain = time_inference(name, m, False) * 1000
    t_clahe = time_inference(name, m, True) * 1000
    print(f"{name} | Params: {p/1e6:.2f}M | Plain Time: {t_plain:.2f}ms | CLAHE Time: {t_clahe:.2f}ms")

