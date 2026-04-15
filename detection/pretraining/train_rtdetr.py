import torch
from ultralytics import RTDETR

# Ultralytics RT-DETR: Real-Time DEtection TRansformer
# This is currently the State-Of-The-Art Transformer architecture for real-time object detection
# It natively supports the YOLO dataset format you have in the OIDv4_ToolKit directory.

def main():
    print("Initializing RT-DETR (Real-Time DEtection TRansformer)...")
    model = RTDETR("rtdetr-l.pt")

    # Start training on your dataset
    print("Starting fully fine-tuning of RT-DETR...")
    results = model.train(
        data="/home/Joshua/Downloads/leopard_toad_identification/dataset/dataset/data.yaml",
        epochs=100,            # Adjust epochs as needed
        imgsz=640,             # Standard image size
        batch=16,               # Adjust batch size based on GPU VRAM
        device="cuda" if torch.cuda.is_available() else "cpu",
        name="rtdetr_finetuning",
        optimizer="auto",
        lr0=0.0001,            # Typical learning rate for fine-tuning transformers
        patience=20            # Early stopping patience
    )
    
    # Evaluate model on validation set
    print("Evaluating model...")
    metrics = model.val()
    
    print("\nTraining completed! Weights and plots are saved in 'rtdetr_finetuning'")

if __name__ == "__main__":
    main()
