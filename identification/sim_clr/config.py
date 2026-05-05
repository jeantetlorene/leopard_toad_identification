import torch
import os


class Config:
    # Default paths
    BASE_DATA_DIR = "/home/Joshua/Downloads/leopard_toad_identification/identification"
    DATA_DIR = os.path.join(BASE_DATA_DIR, "all_leopard_toad_chips")
    WEIGHTS_DIR = os.path.join(BASE_DATA_DIR, "sim_clr/weights/chips")
    LOGS_DIR = os.path.join(BASE_DATA_DIR, "sim_clr/logs/chips")
    PRETRAINED_BACKBONE = "/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/runs/faster_rcnn/train_resnet50/weights/best.pt"

    IMG_SIZE = 640
    VAL_SPLIT = 0.2
    SEED = 42

    BATCH_SIZE = 16
    EPOCHS = 200
    LEARNING_RATE = 3e-4
    TEMPERATURE = 0.12
    EMBEDDING_DIM = 256
    EARLY_STOPPING_PATIENCE = 15
    WARMUP_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
