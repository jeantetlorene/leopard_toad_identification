import torch
import os


class Config:
    DATA_DIR = "/home/Joshua/Downloads/leopard_toad_identification/identification/toads_by_id_crop"
    WEIGHTS_DIR = "/home/Joshua/Downloads/leopard_toad_identification/identification/sim_clr/weights"
    LOGS_DIR = (
        "/home/Joshua/Downloads/leopard_toad_identification/identification/sim_clr/logs"
    )

    IMG_SIZE = 224
    VAL_SPLIT = 0.2
    SEED = 42

    BATCH_SIZE = 32
    EPOCHS = 200
    LEARNING_RATE = 3e-4
    TEMPERATURE = 0.12
    EMBEDDING_DIM = 256
    EARLY_STOPPING_PATIENCE = 15
    WARMUP_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
