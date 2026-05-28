import os
import sys

# Resolve parent directory to import the central configuration
PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)

if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.insert(0, ACTIVE_LEARNING_DIR)

# Import everything from the central active learning config (uniquely named to avoid collision)
from central_config import *
