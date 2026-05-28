import os
import sys

ACTIVE_LEARNING_DIR = os.path.dirname(os.path.abspath(__file__))
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.insert(0, ACTIVE_LEARNING_DIR)

# Expose everything from the central active learning configuration
from central_config import *
