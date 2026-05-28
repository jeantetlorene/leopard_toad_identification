"""
Active Learning package for Western Leopard Toad identification.
Consolidates training, batch inference, static false positive filtering, and active curation.
"""

import os
import sys

ACTIVE_LEARNING_DIR = os.path.dirname(os.path.abspath(__file__))
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)

# Expose key configurations and pipelines
from central_config import *
