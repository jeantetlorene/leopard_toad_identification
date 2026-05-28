"""
Pipelines subpackage for batch object detection inference and static false positive filtering.
"""

import os
import sys

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
if PIPELINES_DIR not in sys.path:
    sys.path.append(PIPELINES_DIR)

from filter_static_false_positives import filter_static_detections
from run_inference_pipeline import process_folder
