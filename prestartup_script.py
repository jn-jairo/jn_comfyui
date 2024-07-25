import sys
import os

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

sys.path.append(MODULE_PATH)

from config import config
import patch

patch.apply_patches(config)
