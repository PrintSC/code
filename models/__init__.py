# models 🚀 AGPL-3.0 License - https://models.com/license

__version__ = "8.3.74"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from models.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from models.utils import ASSETS, SETTINGS
from models.utils.checks import check_yolo as checks
from models.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
)
