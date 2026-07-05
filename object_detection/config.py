import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_ANNO_DIR = RAW_DATA_DIR / "Indoor_Object_Detection_Dataset/Indoor_Object_Detection_Dataset/annotation"
RAW_DATA_IMG_DIR = RAW_DATA_DIR / "Indoor_Object_Detection_Dataset/Indoor_Object_Detection_Dataset"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

DATA_YAML = PROCESSED_DATA_DIR / "yolo" / "data.yaml"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RUN_DIR = REPORTS_DIR / "yolo_runs"
RUN_NAME = "yolo11m_indoor"


RATIOS = (0.8, 0.1, 0.1)   # train, val, test
SEED = 42

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
