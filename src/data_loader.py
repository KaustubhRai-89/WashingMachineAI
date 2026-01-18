import os
import cv2
import yaml
import sqlite3
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime

try:
    from src.augment_dirt import add_dirt
except ImportError:
    from augment_dirt import add_dirt

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

KEYWORD_MAP = {
    "denim": "denim", "jeans": "denim",
    "cotton": "cotton", "t-shirt": "cotton", "tee": "cotton",
    "wool": "wool", "knit": "wool", "sweater": "wool",
    "poly": "polyester", "sport": "polyester", "jersey": "polyester",
    "silk": "silk", "satin": "silk"
}


def init_db():
    conn = sqlite3.connect(config["DB_PATH"])
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            original_path TEXT,
            fabric_label TEXT,
            dirt_label TEXT,
            dirt_intensity REAL,
            created_at TEXT
        )
    ''')
    conn.commit()
    return conn


def get_fabric_label(path):
    path_lower = path.lower()
    for keyword, label in KEYWORD_MAP.items():
        if keyword in path_lower:
            return label
    return "unknown"


def process_dataset():
    conn = init_db()
    cursor = conn.cursor()

    raw_path = config["RAW_DATA_DIR"]
    files = glob(os.path.join(raw_path, "**", "*.jpg"), recursive=True) + \
            glob(os.path.join(raw_path, "**", "*.png"), recursive=True) + \
            glob(os.path.join(raw_path, "**", "*.jpeg"), recursive=True)

    count = 0

    for file_path in tqdm(files):
        try:
            image = cv2.imread(file_path)
            if image is None:
                continue

            target_size = tuple(config["IMAGE_SIZE"])
            image_resized = cv2.resize(image, target_size)

            fabric_label = get_fabric_label(file_path)

            filename_clean = f"clean_{count}.jpg"
            save_path_clean = os.path.join(config["PROCESSED_DATA_DIR"], filename_clean)
            cv2.imwrite(save_path_clean, image_resized)

            cursor.execute("INSERT INTO dataset VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                           (filename_clean, file_path, fabric_label, "clean", 0.0, datetime.now()))

            import random
            dirt_types_to_apply = random.sample(config["DIRT_CLASSES"], 2)

            for dirt_type in dirt_types_to_apply:
                if dirt_type == "clean":
                    continue

                intensity_factor = random.uniform(0.3, 0.9)
                image_dirty, intensity_score, _ = add_dirt(image_resized, dirt_type, intensity_factor)

                filename_dirty = f"dirty_{dirt_type}_{count}.jpg"
                save_path_dirty = os.path.join(config["SYNTHETIC_DIR"], filename_dirty)
                cv2.imwrite(save_path_dirty, image_dirty)

                cursor.execute("INSERT INTO dataset VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                               (filename_dirty, file_path, fabric_label, dirt_type, intensity_score, datetime.now()))

            count += 1

            if count % 100 == 0:
                conn.commit()

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    conn.commit()
    conn.close()
    print(f"Processing Complete! Processed {count} source images.")
    print(f"Check {config['DB_PATH']} for the data registry.")


if __name__ == "__main__":
    os.makedirs(config["PROCESSED_DATA_DIR"], exist_ok=True)
    os.makedirs(config["SYNTHETIC_DIR"], exist_ok=True)

    process_dataset()