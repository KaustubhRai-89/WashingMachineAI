import os
import cv2
import yaml
import json
import sqlite3
import time
import google.generativeai as genai
from tqdm import tqdm

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=API_KEY)


def get_gemini_label(image_path):
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = """
    Analyze this close-up of laundry. 
    1. Identify the Fabric Type (choose one: cotton, denim, wool, polyester, silk).
    2. Identify the Dirt Type (choose one: clean, mud, oil, grass, wine).
    3. Estimate Dirt Intensity (1 to 10).

    Return ONLY a raw JSON string like this:
    {"fabric": "cotton", "dirt": "mud", "intensity": 5}
    """

    try:
        sample_file = genai.upload_file(image_path)
        response = model.generate_content([prompt, sample_file])
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text_response)

    except Exception as e:
        print(f"API Error on {image_path}: {e}")
        return None


def run_labeling_job(limit=50):
    conn = sqlite3.connect(config["DB_PATH"])
    cursor = conn.cursor()

    cursor.execute("SELECT id, filename, original_path FROM dataset WHERE fabric_label='unknown' LIMIT ?", (limit,))
    rows = cursor.fetchall()

    if not rows:
        print("No 'unknown' images found in Database! Good job on the regex.")
        return

    print(f"Starting AI Labeling for {len(rows)} images...")

    for row in tqdm(rows):
        db_id, filename, original_path = row

        if not os.path.exists(original_path):
            print(f"File missing: {original_path}")
            continue

        result = get_gemini_label(original_path)

        if result:
            new_fabric = result.get("fabric", "unknown").lower()

            cursor.execute("""
                UPDATE dataset 
                SET fabric_label = ? 
                WHERE id = ?
            """, (new_fabric, db_id))

            conn.commit()

            time.sleep(4)

    conn.close()
    print("Teacher Labeling Complete.")


if __name__ == "__main__":
    if not os.path.exists(config["DB_PATH"]):
        print("Error: Database not found. Run src/data_loader.py first.")
    else:
        run_labeling_job(limit=10)