from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from collections import Counter
import os
import shutil
import pandas as pd
import subprocess

app = FastAPI()

# Allow cross-origin requests if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST" ,"GET"],
    allow_headers=["*"],
)

DB_FOLDER_ID = "1aDURFJyVuPwakguTfUvkMdujtOv71bJ2"  # 🔁 Replace this with your real folder ID
DB_PATH = "face_db"

def download_drive_folder():
    if not os.path.exists(DB_PATH):
        subprocess.run(["pip", "install", "gdown"])
        os.system(f"gdown --folder https://drive.google.com/drive/folders/{DB_FOLDER_ID} -O {DB_PATH} --fuzzy")

# Run once at startup
download_drive_folder()

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    temp_img_path = f"temp_{file.filename}"
    with open(temp_img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = DeepFace.find(
            img_path=temp_img_path,
            db_path=DB_PATH,
            model_name='Facenet',
            distance_metric='cosine',
            enforce_detection=False
        )

        if len(df) == 0:
            return {"message": "No matches found"}

        # Top 3 matches
        top3_ids = df.iloc[:3, 0].tolist()
        found_children = [path.split("/")[-2] for path in top3_ids]
        counter = Counter(found_children)
        most_common_class, count = counter.most_common(1)[0]

        return {
            "found_child": most_common_class,
            "match_count": count,
            "top_matches": found_children,
            "match_images": [os.path.basename(p) for p in top3_ids],
            "raw_paths": top3_ids
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
