from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ.get("GCS_BUCKET", "mlops-hoangtruongminh22-977661303")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = "/tmp/models/model.pkl"


def download_model():
    """
    Tai file model.pkl tu GCS ve may khi server khoi dong.
    """
    os.makedirs("/tmp/models", exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_KEY)
    blob.download_to_filename(MODEL_PATH)
    print("Model da duoc tai xuong tu GCS.")


try:
    download_model()
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Khong the tai model tu GCS: {e}")
    model = None


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected 12 features (wine quality)"
        )

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pred = model.predict([req.features])[0]

    labels = {0: "thap", 1: "trung_binh", 2: "cao"}

    return {"prediction": int(pred), "label": labels.get(int(pred), "unknown")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
