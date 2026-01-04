from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .inference import Predictor
from .schemas import PredictionResponse
import uvicorn
import os

app = FastAPI(title="CNN Image Classifier")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
# We assume model.pth is in the same directory or we pass the path
# In our structure, it's in backend/model.pth
model_path = os.path.join(os.path.dirname(__file__), "model.pth")
predictor = Predictor(model_path=model_path)

@app.get("/")
def read_root():
    return {"message": "CNN Classifier API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = predictor.predict(contents)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
