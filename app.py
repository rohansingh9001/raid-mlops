from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import load, predict
import uvicorn


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Add the allowed HTTP methods
    allow_headers=["*"]  # Add the allowed headers
)

class QueryInput(BaseModel):
    query: str

# Define the directory where the model is saved
model_path = 'models/text_classification_model.h5'
tokenizer_path = 'tokenizer/tokenizer.pkl'
max_sequence_path = 'tokenizer/max_sequence_length.txt'

model, tokenizer, max_len = load(model_path, tokenizer_path, max_sequence_path)

@app.get("/ping")
async def ping():
    return "pong"

@app.post("/predict")
async def query(query_input: QueryInput):
    # Hardcoded prediction for demonstration
    prediction = predict(query_input.query, model, tokenizer, max_len)
    return {"prediction": prediction}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
