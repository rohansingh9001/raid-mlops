from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import load, predict
import uvicorn

app = FastAPI()

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

@app.get("/predict")
async def query(query_input: QueryInput):
    # Hardcoded prediction for demonstration
    prediction = predict(query_input.query, model, tokenizer, max_len)
    return {"prediction": prediction}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)