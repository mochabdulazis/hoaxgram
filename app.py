from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()
# Load sekali saja
MODEL_NAME = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained("Syetsuki/hoaxgram_app")

class RequestText(BaseModel):
    text: str

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    score = torch.max(probs).item()
    label_id = torch.argmax(probs).item()

    label_map = {0: "non-hoax", 1: "hoax"}
    return label_map[label_id], score

@app.post("/predict")
def predict(req: RequestText):
    label, score = predict_text(req.text)
    return {"label": label, "score": score}
