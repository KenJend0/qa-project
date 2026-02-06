from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import os
import time

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use absolute paths to avoid issues with working directory changes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODELS = {
    "distilbert": os.path.join(PROJECT_ROOT, "notebooks", "outputs", "checkpoints", "distilbert", "final"),
    "bert": os.path.join(PROJECT_ROOT, "notebooks", "outputs", "checkpoints", "bert", "final"),
    "roberta": os.path.join(PROJECT_ROOT, "notebooks", "outputs", "checkpoints", "roberta", "final"),
}

tokenizers = {}
models = {}

for name, path in MODELS.items():
    tokenizers[name] = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=True)
    model.to(DEVICE)
    model.eval()
    models[name] = model


class QARequest(BaseModel):
    context: str
    question: str
    model_name: str


@app.post("/predict")
def predict(req: QARequest):
    # Validate model name
    if req.model_name not in models:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model_name}. Available: {list(models.keys())}")
    
    tokenizer = tokenizers[req.model_name]
    model = models[req.model_name]

    inputs = tokenizer(
        req.question,
        req.context,
        return_tensors="pt",
        truncation=True,
        max_length=384
    )

    # FIX: RoBERTa doesn't use token_type_ids
    if req.model_name == "roberta":
        inputs.pop("token_type_ids", None)

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Measure inference latency
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    latency_ms = (time.time() - start_time) * 1000

    # Extract answer with length constraint
    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    max_answer_length = 15
    best_score = -1e9
    best_start, best_end = 0, 0

    for start_idx in range(len(start_logits)):
        for end_idx in range(start_idx, min(start_idx + max_answer_length, len(end_logits))):
            score = start_logits[start_idx] + end_logits[end_idx]
            if score > best_score:
                best_score = score
                best_start, best_end = start_idx, end_idx

    answer = tokenizer.decode(
        inputs["input_ids"][0][best_start:best_end + 1],
        skip_special_tokens=True
    )

    # Ensure we never return empty answer
    if not answer.strip():
        answer = "No answer found."

    return {
        "answer": answer,
        "score": float(best_score),
        "latency_ms": latency_ms
    }
