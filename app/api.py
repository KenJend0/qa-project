from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "distilbert": "outputs/checkpoints/distilbert/final",
    "bert": "outputs/checkpoints/bert/final",
    "roberta": "outputs/checkpoints/roberta/final",
}

tokenizers = {}
models = {}

for name, path in MODELS.items():
    tokenizers[name] = AutoTokenizer.from_pretrained(path)
    model = AutoModelForQuestionAnswering.from_pretrained(path)
    model.to(DEVICE)
    model.eval()
    models[name] = model


class QARequest(BaseModel):
    context: str
    question: str
    model_name: str


@app.post("/predict")
def predict(req: QARequest):
    tokenizer = tokenizers[req.model_name]
    model = models[req.model_name]

    inputs = tokenizer(
        req.question,
        req.context,
        return_tensors="pt",
        truncation=True,
        max_length=384
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)

    answer = tokenizer.decode(
        inputs["input_ids"][0][start_idx:end_idx + 1],
        skip_special_tokens=True
    )

    return {"answer": answer}
