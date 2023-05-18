from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

ort_model = ORTModelForSequenceClassification.from_pretrained(
    "ModelTC/bert-base-uncased-mrpc",export=True,
    provider="CUDAExecutionProvider",)

tokenizer = AutoTokenizer.from_pretrained("ModelTC/bert-base-uncased-mrpc")
inputs = tokenizer("expectations were low, actual enjoyment was high", return_tensors="pt", padding=True)

outputs = ort_model(**inputs)