from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import string
from time import time

def preprocess_data(text: str) -> str:
   return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

MODEL_PATH = "helinivan/english-sarcasm-detector"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)


def detect_sarcasm(query):

   tokenized_text = tokenizer([preprocess_data(query)], padding=True, truncation=True, max_length=256, return_tensors="pt")
   output = model(**tokenized_text)
   probs = output.logits.softmax(dim=-1).tolist()[0]
   confidence = max(probs)
   prediction = probs.index(confidence)
   return {"is_sarcastic": prediction, "confidence": confidence}

start = time()
print(detect_sarcasm("It's good, at least your bag isn't deep."))
print(time()-start)