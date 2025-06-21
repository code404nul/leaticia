from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from emotion_eval import emotion
import torch
from json import dump, load
from huggingface_hub import snapshot_download
from os.path import dirname, join

BASE_DIR = dirname(__file__)
MEMORY_PATH = join(BASE_DIR, "personality.json")

local_model_path = snapshot_download("KevSun/Personality_LM")

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSequenceClassification.from_pretrained(local_model_path, ignore_mismatched_sizes=True)

model.eval()

def predict_traits(text):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        output = model(**encoded)
    probs = torch.nn.functional.softmax(output.logits, dim=-1)[0]
    return probs.tolist()

def full_profile(text):
    """
    Give result of the BIG 5 result
    input : text

    result : 
    - True if everything end correctly
    - False is the file couldn't be writted
    """
    sentences = sent_tokenize(text, language='english')
    trait_sums = torch.zeros(5)
    for sentence in sentences:
        emotion_charge =  emotion.index_emotionnal_charge(emotion.emotion_classify(sentence))
        if not (emotion_charge > 0.3 and emotion_charge < -0.3):
            trait_sums += torch.tensor(predict_traits(sentence))
        else: print(f"{sentence} et une phrase trop neutre pour une charge emotionnel de {emotion_charge}")

    trait_avg = trait_sums / len(sentences)
    trait_names = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]

    try:
        with open("find_prompt/personality.json", "a") as outfile:
            dump(dict(zip(trait_names, [round(score.item(), 4) for score in trait_avg])), outfile)
        return True
    except: return False

def get_personnality():
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        data = load(f)
    return data