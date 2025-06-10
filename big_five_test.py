from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from emotion_eval import emotion
import torch
from huggingface_hub import snapshot_download

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
    sentences = sent_tokenize(text, language='english')
    trait_sums = torch.zeros(5)
    for sentence in sentences:
        emotion_charge =  emotion.index_emotionnal_charge(emotion.emotion_classify(sentence))
        if not (emotion_charge > 0.3 and emotion_charge < -0.3):
            trait_sums += torch.tensor(predict_traits(sentence))
        else: print(f"{sentence} et une phrase trop neutre pour une charge emotionnel de {emotion_charge}")

    trait_avg = trait_sums / len(sentences)
    trait_names = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]
    return dict(zip(trait_names, [round(score.item(), 4) for score in trait_avg]))


user_text = """I love exploring new ideas, but I often procrastinate. I enjoy time alone to reflect. 
I sometimes enjoy debates with friends. I'm sensitive to criticism. I try to plan, but I get overwhelmed easily."""

print(full_profile(user_text))
