from datasets import load_dataset
from transformers import AutoProcessor, ClapModel

dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
audio_sample = dataset["train"]["audio"][0]["array"]

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

input_text = ["Sound of a dog", "Sound of vaccum cleaner"]

inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
print(probs)
