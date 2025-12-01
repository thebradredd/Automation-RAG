from transformers import AutoTokenizer, Gemma3ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-it", device_map="auto")

print("Gemma 3 loaded successfully!")