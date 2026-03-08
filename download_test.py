import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import BertTokenizer

try:
    print("Attempting to download BertTokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=True)
    print("Successfully downloaded!")
    tokenizer.save_pretrained('./bert_local')
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
