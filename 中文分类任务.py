from datasets import load_dataset
from transformers import BertTokenizer

# 加载编码器工具

tokenizer=BertTokenizer.from_pretrained('bert-base-chinese')

print(tokenizer)

out=tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=['从明天起，做一个幸福的人。','喂马,周游世界'],
    truncation=True,
    padding='max_length',
    max_length=17,
    return_tensors='pt',
    return_length= True
)
for k,v in out.items():
    print(k,v)

print(tokenizer.decode(out['input_ids'][0]))