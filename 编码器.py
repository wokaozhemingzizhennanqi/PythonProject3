from transformers import BertTokenizer, BertModel


sentence = 'hello everyone , today is a good day .'
print(sentence)

vocab = {
    '<SOS>':0,
    '<EOS>':1,
    'hello':2,
    'everyone':3,
    'today':4,
    'is':5,
    'are':6,
    'good':7,
    'day':8,
    ',':9,
    '.':10
}
sent = '<SOS> '+sentence+' <EOS>'

words = sent.split(' ')
print(words)

tokenizer = BertTokenizer.from_pretrained(
    'nlptown/bert-base-multilingual-uncased-sentiment',
    cache_dir='./bert-base-uncased',
    force_download=False,
)

sents = ['你站在桥上看风景','看风景的人在楼上看你','明月装饰了你的窗子','你装饰了别人的梦 ']

# 基本编码函数
out = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],
    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    max_length=25,
    return_tensors=None,
)

print(out)
rs=tokenizer.decode(out)
print(rs)
# for i in words:
#     pass


outs = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[(sents[0],sents[1]),(sents[2],sents[3])],
    truncation=True,
    padding='max_length',
    max_length=25,
    add_special_tokens=True,
    # return_tensors='None',
    return_token_type_ids=True,
    return_special_tokens_mask=True,
    return_attention_mask=True,
    return_length=True,
)

for k,v in outs.items():
    print(k,' : ',v)

# 字典的操作
# 获取字典
vocab = tokenizer.get_vocab()
print(len(vocab))


tokenizer.add_tokens(new_tokens=['明月','装饰','窗子'])
tokenizer.add_special_tokens({'eos_token':'[EOS]'})

for word in ['明月','装饰','窗子']:
    print(tokenizer.get_vocab()[word])



out = tokenizer.encode(text='明月装饰了你的窗子 .',text_pair='None',truncation=True,padding=True,add_special_tokens=True,max_length=25)
print(out)

out = tokenizer.decode(out)
print(out)