import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')

print(tokenizer)

r=tokenizer.batch_encode_plus(['明月装饰了你的窗子','你装饰了别人的梦 '])
print(r)

from datasets import load_dataset, load_from_disk

dataset =load_from_disk("./dataset/ChnSentiCorp")

# 缩小数据规模
dataset['train'] = dataset['train'].shuffle().select(range(2000))
dataset['test'] = dataset['test'].shuffle().select(range(100))

print(dataset['train'][0])

print("---------1")
def f1(data,tokenizer):
    return tokenizer.batch_encode_plus(data['text'],truncation=True)

dataset = dataset.map(f1,batched=True,batch_size=1000,num_proc=4,remove_columns=['text'],fn_kwargs={'tokenizer':tokenizer})

print("---------2",dataset['train'][0])

def f(data):
    return [len(i)<512 for i in data['input_ids']]

dataset = dataset.filter(f,batched=True,batch_size=1000,num_proc=4)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('hfl/rbt3',num_labels=2)
print(dataset)
r=sum([i.nelement() for i in model.parameters()])
print(r)
# 模型试算

data = {
    'input_ids': torch.ones(4,10,dtype=torch.long),
    'token_type_ids': torch.ones(4,10,dtype=torch.long),
    'attention_mask': torch.ones(4,10,dtype=torch.long),
    'labels': torch.ones(4,dtype=torch.long)
}

out=model(**data)

print(out)

# 定义评价函数

from datasets import  load_metric
import numpy as np
from transformers.trainer_utils import EvalPrediction
metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    logits,labels = eval_pred
    pred = logits.argmax(-1)
    return {'accuracy':(pred==labels).mean()}
    # return metric.compute(predictions=pred,references=labels)
# 模拟输出
eval_pred=EvalPrediction(
    predictions=np.array([[0,1],[2,3],[4,5],[6,7]]),
    label_ids = np.array([1,1,0,1]),
)
r=compute_metrics(eval_pred)
print(r)
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir='./output',
    # 定义测试执行策略
    evaluation_strategy='steps',
    # 定义每隔多少step执行一次
    eval_steps=30,
    #保存策略
    save_strategy='steps',
    #每隔多少step保存一次
    save_steps=30,
    #总共训练轮次
    num_train_epochs=1,
    #定义学习率
    learning_rate=1e-4,
    #假如参数权重衰减,防止过拟合
    weight_decay=1e-2,
    #定义训练和测试时候的批次大小
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    #定义是否用GPU
    no_cuda=False,
)

# 定义训练器
from transformers import Trainer
from transformers.data.data_collator import DataCollatorWithPadding
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

# 测试数据整理函数
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
# 获取一批数据
data = dataset['train'][:5]
# 调用数据整理函数
data=data_collator(data)

for k,v in data.items():
    print(k,v.shape)

print(data)

#训练和测试

print("------------4")
#测试
r=trainer.evaluate()
print(r)

print("------------5")
# trainer.train()
print("------------6")

r=trainer.evaluate()
print(r)

import torch

trainer.save_model('./model')
