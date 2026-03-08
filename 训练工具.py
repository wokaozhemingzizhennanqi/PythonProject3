import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
from datasets import load_dataset, load_from_disk, load_metric
from transformers.trainer_utils import EvalPrediction
import numpy as np

# 全局定义，确保pickling兼容性
tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')

def f_map(data, tokenizer=tokenizer):
    return tokenizer.batch_encode_plus(data['text'], truncation=True)

def f_filter(data):
    return [len(i) < 512 for i in data['input_ids']]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = logits.argmax(-1)
    return {'accuracy': (pred == labels).mean()}

if __name__ == '__main__':
    print(tokenizer)
    
    r = tokenizer.batch_encode_plus(['明月装饰了你的窗子', '你装饰了别人的梦 '])
    print(r)

    try:
        dataset = load_from_disk("./dataset/ChnSentiCorp")
    except Exception:
        dataset = load_dataset("seamew/ChnSentiCorp", trust_remote_code=True)

    # 缩小数据规模
    dataset['train'] = dataset['train'].shuffle().select(range(2000))
    dataset['test'] = dataset['test'].shuffle().select(range(100))

    print(dataset['train'][0])
    print("---------1")

    # 移除num_proc参数以在主进程运行，避免Windows多进程错误
    dataset = dataset.map(f_map, batched=True, batch_size=1000, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer})

    print("---------2", dataset['train'][0])

    dataset = dataset.filter(f_filter, batched=True, batch_size=1000)

    model = AutoModelForSequenceClassification.from_pretrained('hfl/rbt3', num_labels=2)
    print(dataset)
    
    print(sum([i.nelement() for i in model.parameters()]))

    # 模型试算
    data = {
        'input_ids': torch.ones(4, 10, dtype=torch.long),
        'token_type_ids': torch.ones(4, 10, dtype=torch.long),
        'attention_mask': torch.ones(4, 10, dtype=torch.long),
        'labels': torch.ones(4, dtype=torch.long)
    }

    out = model(**data)
    print(out)

    # 模拟输出
    eval_pred = EvalPrediction(
        predictions=np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
        label_ids=np.array([1, 1, 0, 1]),
    )
    print(compute_metrics(eval_pred))

    training_args = TrainingArguments(
        output_dir='./output',
        evaluation_strategy='steps',
        eval_steps=30,
        save_strategy='steps',
        save_steps=30,
        num_train_epochs=1,
        learning_rate=1e-4,
        weight_decay=1e-2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        no_cuda=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    # 测试数据整理函数
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_sample = dataset['train'][:5]
    data_sample = data_collator(data_sample)

    for k, v in data_sample.items():
        print(k, v.shape)
    print(data_sample)

    print("------------4")
    print(trainer.evaluate())
    
    print("------------5")
    # trainer.train()
    print("------------6")

    print(trainer.evaluate())

    trainer.save_model('./model')

    # 加载模型
    from transformers import AutoModelForSequenceClassification as _AM
    model = _AM.from_pretrained('./model')

    # 预测
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 取一批数据进行预测
    for i, data in enumerate(trainer.get_train_dataloader()):
        break
    
    for k, v in data.items():
        data[k] = v.to(device)
    
    out = model(**data)
    pred = out.logits.argmax(dim=-1)

    for i in range(min(16, len(pred))):
        print(tokenizer.decode(data['input_ids'][i], skip_special_tokens=True))
        print('label', data['labels'][i].item())
        print('predict=', pred[i].item())
    print(out)
