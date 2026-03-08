import torch
from datasets import load_dataset, load_from_disk

# dataset= load_dataset("liming751218/ChnSentiCorp")
#
#
# dataset.save_to_disk("./dataset/ChnSentiCorp")

class Dataset(torch.utils.data.Dataset):
    def __init__(self,split):
        self.dataset = load_from_disk('./dataset/ChnSentiCorp')[split]

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        text= self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        return text, label

dataset = Dataset('train')
len(dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#
#
# dataset =load_from_disk("./dataset/ChnSentiCorp")
# # # print(dataset)
# #
# # # 加载子集
# # # load_dataset(path='',name='ChnSentiCorp',split='train')
# #
# dataset = dataset['train']
# #
# # sorted_dataset = dataset.sort('label')
# # print(sorted_dataset[0])
#
# # def f(data):
# #     # print(data['text'])
# #     return data['text'].startswith('非常不错')
# #
# # sorted_dataset.filter(f)
# #
# # print(len(sorted_dataset))
# #
# # res=dataset.shard(num_shards=3, index=0)
# # print(res)
# #
# # # def f(data):
# # #     text = data['text']
# #     # text = ['my sentence',]
#
#
# # dataset.to_csv(path_or_buf='./dataset/ChnSentiCorp-train.csv')
#
# # csv= load_dataset(path='csv',data_file='./dataset/ChnSentiCorp-train.csv',split='train')
# # print(csv[20])
#
# # dataset.to_json('./dataset/ChnSentiCorp-train.json')
#
# json = load_dataset(path='json',data_file='./dataset/ChnSentiCorp-train.json',split='train')
# print(json[20])