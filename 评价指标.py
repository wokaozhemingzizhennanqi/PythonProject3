# 2.17 以上才有
# from datasets import load_metrics

from datasets import  load_metric

metric = load_metric('glue',config_name='mrpc')


print(metric)

prediction = [0,1,0]

reference = [0,1,0]

res= metric.compute(predictions=prediction,references=reference)

print(res)