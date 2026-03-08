# from setuptools.dist import sequence
from transformers import pipeline

# pipeline1 = pipeline('sentiment-analysis')
#
# r= pipeline1("i hate you")
# print(r)
# # r=classifier('i hate you')
#
# # print(r)
#
# # 阅读理解
# question_answerer=pipeline("question-answering")
#
# context = r""" Extractive question answering is a type of question answering task where the model must extract an answer directly from a given context or passage.
# The answer is always a continuous span of text that already exists in the context — the model does not generate new sentences or words. It only predicts the start and end positions of the answer inside the context.
# """
# r= question_answerer(question="what is good example of a question answering dataset?",context=context)
# print(r)
#
#
# sentence = 'HuggingFace is creating a <mask> that the community uses to solve NLP tasks'
#
# unmasker = pipeline("fill-mask")
# print(unmasker(sentence))
#
#
# text_generation=pipeline("text-generation")
# r=text_generation("as far as I am concerned,I will", max_length=50,do_sample=False)
# print(r)
#
#
# sequence="""
# """
# ner=pipeline('ner')
# for entity in ner(sequence):
#     print(entity)
#
#
# ARTICLE="""
# In recent years, the deployment scale of self-service entertainment facilities such as self-service KTV, self-service billiards and self-service chess-and-card room in commercial outlets has been expanding flourishingly, as a result the number of equipment in chain stores has grown rapidly. But nowadays most equipments are highly dispersed, and the fluctuation between peak and off-peak periods is apparent, so the traditional experience-based maintenance approaches are suffering from low efficiency. On one hand, the downtime losses caused by sudden equipment failures are difficult to control in a timely manner according to the current mode; on the other hand, the mismatch between service capacity during peak hours and the allocation of operation and maintenance resources directly impacts store revenue performance and will lower user experience. In this context, how to ensure stable equipment operation in these limited maintenance resources and effectively transform equipment status information into operational decision support becomes a practical challenge for the self-service entertainment industry.
# This study uses the full-year 2025 operational data from 24 devices across three stores in Qinhuangdao as its sample. It develops a dual-engine framework consisting of LSTM-based sales forecasting (MAPE: 12.3%) and CNN-VAE-based fault early warning (recall: 90.8%, average lead time: 38 hours). Through a semi-automated process of “forecast–rule mapping–human review,”the framework is embedded into store scheduling and maintenance decisions, and the Difference-in-Differences (DID) method is employed to evaluate the pilot effects. The result shows that the treatment group experienced a 4.8 percentage point increase in equipment availability, an average daily revenue increase of 56.2 RMB, and a reduction of 0.95 failures per day (p < 0.01); the marginal benefits during peak hours were significantly higher.
# The study outcome shows that the value of forecasting does not depend solely on accuracy, but also more critically depend on whether it can operate in efficient coordination with the executive systems. Gradual automation strategies and dynamic threshold calibration are key mechanisms for balancing efficiency and trust. Theoretically, this study will extend predictive maintenance mode from heavy-asset contexts to light-service scenarios, proposes two sets of designs—decision-making under varying conditions and threshold strategies—and elevates interpretability as an organizational adoption variable. Practically, it provides a layered implementation pathway, process interfaces, and a cost-benefit framework.
# """
# summarizer = pipeline("summarization")
#
# r=summarizer(ARTICLE,max_length=130,min_length=30, do_sample=False)
# print(r)
#
# sentence="Hello world"
# translator=pipeline("translation_en_to_fr")
#
# r=translator(sentence)
# print(r)


from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

sentence='我叫拉萨，我住在伦敦'
translater = pipeline(task='translation_zh_to_en', model=model, tokenizer=tokenizer, max_length=130)

r=translater(sentence)
print(r)


