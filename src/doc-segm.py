

from modelscope.pipelines import pipeline

from text.AliTextParagraphSplitter import AliTextParagraphSplitter

#text = ('移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。')

# spliter = AliTextParagraphSplitter()
# sent_list = spliter.split_text(text)
#
# for sent in sent_list:
#     print(sent)

source_name = 'D:\\code\\LLM\\yuliao\\shuihuchuan_shinaian.txt'

with open(source_name, 'r', encoding='utf-8') as file:
    content = file.read()

#print(content)
spliter = AliTextParagraphSplitter()
sent_list = spliter.split_text(content)

for sent in sent_list:
    print(sent)

import re

# 输入一个段落，分成句子，可使用split函数来实现
paragraph = "生活对我们任何人来说都不容易！我们必须努力，最重要的是我们必须相信自己。 \
我们必须相信，我们每个人都能够做得很好，而且，当我们发现这是什么时，我们必须努力工作，直到我们成功。"

# sentences = re.split('(。|！|\!|\.|？|\?)', paragraph)  # 保留分割符
#
# new_sents = []
# for i in range(int(len(sentences) / 2)):
#     sent = sentences[2 * i] + sentences[2 * i + 1]
#     new_sents.append(sent)


