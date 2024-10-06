from typing import List
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# #从model scope下载文件
# from modelscope import snapshot_download
# cache_dir='D:\\code\\LLM'  #下载文件缓存路径
#
# #开始下载模型文件，并返回模型文件所在的路径
# model_dir = snapshot_download(model_id="iic/nlp_bert_document-segmentation_chinese-base", cache_dir=cache_dir)

# TODO 需要改为全局环境变量
model_dir = 'D:\\code\\LLM'


class AliTextParagraphSplitter:
    """
    阿里文本分段模型，主要用于对未分段的文章进行自动分段
    """

    @staticmethod
    def split_text(text: str) -> List[str]:
        model_name = 'iic\\nlp_bert_document-segmentation_chinese-base'

        p = pipeline(
            task=Tasks.document_segmentation,
            model=model_dir + '\\' + model_name,  # 'damo/nlp_bert_document-segmentation_chinese-base',
            device="cpu")
        result = p(documents=text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list
