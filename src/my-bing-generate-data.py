import time

from milvus_model.hybrid import BGEM3EmbeddingFunction
from transformers import AutoTokenizer, AutoModel

from src import my_logger
from src.text.AliTextParagraphSplitter import AliTextParagraphSplitter

logger = my_logger.create_logger()

# ******************** 公共参数
root_home = 'D:\\code\\LLM\\'
startTime = time.time()

# ******************** 读取文本，并进行分段
source_name = root_home + '\\yuliao\\shuihuchuan_shinaian-test.txt'
with open(source_name, 'r', encoding='utf-8') as file:
    content = file.read()

spliter = AliTextParagraphSplitter()
sent_list = spliter.split_text(content)
logger.info("分段 耗时={:.3f} s".format((time.time() - startTime)))

# ******************** 加载BGE-M3模型
startTime = time.time()
model_path = root_home + 'bge-m3'
model_name = 'BAAI/bge-m3'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=model_path,  # Specify the model name or a local model path
    device='cpu',  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)

logger.info("初始化model 耗时={:.3f} s".format((time.time() - startTime)))


# 对分段的内容计算embeddings，并插入Milvus
import demo_collection_util

uri = 'http://192.168.2.10:19530'
db_name = 'ryan'
tb_name = 'book'

collMgr = demo_collection_util.DemoCollectionMgr(uri, db_name)
if not collMgr.get_client().has_collection(collection_name=tb_name):
    collMgr.create_deme_collection(tb_name)


for i, sent in enumerate(sent_list):
    logger.info("{} {}".format(i, sent))
    sent_embeddings = bge_m3_ef.encode_documents([sent])['dense']
    logger.info("{} embedings={}".format(i, sent_embeddings))

    data = [
        {"vector": sent_embeddings[0], "text": sent, "subject": "shuihuzhuan"}
    ]

    res = collMgr.get_client().insert(collection_name=tb_name, data=data)




