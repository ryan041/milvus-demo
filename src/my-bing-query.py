import time

from milvus_model.hybrid import BGEM3EmbeddingFunction
from transformers import AutoTokenizer, AutoModel

from src import my_logger
from src.text.AliTextParagraphSplitter import AliTextParagraphSplitter

logger = my_logger.create_logger()

# ******************** 公共参数
root_home = 'D:\\code\\LLM\\'
startTime = time.time()

# ******************** 加载BGE-M3模型
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


# ************************ 从Milvus查询相关词条内容
import demo_collection_util

uri = 'http://192.168.2.10:19530'
db_name = 'ryan'
tb_name = 'book'

collMgr = demo_collection_util.DemoCollectionMgr(uri, db_name)


queries = ["洪太尉看到的石碑有多高？"]

query_embeddings = bge_m3_ef.encode_queries(queries)
search_params = {
    "metric_type": "IP",
    "params": {
        "radius": 0.5
    }
}

res = collMgr.get_client().search(
    collection_name=tb_name,
    data=query_embeddings["dense"],
    limit=5,
    search_params=search_params,
    output_fields=["text", "subject"])

if len(res) <= 0 or len(res[0]) <= 0:
    logger.info("can't hit your query")
else:
    logger.info("召回内容：")
    context = []
    for re in res:
        for e in re:
            logger.info("distance={} text={}".format(e["distance"], e["entity"]["text"]))
            context.append(e["entity"]["text"])

# ************************ 基于LLM回答
    import json
    from llamaapi import LlamaAPI

    # Initialize the SDK
    llama = LlamaAPI(
        api_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjlmZTgyZmU0LWExY2ItNDEwYy1iMTk0LTE5NDVjZWFhMzQyMSJ9.0KtCHeJs8vm4-AwMeldOdG4E4H6P8FvqSvH6SBvUlCc",
        hostname="http://localhost:11434",
        domain_path="/api/chat"
    )

    prompt_template ='基于以下已知信息，简洁和专业的来回答用户的问题。' \
                + '如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。' \
                + '已知内容:' \
                + '{}' \
                + '问题:' \
                + '{}'

    query_content = prompt_template.format(context, queries)
    logger.info('查询内容={}'.format(query_content))

    api_request_json = {
        "model": "llama3:8b",
        "messages": [
            {"role": "user", "content": query_content},
        ],
        "stream": False,
    }

    # Execute the Request
    response = llama.run(api_request_json)
    logger.info(json.dumps(response.json(), indent=2))
    logger.info(response.json().get("message").get("content"))



