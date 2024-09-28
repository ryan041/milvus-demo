import time

from pymilvus import connections, MilvusClient, DataType
from transformers import AutoTokenizer, AutoModel
from milvus_model.hybrid import BGEM3EmbeddingFunction

import my_logger

logger = my_logger.create_logger()


model_path = 'D:\\code\\LLM\\bge-m3'

startTime = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=model_path, # Specify the model name
    device='cpu', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)

logger.info("初始化model 耗时={:.3f} s".format((time.time() - startTime)))

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]


docs_embeddings = bge_m3_ef.encode_documents(docs)
denseVector = docs_embeddings["dense"]

# Print embeddings
#print("Embeddings:", docs_embeddings)
# Print dimension of dense embeddings
#print("Dense document dim:", bge_m3_ef.dim["dense"], docs_embeddings["dense"][0].shape)
# Since the sparse embeddings are in a 2D csr_array format, we convert them to a list for easier manipulation.
#print("Sparse document dim:", bge_m3_ef.dim["sparse"], list(docs_embeddings["sparse"])[0].shape)

# Each entity has id, vector representation, raw text, and a subject label that we use
# to demo metadata filtering later.
data = [
    {"vector": denseVector[i], "text": docs[i], "subject": "history"}
    for i in range(len(denseVector))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))



import demo_collection_util

uri = 'http://192.168.2.10:19530'
db_name = 'ryan'
tb_name = 'demo'

collMgr = demo_collection_util.DemoCollectionMgr(uri, db_name)
if not collMgr.get_client().has_collection(collection_name=tb_name):
    collMgr.create_deme_collection(tb_name)


# insert data to Milvus
#res = collMgr.get_client().insert(collection_name=tb_name, data=data)


queries = ["When was AI founded",
           "Where was Alan Turing born?"]

query_embeddings = bge_m3_ef.encode_queries(queries)

# Print embeddings
#print("Embeddings:", query_embeddings)
# Print dimension of dense embeddings
#print("Dense query dim:", bge_m3_ef.dim["dense"], query_embeddings["dense"][0].shape)
# Since the sparse embeddings are in a 2D csr_array format, we convert them to a list for easier manipulation.
#print("Sparse query dim:", bge_m3_ef.dim["sparse"], list(query_embeddings["sparse"])[0].shape)

res = collMgr.get_client().search(collection_name=tb_name, data=query_embeddings["dense"], limit=2, output_fields=["text", "subject"])
#logger.info(res)

for re in res:
    for e in re:
        logger.info("distance={} text={}".format(e["distance"], e["entity"]["text"]))
