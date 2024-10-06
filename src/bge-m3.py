import time

from FlagEmbedding import BGEM3FlagModel
from milvus_model.hybrid import BGEM3EmbeddingFunction
from transformers import AutoTokenizer, AutoModel

import torch

print(torch.cuda.is_available())


model_path = 'D:\\code\\LLM\\bge-m3'
model_name = 'BAAI/bge-m3'

startTime = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=model_path,  # Specify the model name or a local model path
    device='cpu',  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)


#model = BGEM3FlagModel('/root/bge-m3', use_fp16=True)

queries = ["What is BGE M3?",
           "Defination of BM25"]
docs = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]


query_embeddings = bge_m3_ef.encode_documents(queries)['dense']
docs_embeddings = bge_m3_ef.encode_documents(docs)['dense']
#similarity = query_embeddings @ docs_embeddings.T
#print(similarity)

print(query_embeddings)
print(docs_embeddings)