import numpy as np
import torch.cuda
from pymilvus import MilvusClient

#client = MilvusClient(uri="http://192.168.2.10:19530", db_name="default")

#if client.has_collection(collection_name="demo_collection"):
#    client.drop_collection(collection_name="demo_collection")
#client.create_collection(
#    collection_name="demo_collection",
#    dimension=768,  # The vectors we will use in this demo has 768 dimensions
#)

#from milvus_model.hybrid import BGEM3EmbeddingFunction
#ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
#dense_dim = ef.dim["dense"]

from transformers import AutoTokenizer, AutoModel

model_path = 'D:\\code\\milvus\\bge-m3'

#model_name = "bert-base-multilingual-cased"  # 这里使用的是BERT的多语言版本，你可以根据需要替换为其他模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# 定义一个函数，将文本转换为embeddings
def text_to_embeddings(text):
    # 对文本进行编码
    inputs = tokenizer(text, return_tensors="pt")
    # 获取模型的输出
    outputs = model(**inputs)
    # 提取最后一层的隐藏状态作为embeddings
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

# 示例：将文本转换为embeddings
text = "这是一个示例文本。"
embeddings = text_to_embeddings(text)
print(embeddings)




def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def l2_normalize(vector):
    norm = np.linalg.norm(vector, ord=2)
    if norm == 0:
        return vector
    return vector / norm




tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)



device = "cuda:0" if torch.cuda.is_available() else "cpu"

input_text = "Hugging Face is creating a tool that democratizes AI."
encoded_input = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt').to(device)
model_input = model.to(device)

# Compute token embeddings
with torch.no_grad():
    model_output = model_input(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Convert to a list and return as JSON response
embeddings_list = sentence_embeddings.tolist()
# normalize handle
normalized_embeddings_list = l2_normalize(np.array(embeddings_list))

print(normalized_embeddings_list.tolist())




