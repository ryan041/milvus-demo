from pymilvus import MilvusClient

client = MilvusClient(uri="http://192.168.2.10:19530", db_name="default")

#if client.has_collection(collection_name="demo_collection"):
#    client.drop_collection(collection_name="demo_collection")
#client.create_collection(
#    collection_name="demo_collection",
#    dimension=768,  # The vectors we will use in this demo has 768 dimensions
#)


#*************** prepare data
from pymilvus import model

# If connection to https://huggingface.co/ failed, uncomment the following path
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
embedding_fn = model.DefaultEmbeddingFunction()



# Text strings to search from.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = embedding_fn.encode_documents(docs)
# The output vector has 768 dimensions, matching the collection that we just created.
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

# Each entity has id, vector representation, raw text, and a subject label that we use
# to demo metadata filtering later.
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))


#*************** insert data
#res = client.insert(collection_name="demo_collection", data=data)
#print(res)

#*************** Vector search
query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(res)









