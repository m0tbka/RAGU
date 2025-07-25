#from langchain_openai import ChatOpenAI

from ragu.utils.io_utils import read_text_from_files
from ragu.common.llm import RemoteLLM
from ragu.graph.graph_builder import KnowledgeGraphBuilder, KnowledgeGraph

from ragu.search_engine.local_search import LocalSearchEngine
from ragu.common.embedder import STEmbedder
from ragu.chunker.chunkers import SmartSemanticChunker
from ragu.chunker.chunkers import SemanticTextChunker
from ragu.chunker.chunkers import SimpleChunker
from ragu.triplet.kg_small_models_pipeline.model import TripletSMs
from ragu.common.index import Index

import json
import os
from llama_index.core import Document

class RAGDataset:
    def __init__(self, qa_file_path, documents_file_path):
        if not os.path.exists(qa_file_path):
            raise FileNotFoundError(f"Файл {qa_file_path} не найден.")

        if not os.path.exists(documents_file_path):
            raise FileNotFoundError(f"Файл {documents_file_path} не найден.")

        with open(qa_file_path, 'r', encoding='utf-8') as file:
            self.qa_dataset = json.load(file)

        with open(documents_file_path, 'r', encoding='utf-8') as file:
            self.documents_dataset = json.load(file)

    def get_documents(self, use_llama_index_type: bool=False) -> list:
        documents = [doc['page_content'] for doc in self.documents_dataset]
        if use_llama_index_type:
            documents = [Document(text=doc) for doc in documents]
        return documents[:1]
    
    def get_number_of_queries(self) -> int:
        return len(self.qa_dataset)

    def get_samples(self):
        for entry in self.qa_dataset:
            yield entry["inputs"]["text"], entry["outputs"]


qa_path = "qa.json"
documents_path = "documents.json"
amogus_reader = RAGDataset(qa_path, documents_path)

text = amogus_reader.get_documents()
queries = amogus_reader.get_samples()

# Getting documnets from folders with .txt files
#text = read_text_from_files('examples/data/ru/')

#print(text)

# Initialize a chunker
#chunker = SemanticTextChunker(
chunker = SimpleChunker(
#    reranker_name="/path/to/reranker_model",
#    model_name="DeepPavlov/rubert-base-cased",
    max_chunk_size=512,
    overlap=128
)

# Initialize a triplet extractor 
#artifact_extractor = TripletLLM(
#    validate=False,
    # Also you can set your own entity types as a list (and others arguments)
    # entity_list_type=[your entity types],
    # batch_size=16
#)
# Initialize a graph builder pipeline

key = "AQVN2EAPcvGE4WvehCSb1kl29NUbsnF9rjkv5Vw5"
client = RemoteLLM("gpt://b1go6qinn0muj8gb8k4o/yandexgpt/latest", "https://llm.api.cloud.yandex.net/v1", key)
#print(client.generate("Привет, Дроид! Мне нужна твоя помощь, чтобы узнать больше о Силе. Как я могу научиться ее использовать?", "Ты ассистент дроид, способный помочь в галактических приключениях."))

#import sys
#sys.exit(0)

graph_builder = KnowledgeGraphBuilder(
    client,
    triplet_extractor=TripletSMs(),
    chunker=chunker
)

# Run building 
knowledge_graph = graph_builder.build(text)

# Save results
knowledge_graph.save_graph("graph.gml").save_community_summary("summary.json")


# Indexing Graph
embedder = STEmbedder("DeepPavlov/rubert-base-cased", trust_remote_code=True)
index = Index(embedder=embedder)
index.make_index(knowledge_graph)


# Quering the Graph
local_search = LocalSearchEngine(client,
knowledge_graph, embedder, index)

#<<<<<<< HEAD
print("TTTTTTTTTTTTT")
print(local_search.query("Как звали детей последнего императора Российской Империи?"))
#=======
i = 0
for q in queries:
    print(local_search.query(q))
    i += 1
    if i > 5:
        break
#>>>>>>> dc28e992696f0ef0edec7388b68e62efed7f1427
