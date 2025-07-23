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

# Getting documnets from folders with .txt files
text = read_text_from_files('examples/data/ru/')

print(text)

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
DEEPSEEK_API_KEY = ''
# Initialize a graph builder pipeline
graph_builder = KnowledgeGraphBuilder(
    RemoteLLM("deepseek/deepseek-chat:free", "https://openrouter.ai/api/v1", DEEPSEEK_API_KEY),
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
local_search = LocalSearchEngine(RemoteLLM("deepseek/deepseek-chat:free", "https://openrouter.ai/api/v1", DEEPSEEK_API_KEY),
knowledge_graph, embedder, index)

print(local_search.query("Как звали детей последнего императора Российской Империи?"))
