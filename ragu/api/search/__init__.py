"""
What happens to a search between the engine and the response.

``mapping`` renders engine results in the wire schema, ``reranking`` keeps a
failing reranker from failing the request, and ``usage`` accounts for what the
request spent. None of it imports FastAPI: the routes call in, never the
reverse.
"""
