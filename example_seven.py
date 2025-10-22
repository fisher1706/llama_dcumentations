import warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os
from pprint import pprint
from dotenv import load_dotenv

from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage


load_dotenv()  # Loads OPENAI_API_KEY

# Embeddings
"""
Only for example
"""
hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
test_embed = hf_embeddings.get_text_embedding("Hello world")


llm_transformations = Groq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.environ["GROQ_API_KEY"]
)

text_splitter = SentenceSplitter(
    separator=" ",
    chunk_size=1024,
    chunk_overlap=128
)

title_extractor = TitleExtractor(
    llm=llm_transformations,
    nodes=5
)

qa_extractor = QuestionsAnsweredExtractor(
    llm=llm_transformations,
    questions=3
)

docs = SimpleDirectoryReader('data_one').load_data()

pipeline = IngestionPipeline(transformations=[
    text_splitter,
    title_extractor,
    qa_extractor
])

nodes = pipeline.run(
    documents=docs,
    in_place=True,
    show_progress=True,
)

# create index
index = VectorStoreIndex(nodes=nodes)

llm_querying = Groq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.environ["GROQ_API_KEY"]
)

query_engine = index.as_query_engine(llm=llm_querying)
response = query_engine.query("what does this model do?")


index.storage_context.persist(persist_dir="./vectors")
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./vectors")

# load index
index_from_storage = load_index_from_storage(
    storage_context,
)
qa = index_from_storage.as_query_engine(llm=llm_querying)

response_store = qa.query("what does this model do?")





if __name__ == '__main__':
    pprint(test_embed)
    print("*" * 200)

    print(response)
    print(response.source_nodes)
    print("*" * 200)

    print(response_store)
