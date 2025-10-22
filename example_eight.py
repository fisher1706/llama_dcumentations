import warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore


load_dotenv()


hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llm_querying = Groq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.environ["GROQ_API_KEY"]
)

text_splitter = SentenceSplitter(
    separator=" ",
    chunk_size=1024,
    chunk_overlap=128
)

title_extractor = TitleExtractor(
    llm=llm_querying,
    nodes=5
)

qa_extractor = QuestionsAnsweredExtractor(
    llm=llm_querying,
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


# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("lawGPT")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=hf_embeddings
)


# You can also load from documents and apply transformations in place
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, transformations=[]
# )

# Or you can initialize your index from your vector store and then add the nodes
# index = VectorStoreIndex.from_vector_store(
#     vector_store=vector_store, embed_model=hf_embeddings
# )
# index.insert_nodes(nodes)


# create a query engine and query
query_engine = index.as_query_engine(llm=llm_querying)

response = query_engine.query("What is this model good at")






if __name__ == '__main__':
    print(response)
