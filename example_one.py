import warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


load_dotenv()  # Loads OPENAI_API_KEY


if __name__ == '__main__':

    documents = SimpleDirectoryReader('data_one').load_data()
    for document in documents:
        print(f"document: {document}\n")

    index = VectorStoreIndex.from_documents(documents)
    print(f"index: {index}")

    query_engine = index.as_query_engine()
    print(f"query_engine: {query_engine}")

    response = query_engine.query("What is the first article?")
    print(f"response: {response}")
