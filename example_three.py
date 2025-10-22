import warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from llama_parse import LlamaParse
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex


load_dotenv()  # Loads OPENAI_API_KEY


if __name__ == '__main__':
    # noinspection PyTypeChecker
    documents = LlamaParse(result_type='text').load_data("data_one/constitution.pdf")
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the first article?")
    print(f"response: {response}")
