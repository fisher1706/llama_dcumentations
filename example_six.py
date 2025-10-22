import warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import os
from pprint import pprint

from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import MetadataMode

load_dotenv()  # Loads OPENAI_API_KEY


llm_transformations = Groq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=os.environ["GROQ_API_KEY"])
text_splitter = SentenceSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)
title_extractor = TitleExtractor(llm=llm_transformations, nodes=5)
qa_extractor = QuestionsAnsweredExtractor(llm=llm_transformations, questions=3)


docs = SimpleDirectoryReader('data_one').load_data()


pipeline = IngestionPipeline(transformations=[text_splitter, title_extractor, qa_extractor])



nodes = pipeline.run(
    documents=docs,
    in_place=True,
    show_progress=True,
)



if __name__ == '__main__':
    print(os.getenv("GROQ_API_KEY"))
    print("*" * 200)

    print(llm_transformations)
    print("*" * 200)

    print(len(nodes))
    print("*" * 200)

    pprint(nodes)
    print("*" * 200)

    pprint(nodes[0].__dict__)
    print("*" * 200)

    print(nodes[0].get_content(metadata_mode=MetadataMode.EMBED))
    print("*" * 200)

    print(nodes[0].get_content(metadata_mode=MetadataMode.EMBED))

