import warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from llama_index.core import Document
from llama_index.core.schema import MetadataMode


document = Document(
    text="This is a super-customized document",
    metadata={
        "file_name": "super_secret_document.txt",
        "category": "finance",
        "author": "LlamaIndex",
    },
    # excluded_embed_metadata_keys=["file_name"],
    excluded_llm_metadata_keys=["category"],
    metadata_seperator="\n",
    metadata_template="{key}:{value}",
    text_template="Metadata:\n{metadata_str}\n-----\nContent:\n{content}",
)


if __name__ == '__main__':
    print("The LLM sees ths: \n", document.get_content(metadata_mode=MetadataMode.LLM))
    print("*" * 200)

    print("The embedder sees this: \n", document.get_content(metadata_mode=MetadataMode.EMBED))



