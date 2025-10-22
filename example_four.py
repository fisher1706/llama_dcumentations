import warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import MetadataMode

from pprint import pprint





docs = SimpleDirectoryReader('data_two').load_data()
docs_name_is_id = SimpleDirectoryReader(input_dir="data_two", filename_as_id=True).load_data()



if __name__ == '__main__':
    print(len(docs))
    print("*" * 300)

    pprint(docs)
    print("*" * 300)

    pprint(docs_name_is_id)
    print("*" * 300)

    pprint(docs[0].__dict__)
    print("*" * 300)

    print(docs[0].get_content(metadata_mode=MetadataMode.EMBED))
    print("*" * 300)

    for doc in docs:
        # define the content/metadata template
        doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"

        # exclude page label from embedding
        if "page_label" not in doc.excluded_llm_metadata_keys:
            doc.excluded_llm_metadata_keys.append("page_label")

    print(docs[0].get_content(metadata_mode=MetadataMode.EMBED))



