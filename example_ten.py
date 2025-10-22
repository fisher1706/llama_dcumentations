import warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os
from pprint import pprint
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage


load_dotenv()



llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)


response = llm.stream_complete(prompt="Tell me a joke")


messages = [
    ChatMessage(role="system", content="You are Sherlock Holmes"),
    ChatMessage(role="user", content="Who framed roger rabbit?"),
]

response_mess = llm.stream_chat(messages)


class Song(BaseModel):
    """Data model for a song."""
    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""
    name: str
    artist: str
    songs: List[Song]


sllm = llm.as_structured_llm(output_cls=Album)
input_msg = ChatMessage.from_str("Generate an example album from The Shining")

response_structured = sllm.chat([input_msg])



if __name__ == '__main__':
    print(response)
    print("*" * 200)

    for token in response:
        print(token.text, end='', flush=True)
    print("*" * 200)

    print(response_mess)
    print("*" * 200)

    print(response_structured)
    print("*" * 200)

    pprint(response_structured.raw.__dict__)


