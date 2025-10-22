import warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os
from dotenv import load_dotenv
from pprint import pprint

from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage


load_dotenv()


llm = Groq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY")
)

response = llm.complete(prompt="Tell me a joke")


messages = [
    ChatMessage(role="system", content="You are Sherlock Holmes"),
    ChatMessage(role="user", content="Who framed roger rabbit?"),
    ChatMessage(role="assistant", content="I'm not sure, but I can help you find out."),
    ChatMessage(role="user", content="Tell me a joke"),
]

response_mess = llm.chat(messages)




if __name__ == '__main__':
    print(os.getenv("GROQ_API_KEY"))
    print("*" * 200)

    print(response)
    print("*" * 200)

    print(type(response))
    print("*" * 200)

    pprint(response.__dict__)
    print("*" * 200)

    print(response_mess)
