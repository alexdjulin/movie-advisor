# misc
import os
# langchain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_community import SpeechToTextLoader

# load environment variables
import dotenv
dotenv.load_dotenv()

project_id = "ai-chitchat"
file_path = "prototypes/audio.wav"

llm_gpt4 = ChatOpenAI(model='gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))

# create prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ("You are a helpful assistant answering questions.")),
        ("ai", "Hello! I am your helpful assistant. How can I help you today?"),
        ("human", "Give me a good French comedy."),
        ("ai", "Sure! How about 'Amélie'?"),
        ("human", "Do you know the height of the Eiffel Tower?"),
        ("ai", "Sure, the Eiffel Tower is 324 meters tall."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

messages = []

# create speech loader
def process_speech():
    project_id = "ai-chitchat"
    file_path = "prototypes/audio.wav"
    loader = SpeechToTextLoader(project_id=project_id, file_path=file_path)
    return loader.load()[0].page_content

str_output_parser = StrOutputParser()

# create chain
chain = prompt | llm_gpt4 | str_output_parser

while True:
    # define input
    # question = input("Me: ")
    # if question == "":
        # break

    # messages.append(HumanMessage(content=question))

    # invoke answer
    answer = chain.invoke(
        {
            "input": process_speech()
        }
    )

    # messages.append(AIMessage(content=answer))
    print("AI:", answer)
