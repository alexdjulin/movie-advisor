# misc
import os
# langchain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# openai
embeddings = OpenAIEmbeddings()
# xata client and config
from xata.client import XataClient
xata = XataClient()
# load environment variables
import dotenv
dotenv.load_dotenv()

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

# create string output parser
str_output_parser = StrOutputParser()

# create chain
chain = prompt | llm_gpt4 | str_output_parser

while True:
    # define input
    question = input("Me: ")
    if question == "":
        break

    messages.append(HumanMessage(content=question))

    # invoke answer
    answer = chain.invoke(
        {
            "input": question,
            "chat_history": messages
        }
    )
    
    messages.append(AIMessage(content=answer))
    print("AI: ", answer)