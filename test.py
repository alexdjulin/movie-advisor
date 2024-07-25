import os
from langchain.agents import AgentExecutor, create_tool_calling_agent, load_tools
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# import logging

# # Enable logging at the DEBUG level
# logging.basicConfig(level=logging.DEBUG)

# # Enable logging for the requests library
# logging.getLogger("requests").setLevel(logging.DEBUG)
# logging.getLogger("urllib3").setLevel(logging.DEBUG)

# load api-keys from environment variables or specify them below
openai_api_key = os.getenv("OPENAI_API_KEY")
tmdb_bearer_token = os.getenv("TMDB_BEARER_TOKEN")

# define the language model
llm = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)

# load the tmdb-api tool
tools = load_tools(["tmdb-api"], llm=llm, tmdb_bearer_token=tmdb_bearer_token)

# define the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie advisor, you answer questions about movies."),
        ("system", "If you don't know an answer, invoke the TMDB-API with a question in natural language."),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# test the agent
result = agent_executor.invoke(
    {
        "query": "Give me the plot of movie 'Inception'?",
    }
)

# print the result
print("Agent: ", result["output"])
