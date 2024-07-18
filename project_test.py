from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
import dotenv
dotenv.load_dotenv()

sep = 100 * "-"

# LLM Model -------------------------------------------------------------------
from langchain_openai import ChatOpenAI
llm_gpt4 = ChatOpenAI(model='gpt-4o')

# TOOLS -----------------------------------------------------------------------
def remove_title_from_all_lists(title):
    """Make sure a title is in none of the lists"""
    global watch_dict
    for titles in watch_dict.values():
        if title in titles:
            titles.remove(title)

@tool
def add_title_to_list_of_movies_I_have_already_watched(title: str) -> None:
    """Add a movie title to the list of movies I have already watched in the past"""
    global watch_dict
    if title not in watch_dict["watched"]:
        remove_title_from_all_lists(title)
        watch_dict["watched"].append(title)

@tool
def add_title_to_list_of_movies_I_want_to_watch_later(title: str) -> None:
    """Add a movie title to the list of movies I want to watch later"""
    global watch_dict
    if title not in watch_dict["must_see"]:
        remove_title_from_all_lists(title)
        watch_dict["must_see"].append(title)

@tool
def add_title_to_list_of_movies_I_never_want_to_watch(title: str) -> None:
    """Add a movie title to the list of movies I never want to watch"""
    global watch_dict
    if title not in watch_dict["not_interested"]:
        remove_title_from_all_lists(title)
        watch_dict["not_interested"].append(title)

@tool
def remove_title_from_list_of_movies_I_have_already_watched(title: str) -> None:
    """Remove a movie title from the list of movies I have already watched"""
    global watch_dict
    if title in watch_dict["watched"]:
        watch_dict["watched"].remove(title)

@tool
def remove_title_from_list_of_movies_I_want_to_watch_later(title: str) -> None:
    """Remove a movie title from the list of movies I want to see"""
    global watch_dict
    if title in watch_dict["must_see"]:
        watch_dict["must_see"].remove(title)

@tool
def add_title_from_list_of_movies_I_never_want_to_watch(title: str) -> None:
    """Remove a movie title from the list of movies I'm not interested in"""
    global watch_dict
    if title in watch_dict["not_interested"]:
        watch_dict["not_interested"].remove(title)


# List of tools
tools = [
    add_title_to_list_of_movies_I_have_already_watched,
    add_title_to_list_of_movies_I_want_to_watch_later,
    add_title_to_list_of_movies_I_never_want_to_watch,
    remove_title_from_list_of_movies_I_have_already_watched,
    remove_title_from_list_of_movies_I_want_to_watch_later,
    add_title_from_list_of_movies_I_never_want_to_watch
]

# link our model to the tools
gpt4_with_tools = llm_gpt4.bind_tools(tools)

# PROMPT ----------------------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "You are a helpful movie advisor. You give me recommendations and you update my history based on my answers:"
            "- Movies I already watched in the past"
            "- Movies I want to watch later"
            "- Movies I never want to watch"
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# AGENT -----------------------------------------------------------------------
agent = create_tool_calling_agent(llm_gpt4, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# MAIN ------------------------------------------------------------------------
if __name__ == "__main__":

    watch_dict = {"watched": [], "must_see": [], "not_interested": []}
    messages = []

    while True:

        print("WATCH STATUS:", watch_dict)

        input_text = input("Alex: ")
        if not input_text:
            break

        messages.append(HumanMessage(content=f"{input_text}. Watch History: {watch_dict}"))

        result = agent_executor.invoke(
            {
                "input": input_text,
                "chat_history": messages,
            }
        )

        messages.append(AIMessage(content=result["output"]))
        print("Agent: ", result["output"])
        print(sep)
