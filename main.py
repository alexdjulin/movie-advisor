#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tools
# env vars
import dotenv
dotenv.load_dotenv()
# langchain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# logger
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger("main")
# openai
embeddings = OpenAIEmbeddings()

sep = 50 * "-"


def main():

    # init table
    tools.init_table()

    # create openai model and link it to tools
    llm_gpt4 = ChatOpenAI(model='gpt-4o-mini')

    # create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", (
                "You are a helpful movie advisor, you recommend me movies based on my watch history and preferences."
                "I order movies into three lists:"
                "- Movies I already watched in the past"
                "- Movies I want to watch later"
                "- Movies I never want to watch"
                "Only update these lists when I specifically ask you to do so."
                "Only recommend movies one by one."
            )),
            ("ai", "Hello! I am your movie advisor. How can I help you today?"),
            ("human", "Give me a good French comedy."),
            ("ai", "Sure! How about 'Amélie'?"),
            ("human", "I already watched that one 10 times and I love it!"),
            ("ai", "Great! I added Amélie to the list of movies you already watched."),
            ("human", "What about a good action movie?"),
            ("ai", "How about 'The Expandables'?"),
            ("human", "No, I'm not a fan of Sylvester Stallone."),
            ("ai", "Got it! I added The Expandables to the list of movies you never want to watch."),
            ("human", "What about a good thriller?"),
            ("ai", "How about 'Seven'?"),
            ("human", "Oh yes, I definitely want to watch that one!"),
            ("ai", "Great! I added Seven to the list of movies you want to watch later."),
            ("human", "Do you know if I already watched 'The Matrix'? Did I liked it?"),
            ("ai", "I checked your movie history and yes, you already watched 'The Matrix'. You said it was groundbreaking movie with strong performance innovative special effects that left a memorable experience on your teenage years."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # create langchain agent
    agent = create_tool_calling_agent(llm_gpt4, tools.agent_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools.agent_tools, verbose=True)

    messages = []

    # chat loop
    while True:

        watch_lists = tools.get_watch_lists()
        print("WATCH STATUS:", watch_lists)

        input_text = input("Alex: ")
        if not input_text:
            break

        messages.append(HumanMessage(content=f"{input_text}. Watch History: {watch_lists}"))

        result = agent_executor.invoke(
            {
                "input": input_text,
                "chat_history": messages,
            }
        )

        messages.append(AIMessage(content=result["output"]))
        print("Agent: ", result["output"])
        print(sep)


if __name__ == "__main__":
    main()
