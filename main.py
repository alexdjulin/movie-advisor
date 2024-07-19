#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tools import *
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
LOG = logging.getLogger("movie_history")
# openai
embeddings = OpenAIEmbeddings()

sep = 50 * "-"


def main():

    # init table
    init_table()

    # create openai model and link it to tools
    llm_gpt4 = ChatOpenAI(model='gpt-4o-mini')

    # create prompt
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

    # create langchain agent
    agent = create_tool_calling_agent(llm_gpt4, agent_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=False)

    messages = []

    # chat loop
    while True:

        watch_lists = get_watch_lists()
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
