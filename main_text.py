#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import tools
# env vars
import dotenv
dotenv.load_dotenv()
# langchain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# openai
embeddings = OpenAIEmbeddings()
sep = 50 * "-"
# terminal colors
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'


def main():

    # init table
    tools.init_table()

    # create openai model and link it to tools
    llm_gpt4 = ChatOpenAI(model='gpt-4o-mini')

    # load messages from prompt jsonl file
    prompt_filepath = "prompt_advisor.jsonl"
    prompt_messages = []

    with open(prompt_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            prompt_messages.append((json.loads(line)['role'], json.loads(line)['content']))

    # add placeholders
    prompt_messages.append(("placeholder", "{chat_history}"))
    prompt_messages.append(("human", "{input}"))
    prompt_messages.append(("placeholder", "{agent_scratchpad}"))

    prompt = ChatPromptTemplate.from_messages(prompt_messages)

    # create langchain agent
    agent = create_tool_calling_agent(llm_gpt4, tools.agent_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools.agent_tools, verbose=True)

    messages = []

    # chat loop
    while True:

        watch_lists = tools.get_watch_lists()
        print("WATCH STATUS:", watch_lists)

        input_text = input(f"{CYAN}Alex: ")
        print(RESET)
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
        print(f"{MAGENTA}Agent: ", result["output"], RESET)
        print(sep)


if __name__ == "__main__":
    main()
    print(RESET) # reset terminal colors
