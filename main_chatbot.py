#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: main_chatbot.py
Description: Run the chatbot using the ai_chatbot module.
Author: @alexdjulin
Date: 2024-07-25
"""

# add chatbot module to path
import os
import sys
ai_chatbot_path = os.path.join(os.path.dirname(__file__), 'ai_chatbot')
# sys.path.insert(0, ai_chatbot_path) # load ai_chatbot modules first in case of name duplicates
sys.path.append(ai_chatbot_path) # load ai_chatbot modules last

# environment variables
import dotenv
dotenv.load_dotenv()

# parse command line arguments
import argparse
parser = argparse.ArgumentParser(description='Chat with an AI chatbot.')
parser.add_argument('--config', '-c', type=str, default='ai_chatbot/config.yaml', help='Path to configuration file.')
parser.add_argument('--input', '-i', type=str, help='Overrides input method to use: {text, voice, voice_k}.')
parser.add_argument('--language', '-l', type=str, help='Overrides chat language (Example: en-US, fr-FR, de-DE). A matching voice should be defined in edgetts_voice, in the config file.')
args = parser.parse_args()


if __name__ == '__main__':

    # parse arguments
    config_file = args.config
    input_method = args.input
    language = args.language

    # load config file
    from config_loader import load_config
    load_config(config_file)

    # initialize xata table
    import tools
    tools.init_table()

    # create avatar instance and start chat
    from ai_chatbot import AiChatbot
    avatar = AiChatbot()

    # create worker agent
    avatar.create_worker_agent()
    # avatar.create_worker_agent(placeholders=["movie_history"])

    # start chat
    avatar.chat_with_avatar(input_method, language)
