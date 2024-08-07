#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tools.py
Description: Tool methods that the langchain agent can use to retrieve informatoin the LLM does not know about.
Author: @alexdjulin
Date: 2024-07-25
"""

import os
from pathlib import Path
import requests
# langchain
from langchain_core.tools import tool
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
# xata
from xata.client import XataClient
from langchain_community.vectorstores.xata import XataVectorStore
# logger: import from ai_chatbot subrepo or create a default one if not available
try:
    from logger import get_logger
    LOG = get_logger(Path(__file__).stem)
except ImportError:
    import logging
    logging.basicConfig(
        level=10,  # debug
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='movie_advisor.log',
        force=True
    )
    LOG = logging.getLogger(Path(__file__).stem)

TABLE_NAME = "movie_history"

xata = XataClient()

vector_store = XataVectorStore(
    embedding=OpenAIEmbeddings(),
    api_key=os.getenv("XATA_API_KEY"),
    db_url=os.getenv("XATA_DATABASE_URL"),
    table_name=TABLE_NAME,
)


def init_table() -> None:
    """Initialise xata table and vectorstore for movie history."""

    # try accessing the table
    try:
        assert xata.data().query(TABLE_NAME).is_success()
    except AssertionError:
        print(f"Table '{TABLE_NAME}' does not exist. Creating table.")
        create_table()


def create_table() -> None:
    """Create xata table to store our movie history."""

    table_schema = {
        "columns": [
            {"name": "title", "type": "text"},
            {"name": "status", "type": "text"},
            {"name": "comment", "type": "text"},
            {"name": "content", "type": "text"},
            {"name": "embedding", "type": "vector", "vector": {"dimension": 1536}}
        ]
    }

    try:
        # create table
        assert xata.table().create(TABLE_NAME).is_success()
        # set schema
        resp = xata.table().set_schema(TABLE_NAME, table_schema)
        assert resp.is_success(), resp

    except AssertionError as e:
        LOG.error(f"Error creating or setting schema for table '{TABLE_NAME}': {e}")
        return

    LOG.info(f"Table '{TABLE_NAME}' created successfully.")


def query_table(columns: list[str] = None, filter: dict = None) -> list[dict]:
    """Query all table records.

    Args:
        columns (list): list of columns to retrieve
        filter (dict): filter to apply to the query

    Returns:
        list: list of record dicts
    """

    payload = {}
    payload["page"] = {"size": 1000} # limiting it to 1000 results
    if columns:
        payload["columns"] = columns
    if filter:
        payload["filter"] = filter

    records = xata.data().query(
        table_name=TABLE_NAME,
        payload=payload,
    )

    LOG.debug(f"Retrieved {len(records)} records from table.")

    return records["records"]


def add_update_movie(movie_record: dict) -> None:
    """Add a new record to the table. If it exists, it will be deleted first
    and replaced by the new version.

    Args:
        movie_record (dict): new or updated record dict
    """

    # query table titles
    table_records = query_table(columns=["title"])

    # delete record if it exists
    for rec in table_records:
        if rec["title"] == movie_record["title"]:
            xata.records().delete(TABLE_NAME, rec["id"])
            break

    # add new or updated record
    doc = Document(page_content=movie_record["content"], metadata={k: v for k, v in movie_record.items() if k != "content"})
    vector_store.add_documents([doc])

    LOG.debug(f"Added record '{movie_record['title']}' to table.")


def get_watch_lists() -> str:
    """Create a dictionary with watchlists content from the table records.

    Returns:
        str: string listing the contents of the four lists
    """

    watched_liked = [rec['title'] for rec in query_table(columns=["title"], filter={"status": "watched_liked"})]
    watched_disliked = [rec['title'] for rec in query_table(columns=["title"], filter={"status": "watched_disliked"})]
    must_see = [rec['title'] for rec in query_table(columns=["title"], filter={"status": "must_see"})]
    not_interested = [rec['title'] for rec in query_table(columns=["title"], filter={"status": "not_interested"})]

    watch_lists = 'Here is my movie history. Do not recommend any of these:\n'
    watch_lists += f"1. Watched and liked ({len(watched_liked)}): " + ", ".join(watched_liked) + "\n"
    watch_lists += f"2. Watched and disliked ({len(watched_disliked)}): " + ", ".join(watched_disliked) + "\n"
    watch_lists += f"3. Must see ({len(must_see)}): " + ", ".join(must_see) + "\n"
    watch_lists += f"4. Not interested ({len(not_interested)}): " + ", ".join(not_interested) + "\n"

    LOG.debug(f"Retrieved watch lists: {watch_lists}")

    return watch_lists


@tool
def add_title_to_movies_I_watched_and_liked(title: str, comment: str) -> None:
    """Add a movie title to the list of movies I have already watched and liked.

    Args:
        title (str): original movie title
        comment (str): my personal comment about the movie. Translate it in English if necessary.
    """

    LOG.debug("Tool call: add_title_to_movies_I_have_already_watched")

    record = {
        "title": title,
        "status": "watched_liked",
        "comment": comment,
        "content": f"{title} (watched_liked) {comment}"
    }
    add_update_movie(record)


@tool
def add_title_to_movies_I_watched_and_disliked(title: str, comment: str) -> None:
    """Add a movie title to the list of movies I have already watched but did not like.

    Args:
        title (str): original movie title
        comment (str): my personal comment about the movie. Translate it in English if necessary.
    """

    LOG.debug("Tool call: add_title_to_movies_I_have_already_watched")

    record = {
        "title": title,
        "status": "watched_disliked",
        "comment": comment,
        "content": f"{title} (watched_disliked) {comment}"
    }
    add_update_movie(record)


@tool
def add_title_to_movies_I_have_never_watched_but_want_to(title: str, comment: str) -> None:
    """Add a movie title to the list of movies I have never watched but I want to watch later.

    Args:
        title (str): movie title
        comment (str): my personal comment about the movie. Translate it in English if necessary.
    """

    LOG.debug("Tool call: add_title_to_movies_I_have_never_watched_but_want_to")

    record = {
        "title": title,
        "status": "must_see",
        "comment": comment,
        "content": f"{title} (must_see) {comment}"
    }
    add_update_movie(record)


@tool
def add_title_to_movies_I_have_never_watched_and_dont_want_to(title: str, comment: str) -> None:
    """Add a movie title to the list of movies I have never watched and I don't want to watch then ever.

    Args:
        title (str): movie title
        comment (str): my personal comment about the movie. Translate it in English if necessary.
    """

    LOG.debug("Tool call: add_title_to_movies_I_have_never_watched_and_dont_want_to")

    record = {
        "title": title,
        "status": "not_interested",
        "comment": comment,
        "content": f"{title} (not_interested) {comment}"
    }
    add_update_movie(record)


@tool
def search_movie_history_for_info_and_preferences(query: str) -> list | None:
    """Search the database for any kind of information regarding my movie history, preferences, likes and dislikes.

    Args:
        title (str): movie title to delete
    """

    LOG.debug("Tool call: search_for_personal_information_in_movie_history")
    found_docs = vector_store.similarity_search(query, k=3)

    if not found_docs:
        LOG.debug("No docs found in table.")
        return

    context = []
    for doc in found_docs:
        context.append(doc.page_content)

    return context


@tool
def query_tmdb_database_for_information_about_a_movie(query: str) -> list:
    """Query the TMDB database for a movie title and return some information about it.

    Args:
        query (str): movie title to search for

    Return:
        list: list of matching movie info dicts
    """

    LOG.debug("Tool call: query_tmdb_database_for_information_about_a_movie")

    # Base URL for the search endpoint
    url = "https://api.themoviedb.org/3/search/movie"

    # Parameters for the API request
    params = {
        'api_key': os.getenv("TMDB_BEARER_TOKEN"),
        'query': query,
        'include_adult': False,
    }

    # Making the GET request to the API
    response = requests.get(url, params=params)

    # Checking if the request was successful
    if response.status_code == 200:
        # Parsing the JSON response
        data = response.json()
        return data['results']
    else:
        # If the request failed, print the status code
        print(f"Error: {response.status_code}")
        return None


@tool
def get_all_movies_from_my_watch_lists() -> dict:
    """Get all movie titles listed in my 3 watch litsts.

    Returns:
        dict: dictionary with the three lists of titles.
    """

    LOG.debug("Tool call: get_all_movies_from_my_watch_lists")
    return get_watch_lists()


# List of tools
agent_tools = [
    add_title_to_movies_I_watched_and_liked,
    add_title_to_movies_I_watched_and_disliked,
    add_title_to_movies_I_have_never_watched_but_want_to,
    add_title_to_movies_I_have_never_watched_and_dont_want_to,
    search_movie_history_for_info_and_preferences,
    query_tmdb_database_for_information_about_a_movie,
    get_all_movies_from_my_watch_lists,
]
