#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import dotenv
# langchain
from langchain_core.tools import tool
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
# xata
from xata.client import XataClient
from langchain_community.vectorstores.xata import XataVectorStore

dotenv.load_dotenv()

TABLE_NAME = "movie-history"
SEP = 50 * "-"

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
        print(f"Error creating or setting schema for table '{TABLE_NAME}': {e}")
        return

    print(f"Table '{TABLE_NAME}' created successfully.")


def get_table_records() -> list:
    """Query all table records.

    Returns:
        list: list of record dicts
    """
    records = xata.data().query(TABLE_NAME)["records"]
    return records


def print_table() -> None:
    """Print all records from the table."""

    records = get_table_records()

    if not records:
        print("Table is empty.")
        return

    for rec in records:
        print(SEP)
        print(f"id: {rec["id"]}")
        print(f"title: {rec["title"]}")
        print(f"status: {rec['status']}")
        print(f"comment: {rec['comment']}")
        print(f"content: {rec['content']}")

    print(SEP)


def add_update_movie(movie_record: dict) -> None:
    """Add a new record to the table. If it exists, it will be deleted first
    and replaced by the new version.

    Args:
        movie_record (dict): new or updated record dict
    """

    table_records = get_table_records()

    # delete record if it exists
    for rec in table_records:
        if rec["title"] == movie_record["title"]:
            xata.records().delete(TABLE_NAME, rec["id"])
            break

    # add new or updated record
    doc = Document(page_content=movie_record["content"], metadata={k: v for k, v in movie_record.items() if k != "content"})
    vector_store.add_documents([doc])


def delete_movie_from_table(title: str) -> None:
    """Delete a record from the table.

    Args:
        title (str): movie title
    """
    records = get_table_records()
    for rec in records:
        if rec["title"] == title:
            xata.records().delete(TABLE_NAME, rec["id"])
            print(f"Record '{title}' deleted successfully")
            return
    print(f"Record '{title}' not found.")


def get_watch_lists() -> dict:
    """Create a dictionary with the watched, must see and not interested movies
    from the table records.

    Returns:
        dict: dictionary with the three lists
    """

    table_records = get_table_records()

    watched = [rec['title'] for rec in table_records if rec["status"] == "watched"]
    must_see = [rec['title'] for rec in table_records if rec["status"] == "must_see"]
    not_interested = [rec['title'] for rec in table_records if rec["status"] == "not_interested"]

    return {"watched": watched, "must_see": must_see, "not_interested": not_interested}

@tool
def add_title_to_movies_I_have_already_watched(title: str, comment: str) -> None:
    """Add a movie title to the list of movies I have already watched in the past.

    Args:
        title (str): movie title
        comment (str): my personal comment about the movie, sumed up in a few key words
    """

    record = {
        "title": title,
        "status": "watched",
        "comment": comment,
        "content": f"{title} (watched) {comment}"
    }
    add_update_movie(record)


@tool
def add_title_to_movies_I_have_never_watched_but_want_to(title: str, comment: str) -> None:
    """Add a movie title to the list of movies I have never watched but I want to watch later.

    Args:
        title (str): movie title
        comment (str): my personal comment about the movie, sumed up in a few key words
    """

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
        comment (str): my personal comment about the movie, sumed up in a few key words
    """

    record = {
        "title": title,
        "status": "not_interested",
        "comment": comment,
        "content": f"{title} (not_interested) {comment}"
    }
    add_update_movie(record)


@tool
def remove_title_from_lists_of_movies(title: str) -> None:
    """Remove a movie title from my list of movies

    Args:
        title (str): movie title to delete
    """
    delete_movie_from_table(title)


# List of tools
agent_tools = [
    add_title_to_movies_I_have_already_watched,
    add_title_to_movies_I_have_never_watched_but_want_to,
    add_title_to_movies_I_have_never_watched_and_dont_want_to,
    remove_title_from_lists_of_movies,
]
