import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# Function to fetch free ebooks from Open Library API
def fetch_free_ebooks(query, count):
    url = f"https://openlibrary.org/search.json?q={query}&mode=ebooks&has_fulltext=true&limit={count}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('docs', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching free ebooks: {e}")
        return []

# Function to fetch children's storybooks from Project Gutenberg API
def fetch_childrens_books(query, count):
    url = f"http://gutendex.com/books/?topic=children&search={query}&limit={count}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching children's storybooks: {e}")
        return []

# Function to display book information
def display_books(books, source="Open Library"):
    for book in books:
        if source == "Open Library":
            title = book.get('title', 'No title')
            authors = ", ".join(book.get('author_name', []))
            subjects = ", ".join(book.get('subject', [])) if 'subject' in book else 'No subjects'
            cover_id = book.get('cover_i', None)
            image_url = f"http://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None
            edition_key = book.get('cover_edition_key') or book.get('edition_key', [None])[0]
            download_url = f"https://openlibrary.org/works/{edition_key}/borrow"
        else:  # Project Gutenberg
            title = book.get('title', 'No title')
            authors = ", ".join(author['name'] for author in book.get('authors', []))
            subjects = ", ".join(book.get('subjects', [])) if 'subjects' in book else 'No subjects'
            image_url = f"http://www.gutenberg.org/cache/epub/{book.get('id')}/pg{book.get('id')}.cover.medium.jpg"
            download_url = f"http://www.gutenberg.org/ebooks/{book.get('id')}"

        st.subheader(title)
        st.text(f"Authors: {authors}")
        st.text(f"Subjects: {subjects}")
        if image_url:
            st.image(image_url, caption=title, width=200)
        st.markdown(f"[Download '{title}']({download_url})")
        st.write("---")

# Streamlit UI
st.title("📚 Ebooks Explorer")

# General ebooks search bar and count selector
search_query = st.text_input("Search for general ebooks")
count = st.number_input("Number of books to fetch", min_value=1, max_value=100, value=10)
search_button = st.button("Search General Ebooks")

# Children's storybooks search bar and count selector
childrens_query = st.text_input("Search for children's storybooks")
childrens_count = st.number_input("Number of children's books to fetch", min_value=1, max_value=100, value=10)
childrens_search_button = st.button("Search Children's Storybooks")

# Fetch and display free ebooks based on search query
if search_button:
    st.header("Explore Free Ebooks")
    free_ebooks = fetch_free_ebooks(search_query, count)
    if free_ebooks:
        display_books(free_ebooks)
    else:
        st.write("No free ebooks available for the search query.")

# Fetch and display children's storybooks based on search query
if childrens_search_button:
    st.header("Explore Children's Storybooks")
    childrens_books = fetch_childrens_books(childrens_query, childrens_count)
    if childrens_books:
        display_books(childrens_books, source="Project Gutenberg")
    else:
        st.write("No children's storybooks available for the search query.")
