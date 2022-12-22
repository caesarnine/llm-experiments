import streamlit as st
from recipe_parser import parse_recipe

st.title("Parsing Recipes with GPT3")
recipe = st.text_area("Paste your recipe text below:", height=200)

if recipe:
    with st.spinner("Parsing recipe..."):
        result = parse_recipe(recipe)

    result  # type: ignore
