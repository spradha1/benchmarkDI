import streamlit as st

# Define the pages
home = st.Page("pages/main.py", title="Home", icon="❄️")
benchmark = st.Page("pages/benchmark.py", title="Benchmark", icon="⚡")

# Set up navigation
pg = st.navigation([home, benchmark])

# Run the selected page
pg.run()