import streamlit as st
import openai
import pinecone

# Pinecone and OpenAI setup
pinecone_api_key = st.secrets['PINECONE_API_KEY']
pinecone_env = "us-central1-gcp"
openai_api_key = st.secrets['OPENAI_API_KEY']
openai.api_key = openai_api_key

st.title('OpenAI-Pinecone Search App')

# Start form
with st.form(key='search_form'):
    # Input query
    search_query = st.text_input("Enter your search query")
    # Button for submitting the form
    submit_button = st.form_submit_button('Search')

# Perform search on form submission
if submit_button:
    # Check if query is empty
    if not search_query:
        st.write('Please enter a search query')
    else:
        # Reinitialize Pinecone and fetch index
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        pinecone_index_name = 'buddhism-sutta'
        pinecone_index = pinecone.Index(pinecone_index_name)

        # Use OpenAI to process the query
        query_embedding = openai.Embedding.create(input=[search_query], model="text-embedding-ada-002")['data'][0]['embedding']

        # Search in Pinecone
        search_results = pinecone_index.query([query_embedding], top_k=100, include_metadata=True)

        base_url = "https://suttacentral.net/{}/en/sujato"

        # Deduplicate matches by sutta
        matches_by_sutta = {}
        for match in search_results['matches']:
            sutta = match['metadata']['sutta']
            if sutta not in matches_by_sutta or match['score'] > matches_by_sutta[sutta]['score']:
                matches_by_sutta[sutta] = match

        # Iterate over deduplicated matches
        for match in matches_by_sutta.values():
            sutta_url = base_url.format(match['metadata']['sutta'].lower())
            st.write(f"{match['score']:.2f}: {sutta_url}")

            # Get the sutta vector
            sutta_vector = pinecone_index.fetch(ids=[match['id']])['vectors'][match['id']]['values']

            # Find the most similar suttas
            similar_suttas = pinecone_index.query([sutta_vector], top_k=6, include_metadata=True)

            # Skip the first result (as it will be the sutta itself)
            for similar_sutta in similar_suttas['matches'][1:]:
                similar_sutta_url = base_url.format(similar_sutta['metadata']['sutta'].lower())
                st.write(f"\tSimilar: {similar_sutta['score']:.2f}: {similar_sutta_url}")
