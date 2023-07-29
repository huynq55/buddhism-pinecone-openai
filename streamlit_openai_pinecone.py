import streamlit as st
import openai
import pinecone

# Pinecone and OpenAI setup
pinecone_api_key = "23c095af-b6e1-453a-98e4-b09a2804ba46"
pinecone_env = "us-central1-gcp"
openai_api_key = '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'
openai.api_key = openai_api_key

st.title('OpenAI-Pinecone Search App')

# Input query
search_query = st.text_input("Enter your search query")

# Perform search on button press
if st.button('Search'):
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

        # Iterate over matches and output to Streamlit
        for match in search_results['matches']:
            sutta_url = base_url.format(match['metadata']['sutta'].lower())
            st.write(f"{match['score']:.2f}: {sutta_url}")
