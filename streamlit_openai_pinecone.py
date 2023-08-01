import streamlit as st
import openai
import pinecone
import torch
import re

from transformers import AutoTokenizer
from splade.models.transformer_rep import Splade

sparse_model_id = 'naver/splade-cocondenser-ensembledistil'
tokenizer = AutoTokenizer.from_pretrained(sparse_model_id)
sparse_model = Splade(sparse_model_id, agg='max')
sparse_model.eval()

def get_sparse_vector(text):
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        sparse_emb = sparse_model(d_kwargs=tokens.to('cpu'))['d_rep'].squeeze()
        indices = sparse_emb.nonzero().squeeze().cpu().tolist()
        values = sparse_emb[indices].cpu().tolist()
        return {'indices': indices, 'values': values}

pinecone_api_key = st.secrets['PINECONE_API_KEY']
pinecone_env = "us-central1-gcp"
openai_api_key = st.secrets['OPENAI_API_KEY']
openai.api_key = openai_api_key

st.title('OpenAI-Pinecone Search App')

with st.form(key='search_form'):
    search_query = st.text_input("Enter your search query")
    submit_button = st.form_submit_button('Search')

if submit_button:
    if not search_query:
        st.write('Please enter a search query')
    else:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        pinecone_index_name = 'buddhism-sutta'
        pinecone_index = pinecone.Index(pinecone_index_name)
        query_embedding = openai.Embedding.create(input=[search_query], model="text-embedding-ada-002")['data'][0]['embedding']
        sparse_vector = get_sparse_vector(search_query)
        search_results = pinecone_index.query(vector=query_embedding, sparse_vector=sparse_vector, top_k=10, include_metadata=True)

        base_url = "https://suttacentral.net/{}/en/sujato"

        query_words = set(re.findall(r'\w+', search_query.lower())) # Extract words from query

        for match in search_results['matches']:
            sutta_url = base_url.format(match['metadata']['sutta'].lower())

            # Extract text from metadata
            text = match['metadata']['text']

            # Highlight matching words, preserving original formatting
            highlighted_text = re.sub(r'(\b' + r'\b|\b'.join(query_words) + r'\b)', r'**\1**', text, flags=re.IGNORECASE)

            st.write(f"{match['score']:.2f}: {sutta_url}")
            st.markdown(highlighted_text)  # Streamlit uses markdown
