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

client = openai.OpenAI()

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

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
pinecone_index_name = 'buddhism-sutta'
pinecone_index = pinecone.Index(pinecone_index_name)

st.title('OpenAI-Pinecone Search App')

with st.form(key='search_form'):
    search_query = st.text_input("Enter your search query")
    submit_button = st.form_submit_button('Search')

word_extractor = re.compile(r'\w+')

if submit_button and search_query:
    query_embedding = client.embeddings.create(input = [search_query], model="text-embedding-ada-002").data[0].embedding
    sparse_vector = get_sparse_vector(search_query)
    search_results = pinecone_index.query(vector=query_embedding, sparse_vector=sparse_vector, top_k=10, include_metadata=True)

    base_url = "https://suttacentral.net/{}/en/sujato"
    query_words = set(word_extractor.findall(search_query.lower()))

    for match in search_results['matches']:
        #sutta_url = base_url.format(match['metadata']['sutta'].lower())
        #text = match['metadata']['text']
        #text = match['text']
        print(match['metadata']['_node_content'])
        #highlighted_text = re.sub(r'(\b' + r'\b|\b'.join(query_words) + r'\b)', r'**\1**', text, flags=re.IGNORECASE)

        #st.write(f"{match['score']:.2f}: {sutta_url}")
        #st.markdown(highlighted_text)
elif submit_button and not search_query:
    st.write('Please enter a search query')
