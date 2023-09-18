import streamlit as st

pinecone_api_key = st.secrets["API_KEYS"]["pinecone"]

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")

openai.api_key = st.secrets["API_KEYS"]["openai"]

def randomize_array(arr):
    sampled_arr = []
    while arr:
        elem = random.choice(arr)
        sampled_arr.append(elem)
        arr.remove(elem)
    return sampled_arr

st.set_page_config(page_title="GPTflix", page_icon="📖", layout="wide")

st.header("Goodreads LLM is like chatGPT for your favorite books!📖\n")

index_name = "1kbooks"
dimension = 1536

pineconeindex = pinecone.Index(index_name)

COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,  
    "max_tokens": 400,
    "model": COMPLETIONS_MODEL,
}

with st.sidebar:
    st.markdown("# About 🙌")
    st.markdown(
        "Goodreads LLM allows you to talk to version of chatGPT \n"
        "that has access to reviews of about 1000 books! 🎬 \n"
        )
    st.markdown(
        "Unline chatGPT, Goodreads LLM tries not to make stuff up\n"
        "and will only answer from injected knowlege 👩‍🏫 \n"
    )
    st.markdown("---")
    st.markdown("A side project by KSB")

    st.markdown("---")
    st.markdown("---")



