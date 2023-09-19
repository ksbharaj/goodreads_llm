import streamlit as st
import pinecone
import tiktoken
import openai
from streamlit_chat import message
import random


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

st.set_page_config(page_title="Goodreads LLM", page_icon="📖", layout="wide")

st.header("Goodreads LLM is like chatGPT for your favorite books!📖\n")

index_name = "1kbooks"
dimension = 1536

pineconeindex = pinecone.Index(index_name)

COMPLETIONS_MODEL = "gpt-3.5-turbo-16k-0613"
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
        "that has access to reviews of about 10,000 books! 🎬 \n"
        )
    st.markdown(
        "Unline chatGPT, Goodreads LLM tries not to make stuff up\n"
        "and will try answer from injected knowlege 👩‍🏫 \n"
    )
    st.markdown("---")
    st.markdown("A side project by KSB")

    st.markdown("---")
    st.markdown("---")

def num_tokens_from_string(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_embedding(text, model):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

MAX_SECTION_LEN = 1000 #in tokens
SEPARATOR = "\n"
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))


def construct_prompt_pinecone(question):
    """
    Fetch relevant information from pinecone DB
    """
    xq = get_embedding(question , EMBEDDING_MODEL)

    #print(xq)

    res = pineconeindex.query([xq], top_k=30, include_metadata=True)

    #print(res)
    # print(most_relevant_document_sections[:2])

    chosen_sections = []    
    chosen_sections_length = 0

    for match in res['matches'][:12]:
        #print(f"{match['score']:.2f}: {match['metadata']['text']}")
        if chosen_sections_length <= MAX_SECTION_LEN:
            document_section = match['metadata']['text']

            #   document_section = str(_[0] + _[1])      
            chosen_sections.append(SEPARATOR + document_section)

            chosen_sections_length += num_tokens_from_string(str(document_section), "cl100k_base")

    for match in randomize_array(res['matches'][-18:]):
        #print(f"{match['score']:.2f}: {match['metadata']['text']}")
        if chosen_sections_length <= MAX_SECTION_LEN:
            document_section = match['metadata']['text']

            #   document_section = str(_[0] + _[1])      
            chosen_sections.append(SEPARATOR + document_section)

            chosen_sections_length += num_tokens_from_string(str(document_section), "cl100k_base")


    # Useful diagnostic information
    #print(f"Selected {len(chosen_sections)} document sections:")
    
    header = """Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say "I don't know". Also, if uncertain or if you think you are hallucinating,
    again just say "I don't know". Again, it is very important to answer as truthfully as possible.
    
    Answer in a very sarcastic tone and make it fun! Surprise the user with your answers. Try to keep your answer to less than 80 words.\n
    You are Goodreads LLM, an AI book-worm that loves reading books!\n
    Context:\n
    """ 
    return header + "".join(chosen_sections) 

def summarize_past_conversation(content):

    APPEND_COMPLETION_PARAMS = {
        "temperature": 0.0,
        "max_tokens": 400,
        "model": COMPLETIONS_MODEL,
    }

    prompt = "Summarize this discussion into a single paragraph keeping the titles of any book mentioned: \n" + content

    try:
        response = openai.Completion.create(
                    prompt=prompt,
                    **APPEND_COMPLETION_PARAMS
                )
    except Exception as e:
        print("I'm afraid your question failed! This is the error: ")
        print(e)
        return None

    choices = response.get("choices", [])
    if len(choices) > 0:
        return choices[0]["text"].strip(" \n")
    else:
        return None


COMPLETIONS_API_PARAMS = {
        "temperature": 0.0,
        "max_tokens": 500,
        "model": COMPLETIONS_MODEL,
    }

def answer_query_with_context_pinecone(query):
    prompt = construct_prompt_pinecone(query) + "\n\n Q: " + query + "\n A:"
    
    print("---------------------------------------------")
    print("prompt:")
    print(prompt)
    print("---------------------------------------------")
    try:
        response = openai.ChatCompletion.create(
                    messages=[{"role": "system", "content": "You are a helpful AI who loves books."},
                            {"role": "user", "content": str(prompt)}],
                            # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                            # {"role": "user", "content": "Where was it played?"}
                            # ]
                    **COMPLETIONS_API_PARAMS
                )
    except Exception as e:
        print("I'm afraid your question failed! This is the error: ")
        print(e)
        return None

    choices = response.get("choices", [])
    if len(choices) > 0:
        return choices[0]["message"]["content"].strip(" \n")
    else:
        return None



# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def clear_text():
    st.session_state["input"] = ""

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("Input a question here! For example: \"Is X book good?\". \n It works best if your question contains the title of a book! You might want to be really specific, like mentioning 'Hunger Games the Book' rather than just 'Hunger Games'. Also, I have no memory of previous questions! (I'm working on this though!) 😅😊","Who are you?", key="input")
    return input_text



user_input = get_text()


if user_input:
    output = answer_query_with_context_pinecone(user_input)

    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i],seed=5 , key=str(i))
        message(st.session_state['past'][i], is_user=True,avatar_style="adventurer",seed=5, key=str(i) + '_user')



