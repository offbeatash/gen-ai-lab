"""
Simple langchain Q&A chatbot using Streamlit and Groq.
"""

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import os

##Page config 
st.set_page_config(page_title="Simple LangChain Q&A Chatbot with Groq", page_icon="🏋️‍♂️", layout="wide")


#title
st.title("Simple LangChain Chat with Groq")
st.markdown("Learn Langchain basics with Groq's ultra-fast inference!")

with st.sidebar:
    st.header("Settings")

    #Api key
    api_key = st.text_input("Enter your Groq API Key", type="password", help="You can get your free API key from: console.groq.com/")

    #model selection
    model_name = st.selectbox(
        "Select a model",
        ["openai/gpt-oss-20b", "meta-llama/llama-prompt-guard-2-22m"],
        help="Select a model to use for the chatbot.",
        index=0)

    #clear button
    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        st.rerun()

#initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

#initialize llm
@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None

    #initialize the groq model
    llm = ChatGroq(
        groq_api_key=api_key,
        model=model_name,
        temperature=0.7,
        streaming=True,)

    #create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the questions clearly and concisely and truthfully."),
        ("user", "{question}"),
    ])

    #create chain
    chain = prompt | llm | StrOutputParser()

    return chain

#get chain
chain = get_chain(api_key, model_name)

if not chain:
    st.warning("Please enter your Groq API Key to start chatting.")
    st.markdown("[Get your free API key here](https://console.groq.com/)")

else:
    #display chat messages

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    #chat input
    if question:= st.chat_input("Ask me anything"):
        #add user message to session state

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

    #generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            #stream response from the groq
            for chunk in chain.stream({"question": question}):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            #add to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")


#example questions

st.markdown("---")
st.markdown("Try asking: ")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- How does Groq work?")
with col2:
    st.markdown("- What is the difference between Groq and other AI accelerators?")
    st.markdown("- How can I use Groq with LangChain?")

#footer
st.markdown("---")
st.markdown("Made with ❤️ by Ashish, using Groq and LangChain")