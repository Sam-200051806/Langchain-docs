from typing import Set
import streamlit as st
from backend.core import run_llm

# Page configuration
st.set_page_config(
    page_title="LangChain Documentation Helper",
    page_icon="📚",
    layout="wide"
)

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model selector
    model = st.selectbox(
        "Select Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    # Max tokens slider
    max_tokens = st.slider(
        "Max Response Length",
        min_value=100,
        max_value=4000,
        value=1000,
        step=100,
        help="Maximum number of tokens in the response"
    )
    
    # Display options
    st.subheader("Display Options")
    show_sources = st.toggle("Show Sources", value=True)
    
    # Clear conversation button
    if st.button("Clear Conversation", type="primary"):
        st.session_state["chat_answers_history"] = []
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_history"] = []
        st.rerun()
    
    # About section
    st.divider()
    st.caption("About")
    st.markdown("""
    This app uses LangChain and Retrieval Augmented Generation to answer 
    questions about LangChain documentation.
    """)

# Main content
st.header("LangChain - Documentation Helper Bot")

# Initialize session state
if (
    "chat_answers_history" not in st.session_state
    or "user_prompt_history" not in st.session_state
    or "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


# Input area
prompt = st.text_input("Ask about LangChain:", placeholder="Enter your question here...")

if prompt:
    with st.spinner("Generating response..."):
        try:
            # Pass sidebar parameters to the backend
            generated_response = run_llm(
                query=prompt, 
                chat_history=st.session_state["chat_history"],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Safely get the result/answer text
            result_text = generated_response.get("result", 
                        generated_response.get("answer", "No answer found"))
            
            # Safely extract sources with fallbacks
            sources = set()
            if "source_documents" in generated_response:
                sources = set(
                    [doc.metadata["source"] for doc in generated_response["source_documents"] 
                     if hasattr(doc, "metadata") and "source" in doc.metadata]
                )
            elif "context" in generated_response:  # Try alternative key
                sources = set(
                    [doc.metadata["source"] for doc in generated_response["context"] 
                     if hasattr(doc, "metadata") and "source" in doc.metadata]
                )
                
            # Format response with sources (if enabled)
            formatted_response = f"{result_text}"
            if sources and show_sources:
                formatted_response += f"\n\n{create_sources_string(sources)}"
            
            # Update session state
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["chat_history"].append({"type": "human", "content": prompt})
            st.session_state["chat_history"].append({"type": "ai", "content": result_text})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error("Check the backend implementation")


# Display chat history
if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)