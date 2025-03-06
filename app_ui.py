import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Open Intelligence",
    page_icon="🌍",
    layout="wide",
)

import agent_module
import asyncio
import nest_asyncio
import os
import time
from pathlib import Path
import streamlit.components.v1 as components

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "pdfs" not in st.session_state:
        st.session_state.pdfs = []
    if "csvs" not in st.session_state:
        st.session_state.csvs = []

def display_chat_history():
    """Display the chat history from session state"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if "markdown_content" in message:
                    st.markdown(message["markdown_content"])
                if "html_path" in message and message["html_path"]:
                    html_path = message["html_path"]
                    if os.path.exists(html_path):
                        with st.expander("View Map", expanded=True):
                            html_file = Path(html_path).read_text(encoding="utf-8")
                            components.html(html_file, height=500)
                    else:
                        st.warning(f"HTML file not found: {html_path}")

def process_user_query(user_query):
    """Process the user query using the agent module and update the chat history"""
    if not user_query.strip():
        return
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Set processing flag
    st.session_state.processing = True
    
    # Add a placeholder for the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Call the agent module
            response = agent_module.run_agent(user_query)
            
            # Extract fields from the response
            markdown_report = response.markdown_report
            html_path = response.html_path
            pdf_path = response.pdf_path
            code_string = response.code_string

            # Create a container for both markdown and map
            with message_placeholder.container():
                # Display the markdown response
                st.markdown(markdown_report)
                
                # Display the map if html_path exists
                if html_path and os.path.exists(html_path):
                    with st.expander("View Map", expanded=True):
                        html_file = Path(html_path).read_text(encoding="utf-8")
                        components.html(html_file, height=500)
            
            # Add the assistant's response to the chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Response generated", 
                "markdown_content": markdown_report,
                "html_path": html_path
            })
            
            st.session_state.pdfs.append(pdf_path)
            st.session_state.csvs.append(code_string)
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_message,
                "markdown_content": error_message
            })
        
        # Clear processing flag
        st.session_state.processing = False

def main():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    nest_asyncio.apply()
    
    # Custom CSS for a more professional look
    st.markdown("""
    <style>
    .header-container {
        position: fixed;
        top: 50px;
        width: calc(100% - 350px); /* Account for sidebar width */
        max-width: 1000px;
        background-color: white;
        padding: 0rem;
        z-index: 100;
        border-bottom: 1px solid #eee;
        margin-left: 10px; /* Add some spacing from the sidebar */
    }
    .content-container {
        margin-top: 150px;  /* Adjust based on header height */
    }
    .stChatMessage {
        background-color: #f7f7f7;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2c3e50;
    }
    .main-header {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: left;
    }
    .sub-header {
        color: #34495e;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 0.5rem;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

    # App header with floating container
    st.markdown("""
        <div class="header-container">
            <div class="main-header">Open Intelligence</div>
            <div class="sub-header">Explore and analyze global events through interactive conversations</div>
        </div>
        <div class="content-container"></div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Display the chat interface
    display_chat_history()
    
    # Display the pdfs
    with st.sidebar:
        pdf_file_name = st.selectbox(label="Select a PDF", options=st.session_state.pdfs)
        if pdf_file_name:
            pdf_path = Path(pdf_file_name)
            st.download_button(label="Download PDF", data=pdf_path, file_name=pdf_path)
    
    with st.sidebar:
        csv_file_name = st.selectbox(label="Select a CSV", options=st.session_state.csvs)
        if csv_file_name:
            csv_path = Path(csv_file_name)
            st.download_button(label="Download CSV", data=csv_path, file_name=csv_path)
    
    # Chat input
    if not st.session_state.processing:
        user_query = st.chat_input("Ask a question about global events...", key="chat_input")
        if user_query:
            process_user_query(user_query)
    else:
        # Disable chat input during processing
        st.chat_input("Processing your request...", key="chat_input_disabled", disabled=True)


if __name__ == "__main__":
    main()








