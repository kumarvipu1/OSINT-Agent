import streamlit as st
import asyncio
import nest_asyncio
import os
import time
from pathlib import Path
import pandas as pd
import streamlit.components.v1 as components
import nx_arangodb as nxadb
from arango import ArangoClient
import pandas as pd
from langchain_experimental.utilities import PythonREPL
from typing import Annotated, List, Optional, Tuple
from pydantic import BaseModel, Field
import json
import re
import dotenv
from pydantic_ai import Agent, RunContext, ModelRetry
from dataclasses import dataclass
from markdown_pdf import MarkdownPdf, Section
from pydantic_ai.models import openai
import dotenv
import os
import logfire

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Open Intelligence",
    page_icon="🌍",
    layout="wide",
)


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
    if 'credentials' not in st.session_state:
        st.session_state.credentials = {
            'openai_api': '',
            'logfire_token': '',
            'arango_url': '',
            'arango_username': '',
            'arango_password': ''
        }

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
                if 'csv_path' in message and message['csv_path']:
                    csv_path = message['csv_path']
                    if csv_path and os.path.exists(csv_path):
                        with st.expander("Inspect data"):
                            st.dataframe(pd.read_csv(csv_path))
                
                # Add PDF download button if pdf_path exists
                if 'pdf_path' in message and message['pdf_path']:
                    pdf_path = message['pdf_path']
                    if pdf_path and os.path.exists(pdf_path):
                        with open(pdf_path, 'rb') as pdf_file:
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_file,
                                file_name=Path(pdf_path).name,
                                mime="application/pdf"
                            )
                        st.caption(f"File stored at: {pdf_path}")



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



    with st.sidebar:
        st.header("API Credentials")
        with st.form("credentials_form"):
            st.session_state.credentials['openai_api'] = st.text_input("OpenAI API Key", 
                value=st.session_state.credentials['openai_api'], type="password")
            st.session_state.credentials['logfire_token'] = st.text_input("Logfire Token (optional)", 
                value=st.session_state.credentials['logfire_token'], type="password")
            st.session_state.credentials['arango_url'] = st.text_input("ArangoDB URL", 
                value=st.session_state.credentials['arango_url'])
            st.session_state.credentials['arango_username'] = st.text_input("ArangoDB Username", 
                value=st.session_state.credentials['arango_username'])
            st.session_state.credentials['arango_password'] = st.text_input("ArangoDB Password", 
                value=st.session_state.credentials['arango_password'], type="password")
            
            if st.form_submit_button("Save Credentials"):
                # Validate that all required fields are filled
                missing_fields = [field for field, value in st.session_state.credentials.items() 
                                if not value.strip()]
                
                required_fields = ['openai_api', 'arango_url', 'arango_username', 'arango_password']
                missing_required = any(field in missing_fields for field in required_fields)
                
                if missing_required:
                    st.error(f"Please fill in all required fields: {', '.join(missing_fields)}")
                else:
                    st.success("Credentials saved!")
    
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

    # Configure logfire if token is provided
    if st.session_state.credentials['logfire_token']:
        logfire.configure(token=st.session_state.credentials['logfire_token'])
    else:
        pass
    
    if st.session_state.credentials['arango_url'] and st.session_state.credentials['arango_username'] and st.session_state.credentials['arango_password']:
        # Initialize the ArangoDB client with session state credentials
        db = ArangoClient(hosts=st.session_state.credentials['arango_url']).db(
            username=st.session_state.credentials['arango_username'], 
            password=st.session_state.credentials['arango_password']
        )
        
        G_db = nxadb.Graph(name="OPEN_INTELLIGENCE", db=db)
    
    
    def load_json_file(file_path: str):
        """
        Load a JSON file and return the data as a dictionary.
        """
        with open(file_path, 'r') as file:
            return json.load(file)
        
    
    
    
    # defining dependencies
    @dataclass
    class agent_state:
        user_query: str = Field(description="The user query to be answered")
        schema: dict = Field(description="The schema of the Graph Database")
    
    # Output structure
    class agent_response(BaseModel):
        markdown_report: str = Field(description="The markdown report of the user query")
        html_path: str = Field(description="The path to the html file for the map")
        pdf_path: str = Field(description="The path to the pdf file")
        code_string: str = Field(description="The code string to be executed")
        aql_query: str = Field(description="The aql query to be executed")
        csv_path: str = Field(description="The path where csv file is stored")
        pdf_path: str = Field(description = "The path where pdf is stored")
        
    
    #Initialize agent model
    if st.session_state.credentials['openai_api']:
        model = openai.OpenAIModel('gpt-4o', api_key=st.session_state.credentials['openai_api'])
        agent = Agent(model=model, deps_type=agent_state, result_type=agent_response)
        
        if agent:
    
            @agent.system_prompt
            def get_agent_system_prompt(ctx: RunContext[agent_state]):
                
                prompt = f"""
                You are a helpful assistant that specialises in OSINT (open source intelligence) and can gather information from the graph database.
                You are familiar with arangoDB and can use the AQL query language to query the graph database. Use the schema provided to answer the user query.
                The user query is:\n {ctx.deps.user_query} \n
                The schema of the graph database is:\n {ctx.deps.schema} \n
                
                You can use the following tools to answer the user query:
                - run_aql_query: to run an AQL query on the graph database
                - get_column_names: to get the column names of the dataframe
                - validate_data_frame: to validate the dataframe
                - python_execution_tool: to execute a python code string
                - write_markdown_to_file: to write markdown content to a file
                
                Use step by step process to answer the user query. The steps involved are:
                
                1. DATA COLLECTION:
                    - understant the user query and explain what the use is asking for to the user
                    - Based on the explanation, decide what data to collect from the graph database
                    - Decide on the AQL query to be used to collect the data
                    - The query should follow this schema : 
                    
                                'date': event['date'],
                                'description': event['description'],
                                'fatalities': event['fatalities'],
                                'event_type': event['label'],
                                'actor_name': actor['name'],
                                'latitude': event['geo']['coordinates'][1],
                                'longitude': event['geo']['coordinates'][0]
                                
                    - For filters on actor use case insensitive comparison and use different variations of keywords to filter the data example : lower(actor.name) like '%usa%' or lower(actor.name) like '%united states%' or lower(actor.name) like '%america%'
                    - or region name like '%middle east%' or '%mideast%' or '%middle east%' or '%mideast%' or %<middle east realted country>%
                    - No not use special characters such as \\n or \\in AQL query,
                    - Run the AQL query
                    - Save the results to a csv file
                
                2. DATA VALIDATION:
                    - Validate the dataframe
                    - Note: all the column names are in lower case
                    - use the get_column_names tool to get the column names of the dataframe, categorise the columns into categorical, numerical and date columns.
                    - Check if the dataframe is empty. If the dataframe does not have the required columns or data types or is empty, repeat from step 1 and modify the query
                    
                
                3. DATA ANALYSIS:
                    - Analyse the contents from the data summary
                    - Plot a world map of the events based on timeline and location using python folium library based on user query, use execute_code tool to execute the code
                    - Save the map as a html file with name <filename>.html
                    
                4. Code Generation:
                    - All the necessary libraries are installed
                    - Load the csv file into pandas dataframe
                    - Use the pandas dataframe to generate the code to plot the map
                    - In the map, show the chronology of the events with flow lines from one event to another
                    - Zoom in to the general area of the events
                    - The code should be in python
                    - use the python_execution_tool to execute the code
                    
                    
                4. REPORT GENERATION:
                    - compile all the information gathered from the graph database and the webscraping and save it as a markdown report
                    - describe each event in detail in the report
                    - use proper markdown formatting to make the report look presentable
                    - start with level 1 heading with relevant title
                    - Use professional news writing style
                    - return the final output as a markdown report
                    - save the markdown file using write_markdown_to_file tool
                    
                
                General Guidelines:
                    - use the tools in a step by step manner
                    - use the tools only when necessary
                    - do not use the tools for any other purpose
                    - If error, continue from the step where the error occurred.
                    - Use of folium library to generate map visualisation is mandatory.
                    
                """
                return prompt
    
    
            # agent tools
    
            # for running aql query
            @agent.tool
            def run_aql_query(ctx: RunContext[None], query: Annotated[str, "The AQL query to be executed"], file_path: Annotated[str, "The path to the file to save the results"]):
                """
                Run an AQL query on the graph database and save the results to a file.
                """
                try:
                    # Clean the query by removing special characters and normalizing whitespace
                    cleaned_query = re.sub(r'[\n\r\t]', ' ', query)  # Replace newlines/tabs with spaces
                    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
                    
                    print(cleaned_query)  # Normalize whitespace
                    
                    results = list(G_db.query(cleaned_query))
                    print(results)
                    
                    # Handle empty results
                    if not results:
                        return "No results found for the query. Please modify the query and try again."
                    
                    # Handle nested data structure
                    if isinstance(results[0], dict) and 'event' in results[0]:
                        # Extract relevant fields from the nested structure
                        flattened_data = []
                        for item in results:
                            event = item['event']
                            actor = item['actor']
                            
                            # Create a flattened dictionary with the fields we want
                            flat_item = {
                                'date': event['date'],
                                'description': event['description'],
                                'fatalities': event['fatalities'],
                                'event_type': event['label'],
                                'actor_name': actor['name'],
                                'latitude': event['geo']['coordinates'][1],
                                'longitude': event['geo']['coordinates'][0]
                            }
                            flattened_data.append(flat_item)
                        
                        data_frame = pd.DataFrame(flattened_data)
                        # Convert date to datetime
                        data_frame['date'] = pd.to_datetime(data_frame['date'])
                        # Sort by date
                        data_frame = data_frame.sort_values('date')
                    else:
                        # Create DataFrame and handle column names safely for non-nested data
                        data_frame = pd.DataFrame(results)
                    
                    # Only convert string column names to lowercase
                    data_frame.columns = [str(col).lower() if isinstance(col, str) else col for col in data_frame.columns]
                    
                    data_frame.to_csv(file_path, index=False)
                    print(f"The results have been saved to {file_path}, proceed with the next step")
                    
                    return f"The results have been saved to {file_path}, proceed with the next step"
                except Exception as e:
                    return f"Error executing query: {str(e)} \n review the query and try again"
                
    
            # for getting the column names
            @agent.tool
            def get_column_names(ctx: RunContext[None], file_path: Annotated[str, "The path to the csv file to get the column names from"]):
                """
                Get the column names of the dataframe.
                """
                try:
                    data_frame = pd.read_csv(file_path)
                    # Convert column names to lower case
                    data_frame.columns = data_frame.columns.str.lower()
                    # get the column names and data types
                    column_names = data_frame.columns.tolist()
                    data_types = data_frame.dtypes.tolist()
                    # create a dictionary of the column names and data types
                    column_names_and_data_types = dict(zip(column_names, data_types))
                    return f'the column names and data types are: {column_names_and_data_types}'
                except Exception as e:
                    return f"Error getting column names: {str(e)} \n review and repeat step 1"
    
            # for validating the dataframe
            @agent.tool
            def validate_data_frame(ctx: RunContext[None], file_path: Annotated[str, "The path to the csv file to validate"], 
                                        categorical_columns: Annotated[List[str], "The list of categorical columns"], 
                                        numerical_columns: Annotated[List[str], "The list of numerical columns"], 
                                        date_columns: Annotated[Optional[List[str]], "The list of date columns"] = None):
                """
                Validate the dataframe and return the schema for the dataframe. This can be later used to create visualisations and reports.
                """
                data_frame = pd.read_csv(file_path)
                
                # validate the dataframe
                if data_frame.empty:
                    print("The dataframe is empty.")
                    return("The dataframe is empty, use a dirrerent query by modifying the query variable")
                else:
                    data_frame.columns = data_frame.columns.str.lower()
                    df_schema = "The dataframe has the following columns: \n"
                    # check column names
                    for col in data_frame.columns:
                        if col in categorical_columns:
                            df_schema += f"{col} (categorical)\n"
                        elif col in numerical_columns:
                            df_schema += f"{col} (numerical)\n"
                        if date_columns is not None:
                            if col in date_columns:
                                df_schema += f"{col} (date)\n"
                                
                    for i, cols in enumerate(categorical_columns):
                        df_schema += f'\nfor {cols} {i} the corresponding info is {data_frame[cols].iloc[i]}\n'
                        
                    # Add dataframe summary with pretty printing
                    df_schema += '\nDataframe Summary:\n'
                    df_schema += '=' * 80 + '\n'
                    
                    # Get first 5 rows for summary
                    summary_df = data_frame.head()
                    
                    # Add column headers
                    headers = ' | '.join(f'{col:<20}' for col in summary_df.columns)
                    df_schema += headers + '\n'
                    df_schema += '-' * 80 + '\n'
                    
                    # Add data rows
                    for _, row in summary_df.iterrows():
                        row_str = ' | '.join(f'{str(val)[:20]:<20}' for val in row)
                        df_schema += row_str + '\n'
                        
                    # Add basic statistics for numerical columns
                    if numerical_columns:
                        df_schema += '\nNumerical Columns Statistics:\n'
                        df_schema += '=' * 80 + '\n'
                        stats = data_frame[numerical_columns].describe()
                        df_schema += str(stats) + '\n'
                                
                return df_schema
    
            # python execution tool 
            @agent.tool() 
            def python_execution_tool(ctx: RunContext[None], code: Annotated[str, "The python code to execute to generate your chart."]): 
                """ Use this tool to exeute pyhton code. If you want to see the output of a value, you should use print statement with `print(...)` function. Parameters: - code: The python code to execute to generate your chart. """ 
                repl = PythonREPL() 
                try: 
                    print('EXECUTING CODE') 
                    output = repl.run(code) # PythonREPL.run() returns the stdout output print('CODE EXECUTED SUCCESSFULLY') 
                    result_str = f"Successfully executed:\n```python\n{code}\n```\n\nOutput:\n{output}" 
                    if 'Error' in result_str: 
                        return f"Failed to execute code. Error:\n {output} \n\n Please review the code and try again" 
                    else: return ( result_str + "\n\nIf you have completed all tasks, generate the final report." ) 
                except Exception as e: 
                    return f"Failed to execute code. Error: {repr(e)} \n\n Please review the code and try again"
                
            # for writing markdown to a file
            @agent.tool
            def write_markdown_to_file(ctx: RunContext[None], content: Annotated[str, "The markdown content to write"], 
                                    filename: Annotated[str, "The name of the file (with or without .md extension)"] = "blog.md") -> str:
                """
                Write markdown content to a file with .md extension.
                Parameters:
                - content: The markdown content to write.
                - filename: The name of the file (with or without .md extension).
                """
                # Ensure filename has .md extension
                if not filename.endswith('.md'):
                    filename += '.md'
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                pdf = MarkdownPdf()
                pdf.add_section(Section(content, toc=False))
                pdf.save(filename.replace('.md', '.pdf'))
                    
                return f"File {filename} has been created successfully. \n the content is:\n {content}"
    
    else:
        with st.sidebar:
            st.info("Enter the credentials to continue")
    
    
    # executing the agent
    def run_agent(user_input: str):
        
        schema = load_json_file("schema.json")
    
        state = agent_state(
            user_query=user_input,
            schema=schema
        )
        
        result = agent.run_sync(user_input, deps=state)
        return result.data

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
                response = run_agent(user_query)
                
                # Extract fields from the response
                markdown_report = response.markdown_report
                html_path = response.html_path
                pdf_path = response.pdf_path
                code_string = response.code_string
                csv_path = response.csv_path
                pdf_path = response.pdf_path
                
    
                # Create a container for both markdown and map
                with message_placeholder.container():
                    # Display the markdown response
                    st.markdown(markdown_report)
                    
                    # Display the map if html_path exists
                    if html_path and os.path.exists(html_path):
                        with st.expander("View Map", expanded=True):
                            html_file = Path(html_path).read_text(encoding="utf-8")
                            components.html(html_file, height=500)
    
                    if csv_path and os.path.exists(csv_path):
                        with st.expander("Inspect data"):
                            st.dataframe(pd.read_csv(csv_path))
                
                # Add the assistant's response to the chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Response generated", 
                    "markdown_content": markdown_report,
                    "html_path": html_path,
                    "csv_path": csv_path,
                    "pdf_path": pdf_path
                })
                
                
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








