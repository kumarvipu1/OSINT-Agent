# Open Intelligence

![Open Intelligence](thumbnail.png)

## Overview

Open Intelligence is an interactive data analysis platform that allows users to explore global events through natural language queries. The platform connects to a graph database of global events and provides visual insights through maps, charts, and comprehensive reports.

## Features

- **Conversational Interface**: Ask questions about global events in natural language
- **Interactive Maps**: Visualize events geographically with interactive folium maps
- **Data Analysis**: Automatically analyze trends and patterns in conflict data
- **Report Generation**: Create comprehensive markdown reports summarizing findings
- **Query Caching**: Save previous analyses for quick reference

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/open_intelligence.git
cd open_intelligence
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
- Use the credentials form in the sidebar to enter your credentials.

> **Note:** Logfire token is optional, it is used to track the usage of the agent.

## Usage

1. Start the application:
```bash
streamlit run app_ui.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Begin by asking a question in the chat interface, such as:
   - "Show me conflict where the number of fatalities is greater than 1000"
   - "What are the top actor groups involved in conflicts in Africa?"
   - "Map the events with the most fatalities in the year 2015"
   - "Compare conflict trends in between 2018 and 2022"

## Architecture

The Open Intelligence platform consists of the following components:

- **Streamlit UI**: A clean, professional chat interface for user interaction
- **Agent Module**: An AI-powered agent that processes user queries uses pydanticAI as a framework and Logfire for observability.
- **Graph Database**: ArangoDB instance storing global event data
- **Data Visualization**: Dynamic visualization tools for maps and charts

## Example Queries

- **Regional Analysis**: "Show me conflicts in North Africa in 2022"
- **Actor Analysis**: "Which groups were most active in Syria?"
- **Temporal Analysis**: "Compare conflict patterns before and after COVID-19"
- **Geospatial Analysis**: "Show me a heatmap of events near border regions"

## Future Work

- Add more data sources and visualizations
- Implement more complex analysis and reporting
- Add more features to the agent module
