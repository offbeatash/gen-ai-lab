# Gen AI Lab

A hands-on learning repo for experimenting with Generative AI and LLM applications.

## Topics

* LLMs
* LangChain
* LangGraph
* RAG
* Tool Calling
* AI Agents
* Prompt Engineering

## Projects

### LangGraph RAG Demo

A conversational AI agent built with **LangGraph** that combines RAG, web search, and tool calling.

The agent can:
* Search a local FAISS knowledge base
* Search the web using Tavily
* Perform arithmetic operations
* Decide which tool to use based on the user's query
* Loop through tool calls until it generates a final answer
* Run through a Streamlit UI

### Tools

| Tool | Description |
|------|-------------|
| `search_docs` | Semantic search over the FAISS knowledge base |
| `web_search` | Web search using Tavily |
| `add` | Adds two integers |
| `subtract` | Subtracts two integers |
| `multiply` | Multiplies two integers |
| `divide` | Divides two integers |

## Technologies

* Python
* LangChain
* LangGraph
* Groq
* Google Gemini Embeddings
* FAISS
* Tavily
* Streamlit
* uv

## Project Structure

```text
.
├── src/
│   └── agenticai/
│       ├── agent.py
│       ├── app.py
│       └── ingest.py
├── data/
├── faiss_index/
├── pyproject.toml
├── uv.lock
└── .env
```

## Setup

Install dependencies:

```bash
uv sync
```

Create a `.env` file for API keys when needed:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Build the FAISS knowledge base:

```bash
uv run python src/agenticai/ingest.py
```

Run the Streamlit application:

```bash
uv run streamlit run src/agenticai/app.py
```

## Learning Goals

This repository is focused on learning and experimenting with:

* LLM applications
* RAG pipelines
* Vector databases
* Tool calling
* Web search integration
* Agentic workflows
* LangGraph state and execution
* Building AI applications with Streamlit

## Future Projects

More experiments and small projects will be added as I learn.