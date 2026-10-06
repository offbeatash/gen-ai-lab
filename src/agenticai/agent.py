import operator
import os
from typing import Annotated, Literal

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from langchain_tavily import TavilySearch

load_dotenv()

# Project root and FAISS index path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
INDEX_DIR = os.path.join(PROJECT_ROOT, "faiss_index")


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divides a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def search_docs(query: str) -> str:
    """Search the knowledge base for AI/ML and related topics."""

    index_file = os.path.join(INDEX_DIR, "index.faiss")

    if not os.path.exists(index_file):
        return f"FAISS index not found at: {INDEX_DIR}"

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    return "\n\n---\n\n".join(
        doc.page_content for doc in docs
    )



tavily_search = TavilySearch(max_results=5)


@tool
def web_search(
    query: str,
    topn: int = 5,
    source: str = "general",
) -> str:
    """Search the web for current information."""
    try:
        result = tavily_search.invoke({"query": query})
        return str(result)
    except Exception as e:
        return f"Web search error: {e}"

tools = [add, multiply, divide, search_docs, web_search]
tools_by_name = {t.name: t for t in tools}


model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

model_with_tools = model.bind_tools(tools)


def llm_call(state: MessagesState) -> dict:
    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful AI assistant. "
                    "Use math tools for calculations. "
                    "Use search_docs for questions about AI, ML, "
                    "LangGraph, RAG, embeddings, transformers, "
                    "and related topics."
                    "Use web_search when the user needs current, "
                    "recent, or web-based information."
                )
            )
        ]
        + state["messages"]
    )

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: MessagesState) -> dict:
    results = []

    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        selected_tool = tools_by_name[tool_name]

        try:
            observation = selected_tool.invoke(
                tool_call["args"]
            )
        except Exception as e:
            observation = f"Tool error: {e}"

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": results}


def should_continue(
    state: MessagesState,
) -> Literal["tool_node", "__end__"]:
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"

    return END


agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END],
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()
