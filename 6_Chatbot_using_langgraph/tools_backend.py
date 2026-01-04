from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
import os
from db import conn
from dotenv import load_dotenv


load_dotenv()

# llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',streaming = True)
model = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation"
)
llm = ChatHuggingFace(llm=model)


####### tools ######

# Tools 
search_tool =   DuckDuckGoSearchRun(region='us-en')

def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
    
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    Always convert the price into indian rupess
    """
    api_key = os.getenv('STOCK_API')
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()


# tool list 
tools = [search_tool,calculator,get_stock_price]

# llm binding with tools 
llm_with_tools = llm.bind_tools(tools)



# defining state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]




# function for node
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)



# Checkpointer
checkpointer = SqliteSaver(conn)


graph = StateGraph(ChatState)
#node
graph.add_node("chat_node", chat_node)
graph.add_node("tools",tool_node)

# edges
graph.add_edge(START,"chat_node")
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge("tools","chat_node")


chatbot = graph.compile(checkpointer=checkpointer)



# retrive all threads function
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
        
    return list(all_threads)
        