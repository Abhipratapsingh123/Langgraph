from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
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





llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',streaming = True)

# model = HuggingFaceEndpoint(
#     repo_id="NousResearch/Hermes-2-Pro-Llama-3-8B",
#     task="text-generation"
# )
# llm = ChatHuggingFace(llm=model)

# embedding mode from hugging face

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



### rag steps 


 ### loader 
loader = PyPDFLoader("C:\\Users\\abhip\\Desktop\\Langraph\\6_Chatbot_using_langgraph\\DA_2026_Syllabus.pdf")
docs = loader.load()


### splitting  
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.split_documents(docs)

### creating embedding for chunks  and storing in vectorstore 
vector_store = FAISS.from_documents(chunks,embedding_model)

### creating a retriver 

retriever = vector_store.as_retriever(search_type = 'similarity',search_kwargs={'k':4})

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

@tool 
def rag_tool(query:str)->dict:
    """
    USE this tool when the query is about syllabus of GATE exam
    Retreive relevant information from the pdf document.
    Use this tool when the use ask factual/conceptual questions
    that might be answered from the stored documents
    """
   
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    
    return {
        'query':query,
        'context':context,
        'metadata':metadata
    }




# tool list 
tools = [search_tool,calculator,get_stock_price,rag_tool]

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
        