from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import START,END,StateGraph ,MessagesState
from pydantic import BaseModel,Field
from langchain.tools import tool
from langgraph.prebuilt.tool_node import ToolNode 
from langgraph.checkpoint.memory import InMemorySaver
from chromadb.api.shared_system_client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain.messages import SystemMessage
import chromadb

#1.配置模型：
chat_model = ChatOpenAI(model="qwen3.8-max",base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
                          api_key="sk-ws-H.YLHYYPX.Yrak.MEUCIQCTdzglRMGy37HmU1VnYcSH6Hyce71ViLrj2zqZP2g3DAIgTwT-lgL83capmx74qqUNUreEATzY-UoeibaY1bRduV0")

embedding_model = OpenAIEmbeddings(model="qwen3.7-text-embedding-flash",base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
                                  api_key="sk-ws-H.YIPXDYD.GQ3w.MEQCIGUk6xKey1vwED7Q_f8nI0ERsDRWSfsbLZw89wKsQf4rAiA-f9DPuXKRa1copGfbyLvFktFzwtZgg4q7YVlhUtA3iQ",
                                  check_embedding_ctx_length=False)

#2.构建节点
#2.1构建工具节点

#先全局化创建好db，防止ToolNode多线程并发访问
_client = chromadb.PersistentClient(
    path=r"E:\PythonProjects\2026.10.27 lang_chain\20261109 RAG_1\data\chroma_db")

_db = Chroma(
    client=_client,
    collection_name="car_info_2",
    embedding_function=embedding_model,)

#创建工具节点
class Search_Chunks(BaseModel):
    question:str = Field(description="关于峨眉山景区的问题")
    
@tool
def search_chunk(question: str) -> Search_Chunks:
    """查询有关峨眉山旅游的风景介绍！"""
    results = _db.similarity_search(question, k=2)
    
    results_context = [c.page_content for c in results]

    # print(results_context)
    return results_context

tools = [search_chunk]

model_with_tools = chat_model.bind_tools(tools)

tool_node = ToolNode(tools)

#2.2构建chat节点
def chat_model(state: MessagesState):
    # 强制告诉模型：只要涉及峨眉山，必须调用工具
    system_prompt = SystemMessage(content="你是一个专业的峨眉山导游助手。当用户询问关于峨眉山的问题时，你必须调用 search_chunk 工具来获取准确信息，不要凭空回答。")
    
    messages = [system_prompt] + state["messages"]
    response = model_with_tools.invoke(messages)
    
    # print(f"==== 模型工具调用情况: {response.tool_calls} ====")
    return {"messages": [response]}

#3.构建条件边
def condition_edge(state:MessagesState):

    if state["messages"][-1].tool_calls:
        return "tool_node"
    else:
        return END
    
#4.构建图
#4.1构建图纸
builder = StateGraph(MessagesState)

#4.2添加节点和边
builder.add_node("chat",chat_model)
builder.add_node("tool_node",tool_node)
builder.add_edge(START,"chat")
builder.add_conditional_edges("chat",condition_edge)
builder.add_edge("tool_node","chat")
# builder.add_edge("chat",END)

#4.3 添加检查点 + 编译图
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)


if __name__ == "__main__":
    
    result = graph.invoke({"messages":[("user","查询峨眉山优美的景点")]},{"configurable": {"thread_id": "user_001"}})

    result_context = result["messages"][-1].content
    print(result_context)