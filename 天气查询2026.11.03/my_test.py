
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage,AIMessage


# #1.构造模型
model = ChatOpenAI(model= "qwen3.8-max",
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    api_key=""
# 2.调用工具列表
from tools import get_weather

#3.创造agent
graph = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个天气预报助手！",
    checkpointer=MemorySaver(),
    )

#4.主函数：
def main():
    while True:
        user_input = input(f"请输入查询的城市名：")
        if user_input == "exit":
            break
        response = graph.invoke({"messages":[{"role": "user", "content": user_input}]},config={"configurable": {"thread_id": "user_001"}})
        
        for message in response['messages']:
            if isinstance(message, AIMessage):
                print(message.content)
        
if __name__ == "__main__":
    main()