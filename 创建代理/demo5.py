import getpass
import os
from re import search

from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import *
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import chat_agent_executor
from langserve import add_routes

os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']="pr-bumpy-bookend-28"
os.environ['TAVILY_API_KEY']="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"


# 1.创建模型
model = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key="213a6cfb72c072e08aa4aac4a8d9654d.Q3sNH4CWtF83tI7W",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 没有代理
# “对不起，我无法提供实时信息，包括当前的天气状况。”
res = model.invoke([HumanMessage(content='雅安天气怎么样？')])
# print(res)

# LangChain内置了工具，可以使用Tavily 搜索引擎作为工具
search = TavilySearchResults(max_results=1) # 限制返回结果
# print(search.invoke('北京天气怎么样？'))

# 模型绑定工具
# 模型可以自动推理是否需要调用工具去完成用户答案
tools = [search]

# 创建代理
agent_exe = chat_agent_executor.create_tool_calling_executor(model,tools)

resp = agent_exe.invoke({'messages':[HumanMessage(content='1+1=？')]})
print(resp['messages'])

resp2 = agent_exe.invoke({'messages':[HumanMessage(content='北京天气怎么样？')]})

print(resp2['messages'])
print(resp2['messages'][2].content)
