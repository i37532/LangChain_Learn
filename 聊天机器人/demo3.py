import os

from fastapi import FastAPI
from langchain_community.chat_message_histories import *
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langserve import add_routes

os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']="pr-bumpy-bookend-27"

# 1.创建模型
model = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key="213a6cfb72c072e08aa4aac4a8d9654d.Q3sNH4CWtF83tI7W",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 定义prompt模板
prompt_template = ChatPromptTemplate.from_messages([
    ('system','你是一个乐于助人的助手，用{language}回答所有问题'),
    MessagesPlaceholder(variable_name='my_msg')
])

# 4.得到 Chain
chain = prompt_template | model

# 保存聊天的历史记录
store = {}  # 保存所有用户的聊天记录。key：sessionId
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg' # 每次聊天时候发送msg的key
)

config = {'configurable':{'session_id':'yr01'}}  # 给当前对话定义一个sessionID

# 第一轮
resp1 = do_message.invoke(
    {
    'my_msg':[HumanMessage(content='你好，1+1=？')],
    'language':'中文',
    },
    config = config
)

print(resp1.content)

# 第二轮
resp2 = do_message.invoke(
    {
    'my_msg':[HumanMessage(content='你好，在刚刚的结果上再+1呢？')],
    'language':'中文',
    },
    config = config
)

print(resp2.content)
