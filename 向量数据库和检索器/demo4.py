import getpass
import os


from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import *
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langserve import add_routes

os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']="pr-bumpy-bookend-28"


# 1.创建模型
model = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key="213a6cfb72c072e08aa4aac4a8d9654d.Q3sNH4CWtF83tI7W",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 准备测试数据 ，假设我们提供的文档数据如下：
documents = [
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

embeddings = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key="569b09310512db6b787f490e05a0f432.sHD6oGSld5kszivl",
)
# 实例化一个向量空间
# 可以计算向量相似度
vector_store = Chroma.from_documents(documents,embedding=embeddings)

# 相似度的查询：返回相似的分数，越低相似越高
# print(vector_store.similarity_search_with_score('哈基米'))

# 检索器（ Runnable对象，才可以链接为 chain
# bind = 1,返回相似度最高的那一个
retriever = RunnableLambda(vector_store.similarity_search_with_score).bind(k=1)

# batch，可以匹配多个向量
# print(retriever.batch(['黑猫', '哈吉汪']))

# 提示模板
msg = '''
使用提供的上下文仅仅回答这个问题：
{question}
上下文：
{text}
'''

prompt_temp = ChatPromptTemplate.from_messages([('human',msg)])

# RunnablePassthrough允许将用户的问题 待一会 再传递给prompt和model
chain = {'question':RunnablePassthrough(),'text':retriever} | prompt_temp | model

resp = chain.invoke('请谈谈鹦鹉的优点')

print(resp.content)