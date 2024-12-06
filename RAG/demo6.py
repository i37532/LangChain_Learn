import getpass
import os
from re import search

import bs4
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import *
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import chat_agent_executor
from langserve import add_routes


os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']="pr-bumpy-bookend-28"
os.environ['TAVILY_API_KEY']="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"

'''
RAG问答是“Retrieval-Augmented Generation”的缩写，翻译为“检索增强生成”。
它是一种结合了信息检索和生成模型的自然语言处理技术。在RAG框架中，系统首先从外部知识库中检索相关信息，
然后将这些信息与生成模型结合，来生成更准确、更详细的回答。
'''

# 1.创建模型
model = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key="213a6cfb72c072e08aa4aac4a8d9654d.Q3sNH4CWtF83tI7W",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 1.加载数据：一篇blog
loader = WebBaseLoader(
    web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'],
    bs_kwargs=dict(
        parse_only = bs4.SoupStrainer(class_=('post-header','post-title','post-content'))  # 标题、作者信息、内容
    )
)

docs = loader.load()
# print(docs)

# 2.对于大文本，需要分割
# eg.
# test_text = "hello world, how about you? thanks, I am fine.  the machine learning class. So what I wanna do today is just spend a little time going over the logistics of the class, and then we'll start to talk a bit about machine learning"
# splitter = RecursiveCharacterTextSplitter(chunk_size=25,chunk_overlap=4)  # chunk_overlap 允许字符重复使用
# res = splitter.split_text(test_text)
# for s in res:
#     print(s)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)  # chunk_overlap 允许字符重复使用
splits = splitter.split_documents(docs)  # splits是个list，里面装的document
# print(type(splits[0]))

# 3.存储
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key="569b09310512db6b787f490e05a0f432.sHD6oGSld5kszivl",
)

# 实例化向量空间
vector_store = Chroma.from_documents(docs,embedding=embeddings)


# 4.检索器
retriever = vector_store.as_retriever()

# 整合
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
don't know. Use three sentences maximum and keep the answer concise.\n

{context}
"""
# 提问和回答的历史记录模板
prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])

# 得到chain
chain = create_stuff_documents_chain(model,prompt)
#
# retrieval_chain =  create_retrieval_chain(retriever,chain)
#
# resp = retrieval_chain.invoke({"input":"What is Task Decomposition?"})
#
# print(resp['answer'])

'''
注意：
一般情况下，我们构建的链（chain）直接使用输入问答记录来关联上下文。但在此案例中，查询检索器也需要对话上下文才能被理解。

解决办法：
添加一个子链(chain)，它采用最新用户问题和聊天历史，并在它引用历史信息中的任何信息时重新表述问题。这可以被简单地认为是构建一个新的“历史感知”检索器。
这个子链的目的：让检索过程融入了对话的上下文。
'''


# 子链提示模板
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, 
formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

retriever_history_temp = ChatPromptTemplate.from_messages([
    ('system',contextualize_q_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human',"{input}")
])

# 创建一个子链
history_chain = create_history_aware_retriever(model,retriever,retriever_history_temp)

# 保存问答历史记录
store = {}
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 创建chain：把前两个链整合
all_chain = create_retrieval_chain(history_chain,chain)

# 创建一个Runnable对象
result_chain = RunnableWithMessageHistory(
    all_chain,
    get_session_history,
    input_message_key='input',
    history_messages_key='chat_history',
    output_message_key='answer',
)

# 第一轮对话
resp1 = result_chain.invoke(
    {'input':'what is Task Decomposition?'},
    config={'configurable':{'session_id':'yr02'}}
)
print(resp1['answer'])

# 第二轮对话
resp2 = result_chain.invoke(
    {'input':'what are common ways of doing it?'},
    config={'configurable':{'session_id':'yr03'}}
)
print(resp2['answer'])