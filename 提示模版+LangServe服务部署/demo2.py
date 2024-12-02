import os

from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
# 2.准备提示
msg = [
    SystemMessage(content='请将以下内容翻译为英语'),
    HumanMessage(content='LangChain 是一个用于开发由语言模型驱动的应用程序的框架。')
]
# 定义prompt模板
prompt_template = ChatPromptTemplate.from_messages([
    ('system','请将以下内容翻译成{language}'),
    ('user',"{text}")
])

# 简单解析响应数据
# 3.创建返回的数据解析器
parser = StrOutputParser()

# 4.得到 Chain
chain = prompt_template | model | parser

# 5.直接使用chain调用
# print(chain.invoke({'language':'Japanese','text':'识时务者为俊杰'}))

# 程序部署成服务
# 创建fastAPI
app = FastAPI(title='my LangChain demo',version='0.0.1',description='翻译语言')

add_routes(
    app,
    chain,
    path='/chaindemo'
)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
