import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

os.environ['LANGCHAIN_API_KEY'] = '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'
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

res = model.invoke(msg)
print(res)

# 简单解析响应数据
# 3.创建返回的数据解析器
parser = StrOutputParser()
return_str = parser.invoke(res)
print(return_str)

# 4.得到 Chain
chain = model | parser

# 5.直接使用chain调用
print(chain.invoke(msg))