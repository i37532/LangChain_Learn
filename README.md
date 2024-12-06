# 案例一：Langchain入门 DEMO  

---  

# 案例二：Langchain 实现 LLM 应用程序  

构建一个简单的大型语言模型（LLM）应用程序的快速入门：  

- **调用语言模型**  
- **使用 Output Parsers**：输出解析器  
- **使用 PromptTemplate**：提示模板  
- **使用 LangSmith**：追踪你的应用程序  
- **使用 LangServe**：部署你的应用程序  

---  

# 案例三：Langchain 构建聊天机器人  

这个聊天机器人能够进行对话并记住之前的互动。需要安装：  

```bash  
pip install langchain_community
```
- **Chat History**：它允许聊天机器人“记住”过去的互动，并在回应后续问题时考虑它们。
- **流式输出**

# 案例四：Langchain 构建向量数据库和检索器

支持从向量数据库和其他来源检索数据，以便与 LLM（大型语言模型）工作流程集成。这些功能对于需要获取数据以作为模型推理的一部分的应用程序至关重要，类似于检索增强生成（RAG）的情况。需要安装：

```bash
pip install langchain-chroma  
```

- **文档**
- **向量存储**
- **检索器**

------

# 案例五：Langchain 构建代理

语言模型本身无法执行动作，它们只能输出文本。代理是使用大型语言模型（LLM）作为推理引擎来确定要执行的操作，以及这些操作的输入应该是什么。然后，这些操作的结果可以反馈到代理中，代理将决定是否需要更多的操作，或者是否可以结束。

需要安装：

```bash
pip install langgraph  
```

- **定义工具**
- **创建代理**

------

# 案例六：Langchain 构建 RAG 的对话应用

本案例是一个复杂的问答 (Q&A) 聊天机器人。应用程序可以回答有关特定源信息的问题。使用了一种称为检索增强生成 (RAG) 的技术。

### RAG简介

RAG 是一种增强大型语言模型（LLM）知识的方法，它通过引入额外的数据来实现。

需要安装：

```bash
pip install langgraph  
```
### 实现思路：

1. **加载**：首先，我们需要加载数据。这是通过 Document Loaders 完成的。
2. **分割**：Text Splitters 将大型文档分割成更小的块。这对于索引数据和将其传递给模型很有用，因为大块数据更难搜索，并且不适合模型的有限上下文窗口。
3. **存储**：我们需要一个地方来存储和索引我们的分割，以便以后可以搜索。这通常使用 Vector Store 和 Embeddings 模型完成。
4. **检索**：给定用户输入，使用检索器从存储中检索相关分割。
5. **生成**：ChatModel / LLM 使用包括问题和检索到的数据的提示生成答案。
