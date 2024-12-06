### 1. `vector_store`

**定义**:
`vector_store` 是一个向量数据库实例，它基于文档的嵌入（embeddings）来存储和检索信息。这里使用的是 `Chroma` 工具，它允许我们通过计算文档的向量表示来进行高效的相似性搜索。

**作用**:

- **存储文档嵌入**: 在代码中，文档被转化为向量嵌入（embeddings），这些向量将用于衡量不同文档之间的相似性。`Chroma.from_documents(documents, embedding=embeddings)` 表示将提供的文档与对应的向量一起存入 Chroma 向量数据库。
- **支持相似性搜索**: 向量存储的关键功能是能够执行快速的相似性搜索，允许用户查询与给定问题最相关的文档。

### 2. `retriever`

**定义**:
`retriever` 是一个可执行的对象（Runnable），它使用向量存储来检索与用户查询相关的文档。这里使用的是 `RunnableLambda`，可以看作是一个封装了具体检索逻辑的对象。

**作用**:

- **检索相关文档**: `retriever` 通过调用 `vector_store.similarity_search_with_score` 来搜索与用户问题相关的文档。使用 `bind(k=1)` 指定只返回一个与查询最相关的结果。
- **与其他组件链接**: 由于它是一个 Runnable 对象，可以通过管道将其与其他组件（如提示模板和模型）连接，实现整个对话或查询的处理链。在代码中，检索器与 `PromptTemplate` 和 `ChatOpenAI` 模型组合成一个完整的处理链(`chain`)，方便地将用户的问题与相关文档连接起来。
