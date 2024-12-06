### 1.保存聊天的历史记录

每个SESSION_ID对应一次对话。

```python
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

### 2. `RunnableWithMessageHistory`

**定义**:
`RunnableWithMessageHistory` 是一种工具，它将一个可调用的对象（如链、函数或模型）与历史消息记录结合起来。它通过记录会话中所有的消息，使得上下文得以保留，非常适合需要持续对话的情景，比如聊天机器人或客户支持系统。

### 3. 代码分析

1. **创建 `do_message` 实例**:

   ```python
   do_message = RunnableWithMessageHistory(  
       chain,  
       get_session_history,  
       input_messages_key='my_msg'   
   )  
   ```

   - `chain`: 这是一个用于处理输入并生成响应的处理链。可能是一个将模型与解析器组合在一起的对象。
   - `get_session_history`: 这是一个函数或方法，用于获取当前会话的历史消息。这些历史消息可以被用作上下文，以便在生成响应时考虑之前的对话内容
   - `input_messages_key='my_msg'`: 这个参数指定每次发送消息时使用的键名。这就是代码发送用户消息的标识符，表明应该从输入中提取与消息相关的内容。