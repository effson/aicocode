#  Anthropic 协议                                                                                                                                              

## 普通文本消息 
```yaml
{
  "role": "user", 
  "content": "hello"
}                                                                                                                                                   
{
  "role": "assistant", 
  "content": "hello back"
}
```

## 工具调用的 assistant 消息 
```yaml
{
  "role": "assistant", 
  "content": [                                           
    {
      "type": "thinking", 
      "thinking": "...", 
      "signature": "..."
    },   // 可选 thinking block          
    {
      "type": "text",
      "text": "let me..."
    },
    {
      "type": "tool_use", 
      "id": "toolu_xxx", 
      "name": "Bash", 
      "input": {"command": "ls"}
    }        
  ]
}                                                                                                                                                                                     
```
##  工具结果 —  user消息   
```yaml
{
  "role": "user", 
  "content": [                                                
    {
      "type": "tool_result", 
      "tool_use_id": "toolu_xxx", 
      "content": "file1.txt\n", 
      "is_error": false
    }       
  ]
}                                                                                                                                                                                     
```
## tool Schema:   
```yaml
{                                                                          
  "name": "Bash",                                                                  
  "description": "...",                                                                    
  "input_schema": { // ← 字段名: input_schema                                                                 
     "type": "object",                                                                    
     "properties": {...}                                                                 
  }                                                                             
} 
```

---

# OpenAI 协议 


## 普通文本消息
```yaml                                                                                                                                                                
{
  "role": "user", 
  "content": "hello"
}                                                                                                                                                   
{
  "role": "assistant", 
  "content": "hello back"
}
```

## Thinking block — 独立 reasoning item
```yaml                                                                                                                                                                
{
  "type": "reasoning", 
  "id": "sig", 
  "summary": [
  	{
  	  "type": "summary_text", 
  	  "text": "..."
  	}
  ]
}
```
## 工具调用 — 类型标签 item
```yaml
{
  "type": "function_call", 
  "name": "Bash", 
  "call_id": "xxx", 
  "arguments": "{\"command\":\"ls\"}"
}   
```

## 工具结果 — 独立的 function_call_output item
```yaml
{
  "type": "function_call_output", 
  "call_id": "xxx", 
  "output": "file1.txt\n"
} 
```

## tool Schema
```yaml
{                                                                            
  "type": "function",
  "name": "Bash",                                                     
  "description": "...",
  "parameters": schema
}     
```
---

# OpenAI 兼容协议

## 概览
```yaml
{
  "role": "user", 
  "content": "hello"
}
{
  "role": "assistant", 
  "content": "hello back",
  "reasoning_content": "...",
  "tool_calls": [
  	{                                                            
      "id": "xxx",                                                           
      "type": "function",                                                         
      "function": {                                                          
        "name": "Bash",                                     
        "arguments": "{\"command\":\"ls\"}"   // ← JSON 字符串，不是对象               
      }                                                                  
    }
  ]
}
```

## 普通文本消息
```yaml
{
  "role": "user", 
  "content": "hello"
}                                                                                                                                                   
{
  "role": "assistant", 
  "content": "hello back"
}   
```
## 工具调用 — 内嵌在 assistant 消息中
```yaml
{                                                                   
  "role": "assistant",                                                     
  "content": "let me check...",                                                     
  "tool_calls": [
  	{                                                            
      "id": "xxx",                                                           
      "type": "function",                                                         
      "function": {                                                          
        "name": "Bash",                                     
        "arguments": "{\"command\":\"ls\"}"   // ← JSON 字符串，不是对象               
      }                                                                  
    }
  ]                                                                       
}   
```
## 工具结果 — 独立 role: "tool"
```yaml
{
  "role": "tool", 
  "tool_call_id": "xxx", 
  "content": "file1.txt\n"
}  
```
## Thinking — reasoning_content 非标准字段（DeepSeek/小米）
```yaml
{
  "role": "assistant", 
  "content": "...", 
  "reasoning_content": "..."
}  
```
## tool Schem
```yaml
{                                                                  
  "type": "function",                                                
  "function": { // ← 多一层 function 嵌套                                
    "name": "Bash",                                                       
     "description": "...",                                              
     "parameters": { // ← 字段名: parameters
       "type": "object",                                               
       "properties": {...}                                                              
    }                                                                           
  }
}
```

---

# AicoCode Message

```python
@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    tool_uses: list[ToolUseBlock] = field(default_factory=list)
    tool_results: list[ToolResultBlock] = field(default_factory=list)
    thinking_blocks: list[ThinkingBlock] = field(default_factory=list)
```
```python
@dataclass
class ToolUseBlock:
    tool_use_id: str
    tool_name: str
    arguments: dict[str, Any]
```
```python
@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
```
```python
@dataclass
class ThinkingBlock:
    thinking: str
    signature: str
```