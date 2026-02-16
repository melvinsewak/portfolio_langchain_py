# Portfolio LangChain Python Examples

A collection of production-ready LangChain examples showcasing different functionalities and use cases. All examples are configurable through environment variables for easy deployment and customization.

## 🎯 Features

- **Environment-based configuration** - All settings managed through `.env` file
- **Production-ready patterns** - Real-world use cases and best practices
- **Interactive & demo modes** - Run examples interactively or with pre-defined scenarios
- **Comprehensive examples** - Multiple use cases from basic to advanced
- **Error handling** - Robust error handling and validation

## 📋 Examples Included

### 1. Conversational Chatbot (`01_conversational_chatbot.py`)
A production-ready conversational AI chatbot featuring:
- **AgentExecutor** for managing agent execution
- **ConversationBufferMemory** for maintaining chat history
- **Custom tools** (Calculator, Word Counter)
- **chat_history** and **agent_scratchpad** integration
- Interactive and example modes

### 2. RAG Question-Answering System (`02_rag_qa_system.py`)
Retrieval Augmented Generation system with:
- Document loading and text splitting
- **Chroma vector store** for similarity search
- **OpenAI embeddings** for vectorization
- Question answering over documents
- Source attribution

### 3. Multi-Tool Agent (`03_multi_tool_agent.py`)
Advanced agent with multiple specialized tools:
- Calculator for mathematical operations
- Data statistics analyzer
- DateTime information provider
- Text analyzer
- File writer
- Agent reasoning and tool selection

### 4. Streaming Responses (`04_streaming_responses.py`)
Real-time token streaming demonstrations:
- Basic LLM streaming
- Chain streaming
- Custom callback handlers
- Token counting
- Interactive streaming chat

### 5. Multi-Agent System (`05_multi_agent_system.py`)
Collaborative multi-agent system with orchestration:
- **Multiple specialized agents** working together
- **Research agent** for information gathering
- **Writer agent** for content creation
- **Critic agent** for quality review
- **LangGraph** for agent orchestration
- Sequential agent workflow
- Agent-to-agent communication

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/melvinsewak/portfolio_langchain_py.git
   cd portfolio_langchain_py
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_actual_api_key_here
   ```

### Running Examples

Each example can be run in two modes:

#### Interactive Mode
Allows you to interact with the example in real-time:

```bash
# Conversational Chatbot
python examples/01_conversational_chatbot.py --mode interactive

# RAG Q&A System
python examples/02_rag_qa_system.py --mode interactive

# Multi-Tool Agent
python examples/03_multi_tool_agent.py --mode interactive

# Streaming Chat
python examples/04_streaming_responses.py --mode interactive

# Multi-Agent System
python examples/05_multi_agent_system.py --mode interactive
```

#### Example/Demo Mode
Runs pre-defined scenarios to demonstrate capabilities:

```bash
# Conversational Chatbot Demo
python examples/01_conversational_chatbot.py --mode example

# RAG Q&A Demo
python examples/02_rag_qa_system.py --mode example

# Multi-Tool Agent Demo
python examples/03_multi_tool_agent.py --mode example

# All Streaming Examples
python examples/04_streaming_responses.py --mode all

# Multi-Agent System Demo
python examples/05_multi_agent_system.py --mode example
```

## ⚙️ Configuration

All examples are configured through the `.env` file:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - Model configuration
OPENAI_MODEL=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=1000

# Optional - Behavior
VERBOSE=false

# Optional - RAG configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key (required) | - |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-3.5-turbo` |
| `TEMPERATURE` | Sampling temperature (0.0 to 1.0) | `0.7` |
| `MAX_TOKENS` | Maximum tokens in response | `1000` |
| `VERBOSE` | Enable verbose logging | `false` |
| `CHROMA_PERSIST_DIRECTORY` | Vector store location | `./chroma_db` |
| `CHUNK_SIZE` | Document chunk size for RAG | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |

## 📚 Example Usage

### Running Examples from Command Line

The examples are designed to be run as standalone scripts:

```bash
# Conversational Chatbot
python examples/01_conversational_chatbot.py --mode interactive

# RAG Q&A System  
python examples/02_rag_qa_system.py --mode example

# Multi-Tool Agent
python examples/03_multi_tool_agent.py --mode interactive
```

### Programmatic Usage (Advanced)

If you need to import and use the functions programmatically, you can use importlib:

```python
import sys
import importlib.util

# Load the conversational chatbot module
spec = importlib.util.spec_from_file_location(
    "chatbot",
    "examples/01_conversational_chatbot.py"
)
chatbot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chatbot_module)

# Create and use the agent
agent = chatbot_module.create_conversational_agent()
response = agent.invoke({"input": "What is 25 * 4?"})
print(response['output'])
```

Or add the examples directory to your Python path:

```python
import sys
sys.path.insert(0, 'examples')

# Import using the filename as-is (requires Python import hooks or renaming)
# For simpler integration, consider running examples as scripts
```

**Note:** Due to the numeric prefixes in filenames (e.g., `01_`, `02_`), the examples are best run as standalone scripts rather than imported as modules. For production use, consider creating wrapper modules with standard Python naming conventions.

## 🏗️ Project Structure

```
portfolio_langchain_py/
├── examples/
│   ├── __init__.py
│   ├── 01_conversational_chatbot.py    # Chatbot with memory and tools
│   ├── 02_rag_qa_system.py             # RAG Q&A system
│   ├── 03_multi_tool_agent.py          # Multi-tool agent
│   └── 04_streaming_responses.py       # Streaming examples
├── utils/
│   ├── __init__.py
│   └── config.py                        # Configuration management
├── .env.example                         # Example environment file
├── .gitignore                          # Git ignore rules
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🔧 Advanced Usage

### Custom Tools
You can easily add custom tools to the agents:

```python
from langchain.tools import Tool

def custom_function(input: str) -> str:
    # Your custom logic
    return f"Processed: {input}"

custom_tool = Tool(
    name="CustomTool",
    func=custom_function,
    description="Description of what your tool does"
)

# Add to agent's tools list
tools.append(custom_tool)
```

### Custom Documents for RAG
Load your own documents for the RAG system:

```python
from langchain.docstore.document import Document

documents = [
    Document(
        page_content="Your document content here",
        metadata={"source": "document1", "topic": "subject"}
    ),
    # Add more documents...
]

qa_chain = setup_rag_system(documents=documents)
```

## 🛠️ Production Considerations

These examples include production-ready features:

1. **Environment-based configuration** - Easy deployment across environments
2. **Error handling** - Graceful error handling and user feedback
3. **Validation** - Configuration validation before execution
4. **Structured tools** - Type-safe tool inputs using Pydantic
5. **Persistence** - Vector stores with persistence for RAG
6. **Streaming** - Real-time response streaming for better UX
7. **Memory management** - Conversation history management
8. **Monitoring** - Optional verbose mode for debugging

## 📖 Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Vector Stores Guide](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [Agents Guide](https://python.langchain.com/docs/modules/agents/)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new examples
- Improve existing examples
- Fix bugs
- Improve documentation

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [OpenAI](https://openai.com/)

---

**Note**: Remember to keep your API keys secure and never commit them to version control!