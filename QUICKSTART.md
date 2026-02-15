# Quick Start Guide

Welcome to the LangChain Portfolio Examples! This guide will help you get started quickly.

## Prerequisites

- Python 3.8 or higher
- An OpenAI API key (get one at https://platform.openai.com/api-keys)

## Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Verify Setup

Run the validation script:
```bash
python validate_structure.py
```

You should see all checks passing ✅

## Running Your First Example

### Option 1: Demo Mode (Recommended for First Try)

Run a pre-defined demo to see the chatbot in action:
```bash
python examples/01_conversational_chatbot.py --mode example
```

This will run through several example conversations demonstrating:
- Basic conversation
- Calculator tool usage
- Word counting
- Memory (referencing previous messages)

### Option 2: Interactive Mode

Chat with the bot interactively:
```bash
python examples/01_conversational_chatbot.py --mode interactive
```

Try asking:
- "What is 15 * 8 + 42?"
- "How many words are in: The quick brown fox jumps over the lazy dog?"
- "What was my first question?" (tests memory)

Type 'exit' to quit.

## What's Next?

### Explore Other Examples

**RAG Q&A System** - Ask questions about documents:
```bash
python examples/02_rag_qa_system.py --mode example
```

**Multi-Tool Agent** - Agent with multiple specialized tools:
```bash
python examples/03_multi_tool_agent.py --mode example
```

**Streaming Responses** - See real-time token streaming:
```bash
python examples/04_streaming_responses.py --mode all
```

### Customize Configuration

Edit `.env` to adjust behavior:

```env
# Use GPT-4 for better responses (costs more)
OPENAI_MODEL=gpt-4

# Adjust creativity (0.0 = deterministic, 1.0 = creative)
TEMPERATURE=0.9

# See detailed agent reasoning
VERBOSE=true
```

## Common Issues

### "Module not found" error
Make sure you installed dependencies:
```bash
pip install -r requirements.txt
```

### "Missing required environment variable: OPENAI_API_KEY"
Check that:
1. You created a `.env` file (not `.env.example`)
2. You added your actual API key
3. The key starts with `sk-`

### Rate limit errors
If you hit OpenAI rate limits:
1. Wait a few minutes
2. Reduce `MAX_TOKENS` in `.env`
3. Switch to a different model if available

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Explore the code** - Each example is well-documented
3. **Customize tools** - Add your own tools to the agents
4. **Build your own** - Use these as templates for your projects

## Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review the inline code comments in each example
- Visit [LangChain Documentation](https://python.langchain.com/)

Happy coding! 🚀
