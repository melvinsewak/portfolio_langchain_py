"""
Streaming Responses Example using LangChain.

This example demonstrates streaming capabilities with LangChain:
- Real-time token streaming from LLMs
- Streaming with chains
- Streaming with agents
- Environment-based configuration

Features:
- Token-by-token response streaming
- Better user experience for long responses
- Configurable streaming behavior
- Multiple streaming patterns
"""
import sys
import os
from typing import Iterator

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from utils import get_config


def stream_basic_llm():
    """Demonstrate basic LLM streaming."""
    print("=" * 60)
    print("🌊 Basic LLM Streaming")
    print("=" * 60)
    print("\nStreaming a response token by token...\n")
    
    config = get_config()
    
    if not config.validate():
        print("❌ Configuration validation failed.")
        sys.exit(1)
    
    # Create LLM with streaming callback
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        openai_api_key=config.openai_api_key,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    prompt = "Write a short poem about artificial intelligence and the future."
    
    print("🤖 Assistant: ", end="", flush=True)
    response = llm.invoke(prompt)
    print("\n")
    
    return response


def stream_with_chain():
    """Demonstrate streaming with a chain."""
    print("=" * 60)
    print("⛓️  Chain Streaming")
    print("=" * 60)
    print("\nStreaming with a prompt template chain...\n")
    
    config = get_config()
    
    if not config.validate():
        print("❌ Configuration validation failed.")
        sys.exit(1)
    
    # Create LLM with streaming
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        openai_api_key=config.openai_api_key,
        streaming=True,
    )
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains complex topics in simple terms."),
        ("human", "{topic}")
    ])
    
    # Create chain with output parser
    chain = prompt | llm | StrOutputParser()
    
    topic = "Explain how neural networks work"
    
    print(f"📚 Topic: {topic}\n")
    print("🤖 Assistant: ", end="", flush=True)
    
    # Stream the response
    for chunk in chain.stream({"topic": topic}):
        print(chunk, end="", flush=True)
    
    print("\n")


def stream_multiple_prompts():
    """Demonstrate streaming multiple prompts."""
    print("=" * 60)
    print("📝 Multiple Streaming Prompts")
    print("=" * 60)
    
    config = get_config()
    
    if not config.validate():
        print("❌ Configuration validation failed.")
        sys.exit(1)
    
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        openai_api_key=config.openai_api_key,
        streaming=True,
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a concise assistant. Keep responses brief."),
        ("human", "{question}")
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    
    questions = [
        "What is machine learning?",
        "What is the difference between AI and ML?",
        "What are the main types of machine learning?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print(f"{'='*60}")
        print("🤖 Assistant: ", end="", flush=True)
        
        for chunk in chain.stream({"question": question}):
            print(chunk, end="", flush=True)
        
        print()  # New line after each response


def stream_with_custom_callback():
    """Demonstrate streaming with a custom callback handler."""
    print("=" * 60)
    print("🎯 Custom Callback Streaming")
    print("=" * 60)
    print("\nStreaming with custom token counting...\n")
    
    from langchain.callbacks.base import BaseCallbackHandler
    
    class TokenCounterCallback(BaseCallbackHandler):
        """Custom callback to count tokens."""
        
        def __init__(self):
            self.token_count = 0
            self.tokens = []
        
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            """Called when a new token is generated."""
            self.token_count += 1
            self.tokens.append(token)
            print(token, end="", flush=True)
        
        def on_llm_end(self, *args, **kwargs) -> None:
            """Called when LLM finishes."""
            print(f"\n\n📊 Total tokens streamed: {self.token_count}")
    
    config = get_config()
    
    if not config.validate():
        print("❌ Configuration validation failed.")
        sys.exit(1)
    
    counter_callback = TokenCounterCallback()
    
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        openai_api_key=config.openai_api_key,
        streaming=True,
        callbacks=[counter_callback]
    )
    
    prompt = "Explain the concept of streaming in 3 sentences."
    
    print("🤖 Assistant: ", end="", flush=True)
    response = llm.invoke(prompt)
    print()


def interactive_streaming_chat():
    """Run an interactive chat with streaming responses."""
    print("=" * 60)
    print("💬 Interactive Streaming Chat")
    print("=" * 60)
    print("\nChat with streaming responses. Type 'exit' to quit.\n")
    print("=" * 60)
    
    config = get_config()
    
    if not config.validate():
        print("❌ Configuration validation failed.")
        sys.exit(1)
    
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        openai_api_key=config.openai_api_key,
        streaming=True,
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and friendly AI assistant."),
        ("human", "{input}")
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            print("🤖 Assistant: ", end="", flush=True)
            
            # Stream the response
            for chunk in chain.stream({"input": user_input}):
                print(chunk, end="", flush=True)
            
            print()  # New line after response
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'exit' to quit.")


def run_all_examples():
    """Run all streaming examples in sequence."""
    examples = [
        ("Basic LLM Streaming", stream_basic_llm),
        ("Chain Streaming", stream_with_chain),
        ("Multiple Prompts Streaming", stream_multiple_prompts),
        ("Custom Callback Streaming", stream_with_custom_callback),
    ]
    
    for title, func in examples:
        try:
            func()
            print()
        except Exception as e:
            print(f"❌ Error in {title}: {str(e)}\n")
    
    print("=" * 60)
    print("✅ All streaming examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Streaming Responses Example")
    parser.add_argument(
        "--mode",
        choices=["all", "basic", "chain", "multiple", "custom", "interactive"],
        default="all",
        help="Which streaming example to run"
    )
    
    args = parser.parse_args()
    
    mode_map = {
        "all": run_all_examples,
        "basic": stream_basic_llm,
        "chain": stream_with_chain,
        "multiple": stream_multiple_prompts,
        "custom": stream_with_custom_callback,
        "interactive": interactive_streaming_chat,
    }
    
    mode_map[args.mode]()
