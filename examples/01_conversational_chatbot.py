"""
Conversational Chatbot Example using LangChain.

This example demonstrates a production-ready conversational chatbot with:
- AgentExecutor for managing agent execution
- ConversationBufferMemory for maintaining chat history
- Custom tools for the agent to use
- Environment-based configuration

Features:
- Maintains conversation history across multiple interactions
- Uses custom tools (calculator, search)
- Configurable through .env file
- Structured output and error handling
"""
import sys
import os
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from utils import get_config


def create_calculator_tool() -> Tool:
    """Create a simple calculator tool."""
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        try:
            # Only allow safe mathematical operations
            allowed_chars = set('0123456789+-*/()%. ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)
            return f"The result is: {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"
    
    return Tool(
        name="Calculator",
        func=calculate,
        description="Useful for performing mathematical calculations. Input should be a valid mathematical expression."
    )


def create_word_counter_tool() -> Tool:
    """Create a word counter tool."""
    def count_words(text: str) -> str:
        """Count the number of words in a text."""
        words = text.split()
        return f"The text contains {len(words)} words."
    
    return Tool(
        name="WordCounter",
        func=count_words,
        description="Useful for counting words in a text. Input should be the text to count words in."
    )


def create_conversational_agent():
    """
    Create a conversational agent with memory and custom tools.
    
    Returns:
        AgentExecutor: Configured agent executor ready for conversation
    """
    # Load configuration
    config = get_config()
    
    if not config.validate():
        print("❌ Configuration validation failed. Please check your .env file.")
        sys.exit(1)
    
    # Initialize the language model
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        openai_api_key=config.openai_api_key
    )
    
    # Create tools
    tools = [
        create_calculator_tool(),
        create_word_counter_tool(),
    ]
    
    # Create the prompt template with placeholders for chat history and agent scratchpad
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to various tools.
        
You can help users with:
- Mathematical calculations using the Calculator tool
- Counting words in text using the WordCounter tool
- General conversation and questions

Always be friendly, clear, and helpful in your responses."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=config.verbose,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor


def run_interactive_chat():
    """Run an interactive chat session with the agent."""
    print("=" * 60)
    print("🤖 Conversational Chatbot with Memory and Tools")
    print("=" * 60)
    print("\nThis chatbot has access to:")
    print("  • Calculator - for mathematical calculations")
    print("  • Word Counter - for counting words in text")
    print("\nIt maintains conversation history across messages.")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.\n")
    print("=" * 60)
    
    # Create the agent
    agent_executor = create_conversational_agent()
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye! Have a great day!")
                break
            
            # Get response from agent
            response = agent_executor.invoke({"input": user_input})
            
            # Print the response
            print(f"\n🤖 Assistant: {response['output']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'exit' to quit.")


def run_example_conversation():
    """Run a pre-defined example conversation to demonstrate capabilities."""
    print("=" * 60)
    print("📝 Running Example Conversation")
    print("=" * 60)
    
    agent_executor = create_conversational_agent()
    
    # Example conversation
    examples = [
        "Hello! Can you introduce yourself?",
        "What is 25 * 4 + 100?",
        "How many words are in this sentence: The quick brown fox jumps over the lazy dog?",
        "What was my first question?",  # Tests memory
    ]
    
    for i, message in enumerate(examples, 1):
        print(f"\n👤 User: {message}")
        try:
            response = agent_executor.invoke({"input": message})
            print(f"🤖 Assistant: {response['output']}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Example conversation completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Conversational Chatbot Example")
    parser.add_argument(
        "--mode",
        choices=["interactive", "example"],
        default="interactive",
        help="Run mode: 'interactive' for chat session, 'example' for demo"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        run_interactive_chat()
    else:
        run_example_conversation()
