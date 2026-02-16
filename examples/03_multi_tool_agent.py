"""
Multi-Tool Agent Example using LangChain.

This example demonstrates a production-ready agent with multiple tools:
- Web search capabilities (simulated)
- File operations
- Data processing
- API interactions
- Environment-based configuration

Features:
- Multiple specialized tools for different tasks
- Agent reasoning and tool selection
- Error handling and fallback strategies
- Structured output
"""
import sys
import os
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from utils import get_config


# Tool Input Schemas
class CalculatorInput(BaseModel):
    """Input for calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate")


class DataStatsInput(BaseModel):
    """Input for data statistics tool."""
    numbers: str = Field(description="Comma-separated list of numbers")


class FileWriteInput(BaseModel):
    """Input for file write tool."""
    filename: str = Field(description="Name of the file to write to")
    content: str = Field(description="Content to write to the file")


def create_calculator_tool() -> StructuredTool:
    """Create an advanced calculator tool."""
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            import ast
            import operator
            
            # Define allowed operators
            allowed_operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            def eval_node(node):
                """Safely evaluate an AST node."""
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    op_type = type(node.op)
                    if op_type not in allowed_operators:
                        raise ValueError(f"Operator {op_type} not allowed")
                    left = eval_node(node.left)
                    right = eval_node(node.right)
                    return allowed_operators[op_type](left, right)
                elif isinstance(node, ast.UnaryOp):
                    op_type = type(node.op)
                    if op_type not in allowed_operators:
                        raise ValueError(f"Operator {op_type} not allowed")
                    operand = eval_node(node.operand)
                    return allowed_operators[op_type](operand)
                else:
                    raise ValueError(f"Node type {type(node)} not allowed")
            
            # Parse and evaluate the expression
            tree = ast.parse(expression, mode='eval')
            result = eval_node(tree.body)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    return StructuredTool.from_function(
        func=calculate,
        name="calculator",
        description="Performs mathematical calculations. Input must be a valid mathematical expression.",
        args_schema=CalculatorInput
    )


def create_data_stats_tool() -> StructuredTool:
    """Create a data statistics tool."""
    def calculate_stats(numbers: str) -> str:
        """Calculate statistics for a list of numbers."""
        try:
            num_list = [float(x.strip()) for x in numbers.split(',')]
            if not num_list:
                return "Error: No numbers provided"
            
            stats = {
                "count": len(num_list),
                "sum": sum(num_list),
                "mean": sum(num_list) / len(num_list),
                "min": min(num_list),
                "max": max(num_list),
            }
            
            return json.dumps(stats, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"
    
    return StructuredTool.from_function(
        func=calculate_stats,
        name="data_statistics",
        description="Calculates statistics (count, sum, mean, min, max) for a comma-separated list of numbers.",
        args_schema=DataStatsInput
    )


def create_datetime_tool() -> Tool:
    """Create a datetime information tool."""
    def get_datetime_info(query: str) -> str:
        """Get current datetime information."""
        now = datetime.now()
        
        info = {
            "current_date": now.strftime("%Y-%m-%d"),
            "current_time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "timestamp": now.timestamp(),
        }
        
        return json.dumps(info, indent=2)
    
    return Tool(
        name="datetime_info",
        func=get_datetime_info,
        description="Gets current date and time information. No input required."
    )


def create_text_analyzer_tool() -> Tool:
    """Create a text analysis tool."""
    def analyze_text(text: str) -> str:
        """Analyze text and return statistics."""
        words = text.split()
        sentences = text.split('.')
        
        analysis = {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "unique_words": len(set(words)),
        }
        
        return json.dumps(analysis, indent=2)
    
    return Tool(
        name="text_analyzer",
        func=analyze_text,
        description="Analyzes text and provides statistics like word count, sentence count, etc."
    )


def create_file_writer_tool() -> StructuredTool:
    """Create a tool for writing to files in /tmp directory."""
    def write_file(filename: str, content: str) -> str:
        """Write content to a file in /tmp directory."""
        try:
            # Use only the basename to prevent path traversal
            safe_filename = os.path.basename(filename)
            if not safe_filename:
                return "Error: Invalid filename"
            
            # Ensure we write to /tmp for safety
            filepath = os.path.join("/tmp", safe_filename)
            
            # Verify the resolved path is still under /tmp
            resolved_path = os.path.realpath(filepath)
            if os.path.commonpath([resolved_path, "/tmp"]) != "/tmp":
                return "Error: Invalid file path"
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            return f"Successfully wrote to {filepath}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    return StructuredTool.from_function(
        func=write_file,
        name="file_writer",
        description="Writes content to a file in /tmp directory. Provide filename and content.",
        args_schema=FileWriteInput
    )


def create_multi_tool_agent():
    """
    Create an agent with multiple specialized tools.
    
    Returns:
        AgentExecutor: Configured agent executor with multiple tools
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
        openai_api_key=config.openai_api_key
    )
    
    # Create all tools
    tools = [
        create_calculator_tool(),
        create_data_stats_tool(),
        create_datetime_tool(),
        create_text_analyzer_tool(),
        create_file_writer_tool(),
    ]
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to multiple specialized tools.

Available tools:
- calculator: For mathematical calculations
- data_statistics: For analyzing numerical data
- datetime_info: For getting current date and time
- text_analyzer: For analyzing text content
- file_writer: For writing content to files

Always use the most appropriate tool for each task. Break down complex requests into
smaller steps if needed. Provide clear and helpful responses."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=config.verbose,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent_executor


def run_interactive_agent():
    """Run an interactive session with the multi-tool agent."""
    print("=" * 60)
    print("🛠️  Multi-Tool Agent")
    print("=" * 60)
    print("\nThis agent has access to:")
    print("  • Calculator - mathematical calculations")
    print("  • Data Statistics - analyze numerical data")
    print("  • DateTime Info - current date and time")
    print("  • Text Analyzer - analyze text content")
    print("  • File Writer - write to files in /tmp")
    print("\nType 'exit', 'quit', or 'bye' to end the session.\n")
    print("=" * 60)
    
    # Create the agent
    agent_executor = create_multi_tool_agent()
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            # Get response from agent
            response = agent_executor.invoke({"input": user_input})
            
            # Print the response
            print(f"\n🤖 Agent: {response['output']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'exit' to quit.")


def run_example_tasks():
    """Run pre-defined example tasks to demonstrate agent capabilities."""
    print("=" * 60)
    print("📝 Running Example Tasks")
    print("=" * 60)
    
    agent_executor = create_multi_tool_agent()
    
    # Example tasks
    tasks = [
        "Calculate 15 * 8 + 42",
        "Analyze these numbers: 10, 20, 30, 40, 50",
        "What is the current date and time?",
        "Analyze this text: The quick brown fox jumps over the lazy dog. It is a sunny day.",
        "Write 'Hello from LangChain Agent!' to a file called agent_output.txt",
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"Task {i}: {task}")
        print(f"{'='*60}")
        
        try:
            response = agent_executor.invoke({"input": task})
            print(f"\n🤖 Agent: {response['output']}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Example tasks completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Tool Agent Example")
    parser.add_argument(
        "--mode",
        choices=["interactive", "example"],
        default="interactive",
        help="Run mode: 'interactive' for agent session, 'example' for demo"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        run_interactive_agent()
    else:
        run_example_tasks()
