"""
Multi-Agent System Example using LangChain.

This example demonstrates a production-ready multi-agent system with:
- Multiple specialized agents working together
- Supervisor agent for task coordination
- Agent-to-agent communication
- Collaborative problem solving
- Environment-based configuration

Features:
- Research agent for information gathering
- Writer agent for content creation
- Critic agent for quality review
- Supervisor agent for orchestration
- Sequential and parallel agent execution
"""
import sys
import os
from typing import TypedDict, Annotated, Sequence
import operator

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from utils import get_config


# Define the state for multi-agent workflow
class AgentState(TypedDict):
    """State passed between agents."""
    messages: Annotated[Sequence[str], operator.add]
    task: str
    research_output: str
    draft_output: str
    final_output: str
    next_agent: str


def create_research_agent():
    """
    Create a research agent that gathers information.
    
    Returns:
        Function that executes research tasks
    """
    config = get_config()
    
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=0.3,  # Lower temperature for research
        openai_api_key=config.openai_api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research specialist agent. Your role is to gather and analyze information.
        
When given a topic or question:
1. Break down the topic into key points
2. Provide factual, well-researched information
3. Cite relevant concepts and ideas
4. Organize information logically

Be thorough and accurate in your research."""),
        ("human", "{input}"),
    ])
    
    def research(state: AgentState) -> AgentState:
        """Execute research task."""
        task = state["task"]
        messages = state.get("messages", [])
        
        print(f"\n🔍 Research Agent working on: {task}")
        
        chain = prompt | llm
        result = chain.invoke({"input": task})
        
        state["research_output"] = result.content
        state["messages"] = messages + [f"Research Agent: Completed research on '{task}'"]
        state["next_agent"] = "writer"
        
        print(f"✅ Research completed ({len(result.content)} characters)")
        
        return state
    
    return research


def create_writer_agent():
    """
    Create a writer agent that creates content based on research.
    
    Returns:
        Function that executes writing tasks
    """
    config = get_config()
    
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=0.7,  # Higher temperature for creative writing
        openai_api_key=config.openai_api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional writer agent. Your role is to create engaging content.
        
When given research material:
1. Transform information into well-structured content
2. Write in a clear, engaging style
3. Ensure proper flow and coherence
4. Include relevant examples and explanations

Create content that is informative and accessible."""),
        ("human", "Based on this research:\n\n{research}\n\nCreate content about: {task}"),
    ])
    
    def write(state: AgentState) -> AgentState:
        """Execute writing task."""
        task = state["task"]
        research = state.get("research_output", "")
        messages = state.get("messages", [])
        
        print(f"\n✍️  Writer Agent creating content...")
        
        chain = prompt | llm
        result = chain.invoke({"research": research, "task": task})
        
        state["draft_output"] = result.content
        state["messages"] = messages + [f"Writer Agent: Created draft content"]
        state["next_agent"] = "critic"
        
        print(f"✅ Draft completed ({len(result.content)} characters)")
        
        return state
    
    return write


def create_critic_agent():
    """
    Create a critic agent that reviews and improves content.
    
    Returns:
        Function that executes review tasks
    """
    config = get_config()
    
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=0.4,  # Balanced temperature for critique
        openai_api_key=config.openai_api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a quality assurance critic agent. Your role is to review and improve content.
        
When reviewing content:
1. Check for accuracy and completeness
2. Identify areas for improvement
3. Suggest enhancements
4. Ensure clarity and readability
5. Provide a polished final version

Be constructive and thorough in your review."""),
        ("human", "Review and improve this content:\n\n{draft}\n\nOriginal task: {task}"),
    ])
    
    def critique(state: AgentState) -> AgentState:
        """Execute critique and improvement task."""
        task = state["task"]
        draft = state.get("draft_output", "")
        messages = state.get("messages", [])
        
        print(f"\n🔎 Critic Agent reviewing content...")
        
        chain = prompt | llm
        result = chain.invoke({"draft": draft, "task": task})
        
        state["final_output"] = result.content
        state["messages"] = messages + [f"Critic Agent: Reviewed and finalized content"]
        state["next_agent"] = "end"
        
        print(f"✅ Review completed ({len(result.content)} characters)")
        
        return state
    
    return critique


def create_multi_agent_system():
    """
    Create a multi-agent system with supervisor orchestration.
    
    Returns:
        Compiled workflow graph
    """
    # Create agents
    research_agent = create_research_agent()
    writer_agent = create_writer_agent()
    critic_agent = create_critic_agent()
    
    # Create workflow graph
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("researcher", research_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("critic", critic_agent)
    
    # Define routing logic
    def route_agent(state: AgentState) -> str:
        """Route to next agent based on state."""
        next_agent = state.get("next_agent", "end")
        if next_agent == "end":
            return END
        return next_agent
    
    # Add edges
    workflow.set_entry_point("researcher")
    workflow.add_conditional_edges(
        "researcher",
        route_agent,
        {
            "writer": "writer",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "writer",
        route_agent,
        {
            "critic": "critic",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "critic",
        route_agent,
        {
            END: END
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def run_multi_agent_task(task: str):
    """
    Execute a task using the multi-agent system.
    
    Args:
        task: The task description
    """
    print("=" * 60)
    print("🤖 Multi-Agent System")
    print("=" * 60)
    print(f"\n📋 Task: {task}\n")
    
    # Create the multi-agent system
    app = create_multi_agent_system()
    
    # Initialize state
    initial_state = {
        "messages": [],
        "task": task,
        "research_output": "",
        "draft_output": "",
        "final_output": "",
        "next_agent": "researcher"
    }
    
    # Execute the workflow
    print("🚀 Starting multi-agent workflow...\n")
    result = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 Workflow Summary")
    print("=" * 60)
    
    print(f"\n📝 Messages exchanged:")
    for i, msg in enumerate(result["messages"], 1):
        print(f"  {i}. {msg}")
    
    print(f"\n" + "=" * 60)
    print("🎯 Final Output")
    print("=" * 60)
    print(f"\n{result['final_output']}\n")
    
    return result


def run_interactive_multi_agent():
    """Run interactive multi-agent session."""
    print("=" * 60)
    print("🤖 Interactive Multi-Agent System")
    print("=" * 60)
    print("\nThis system uses three specialized agents:")
    print("  🔍 Research Agent - Gathers information")
    print("  ✍️  Writer Agent - Creates content")
    print("  🔎 Critic Agent - Reviews and improves")
    print("\nType 'exit', 'quit', or 'bye' to end.\n")
    print("=" * 60)
    
    # Validate config
    config = get_config()
    if not config.validate():
        print("❌ Configuration validation failed. Please check your .env file.")
        sys.exit(1)
    
    while True:
        try:
            task = input("\n📋 Enter a task for the agents: ").strip()
            
            if not task:
                continue
            
            if task.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            # Execute the task
            run_multi_agent_task(task)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'exit' to quit.")


def run_example_tasks():
    """Run pre-defined example tasks."""
    print("=" * 60)
    print("📝 Running Example Multi-Agent Tasks")
    print("=" * 60)
    
    # Validate config
    config = get_config()
    if not config.validate():
        print("❌ Configuration validation failed. Please check your .env file.")
        sys.exit(1)
    
    # Example tasks
    tasks = [
        "Explain the benefits of multi-agent systems in AI",
        "Write a brief guide on using LangChain for beginners",
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n\n{'#' * 60}")
        print(f"Example {i} of {len(tasks)}")
        print(f"{'#' * 60}\n")
        
        try:
            run_multi_agent_task(task)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ All example tasks completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent System Example")
    parser.add_argument(
        "--mode",
        choices=["interactive", "example"],
        default="interactive",
        help="Run mode: 'interactive' for custom tasks, 'example' for demo"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        run_interactive_multi_agent()
    else:
        run_example_tasks()
