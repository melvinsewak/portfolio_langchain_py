"""
Advanced LangGraph Patterns Example using LangChain.

This example demonstrates advanced multi-agent patterns with LangGraph:
- Parallel agent execution
- Conditional routing with multiple paths
- Dynamic workflow based on state
- Complex agent coordination
- Environment-based configuration

Features:
- Parallel task execution
- Conditional branching based on results
- Dynamic path selection
- Agent specialization
- State management across branches
"""
import sys
import os
from typing import TypedDict, Annotated, Sequence, Literal
import operator

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from utils import get_config


# Define workflow state
class WorkflowState(TypedDict):
    """State for the advanced workflow."""
    task: str
    messages: Annotated[Sequence[str], operator.add]
    analysis_result: str
    technical_review: str
    business_review: str
    final_decision: str
    decision_path: str
    parallel_results: dict


def create_analyzer_agent():
    """Create an analyzer agent that evaluates the task."""
    config = get_config()
    
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=0.2,
        openai_api_key=config.openai_api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an analyzer agent. Analyze the given task and determine:
1. Is it technical or business-focused?
2. What's the complexity level (simple/medium/complex)?
3. What type of review is needed?

Respond with: TYPE: [technical/business/both], COMPLEXITY: [simple/medium/complex]"""),
        ("human", "{input}"),
    ])
    
    def analyze(state: WorkflowState) -> WorkflowState:
        """Analyze the task."""
        task = state["task"]
        messages = state.get("messages", [])
        
        print(f"\n🔍 Analyzer Agent processing: {task}")
        
        chain = prompt | llm
        result = chain.invoke({"input": task})
        
        state["analysis_result"] = result.content
        state["messages"] = messages + ["Analyzer: Completed analysis"]
        
        # Determine path based on analysis
        if "TYPE: technical" in result.content:
            state["decision_path"] = "technical"
        elif "TYPE: business" in result.content:
            state["decision_path"] = "business"
        else:
            state["decision_path"] = "both"
        
        print(f"  ✅ Analysis complete. Path: {state['decision_path']}")
        
        return state
    
    return analyze


def create_technical_reviewer():
    """Create a technical review agent."""
    config = get_config()
    
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=0.3,
        openai_api_key=config.openai_api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical review specialist. Review technical aspects:
- Feasibility
- Technical risks
- Implementation approach
- Technology stack recommendations"""),
        ("human", "Task: {task}\nAnalysis: {analysis}"),
    ])
    
    def review(state: WorkflowState) -> WorkflowState:
        """Perform technical review."""
        messages = state.get("messages", [])
        
        print("\n⚙️  Technical Reviewer processing...")
        
        chain = prompt | llm
        result = chain.invoke({
            "task": state["task"],
            "analysis": state["analysis_result"]
        })
        
        state["technical_review"] = result.content
        state["messages"] = messages + ["Technical Reviewer: Completed review"]
        
        print(f"  ✅ Technical review complete")
        
        return state
    
    return review


def create_business_reviewer():
    """Create a business review agent."""
    config = get_config()
    
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=0.3,
        openai_api_key=config.openai_api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a business review specialist. Review business aspects:
- Market viability
- Cost-benefit analysis
- ROI potential
- Business risks"""),
        ("human", "Task: {task}\nAnalysis: {analysis}"),
    ])
    
    def review(state: WorkflowState) -> WorkflowState:
        """Perform business review."""
        messages = state.get("messages", [])
        
        print("\n💼 Business Reviewer processing...")
        
        chain = prompt | llm
        result = chain.invoke({
            "task": state["task"],
            "analysis": state["analysis_result"]
        })
        
        state["business_review"] = result.content
        state["messages"] = messages + ["Business Reviewer: Completed review"]
        
        print(f"  ✅ Business review complete")
        
        return state
    
    return review


def create_parallel_processor():
    """Create agents that run in parallel."""
    config = get_config()
    
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=0.4,
        openai_api_key=config.openai_api_key
    )
    
    def process_both(state: WorkflowState) -> WorkflowState:
        """Process both technical and business reviews in parallel."""
        print("\n🔀 Running parallel reviews...")
        
        # In a real scenario, these could run truly in parallel
        # For demonstration, we'll call them sequentially but show the pattern
        
        tech_prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide a brief technical assessment."),
            ("human", "{task}"),
        ])
        
        biz_prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide a brief business assessment."),
            ("human", "{task}"),
        ])
        
        tech_result = (tech_prompt | llm).invoke({"task": state["task"]})
        biz_result = (biz_prompt | llm).invoke({"task": state["task"]})
        
        state["technical_review"] = tech_result.content
        state["business_review"] = biz_result.content
        state["parallel_results"] = {
            "technical": tech_result.content,
            "business": biz_result.content
        }
        
        messages = state.get("messages", [])
        state["messages"] = messages + ["Parallel Processor: Completed both reviews"]
        
        print("  ✅ Parallel processing complete")
        
        return state
    
    return process_both


def create_decision_maker():
    """Create a decision-making agent."""
    config = get_config()
    
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=0.5,
        openai_api_key=config.openai_api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a decision maker. Based on the reviews provided, make a final decision.
Provide a clear recommendation: APPROVE, REJECT, or NEEDS_REVISION with reasoning."""),
        ("human", """Task: {task}
Technical Review: {technical}
Business Review: {business}
Make your decision."""),
    ])
    
    def decide(state: WorkflowState) -> WorkflowState:
        """Make final decision."""
        messages = state.get("messages", [])
        
        print("\n🎯 Decision Maker processing...")
        
        chain = prompt | llm
        result = chain.invoke({
            "task": state["task"],
            "technical": state.get("technical_review", "N/A"),
            "business": state.get("business_review", "N/A")
        })
        
        state["final_decision"] = result.content
        state["messages"] = messages + ["Decision Maker: Final decision made"]
        
        print(f"  ✅ Decision complete")
        
        return state
    
    return decide


def route_after_analysis(state: WorkflowState) -> Literal["technical", "business", "both"]:
    """Route to appropriate review based on analysis."""
    return state["decision_path"]


def create_advanced_workflow():
    """
    Create an advanced workflow with parallel execution and conditional routing.
    
    Returns:
        Compiled workflow graph
    """
    # Create agents
    analyzer = create_analyzer_agent()
    technical_reviewer = create_technical_reviewer()
    business_reviewer = create_business_reviewer()
    parallel_processor = create_parallel_processor()
    decision_maker = create_decision_maker()
    
    # Create workflow
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("analyzer", analyzer)
    workflow.add_node("technical_review", technical_reviewer)
    workflow.add_node("business_review", business_reviewer)
    workflow.add_node("parallel_review", parallel_processor)
    workflow.add_node("decision", decision_maker)
    
    # Set entry point
    workflow.set_entry_point("analyzer")
    
    # Add conditional routing after analysis
    workflow.add_conditional_edges(
        "analyzer",
        route_after_analysis,
        {
            "technical": "technical_review",
            "business": "business_review",
            "both": "parallel_review"
        }
    )
    
    # Route all paths to decision maker
    workflow.add_edge("technical_review", "decision")
    workflow.add_edge("business_review", "decision")
    workflow.add_edge("parallel_review", "decision")
    
    # End after decision
    workflow.add_edge("decision", END)
    
    # Compile the workflow
    app = workflow.compile()
    
    return app


def run_workflow_example(task: str):
    """
    Execute a task through the advanced workflow.
    
    Args:
        task: Task description
    """
    print("=" * 60)
    print("🔀 Advanced LangGraph Workflow")
    print("=" * 60)
    print(f"\n📋 Task: {task}\n")
    
    # Create workflow
    app = create_advanced_workflow()
    
    # Initialize state
    initial_state = {
        "task": task,
        "messages": [],
        "analysis_result": "",
        "technical_review": "",
        "business_review": "",
        "final_decision": "",
        "decision_path": "",
        "parallel_results": {}
    }
    
    # Execute workflow
    print("🚀 Starting workflow...\n")
    result = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 Workflow Results")
    print("=" * 60)
    
    print(f"\n🔍 Analysis Result:")
    print(f"  {result['analysis_result'][:200]}...")
    
    print(f"\n📍 Decision Path: {result['decision_path']}")
    
    if result.get('technical_review'):
        print(f"\n⚙️  Technical Review:")
        print(f"  {result['technical_review'][:200]}...")
    
    if result.get('business_review'):
        print(f"\n💼 Business Review:")
        print(f"  {result['business_review'][:200]}...")
    
    print(f"\n🎯 Final Decision:")
    print(f"  {result['final_decision']}")
    
    print(f"\n📝 Workflow Messages:")
    for i, msg in enumerate(result['messages'], 1):
        print(f"  {i}. {msg}")
    
    print("\n" + "=" * 60)


def run_interactive_workflow():
    """Run interactive workflow session."""
    print("=" * 60)
    print("🔀 Interactive Advanced Workflow")
    print("=" * 60)
    print("\nThis workflow demonstrates:")
    print("  🔍 Conditional routing based on analysis")
    print("  🔀 Parallel execution for complex tasks")
    print("  📊 Multi-path decision making")
    print("\nType 'exit', 'quit', or 'bye' to end.\n")
    print("=" * 60)
    
    config = get_config()
    if not config.validate():
        print("❌ Configuration validation failed.")
        sys.exit(1)
    
    while True:
        try:
            task = input("\n📋 Enter a task/proposal: ").strip()
            
            if not task:
                continue
            
            if task.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            run_workflow_example(task)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def run_example_workflows():
    """Run pre-defined example workflows."""
    print("=" * 60)
    print("📝 Running Example Workflows")
    print("=" * 60)
    
    config = get_config()
    if not config.validate():
        print("❌ Configuration validation failed.")
        sys.exit(1)
    
    examples = [
        "Build a new machine learning model to predict customer churn",
        "Create a mobile app for employee onboarding",
        "Implement a new API endpoint for user authentication",
    ]
    
    for i, task in enumerate(examples, 1):
        print(f"\n\n{'#' * 60}")
        print(f"Example {i} of {len(examples)}")
        print(f"{'#' * 60}\n")
        
        try:
            run_workflow_example(task)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced LangGraph Patterns Example")
    parser.add_argument(
        "--mode",
        choices=["interactive", "example"],
        default="example",
        help="Run mode: 'interactive' for custom tasks, 'example' for demo"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        run_interactive_workflow()
    else:
        run_example_workflows()
