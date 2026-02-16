"""
Agent with External API Tools Example using LangChain.

This example demonstrates an agent using external API tools:
- Weather API integration
- HTTP request tool
- JSON API parser
- Error handling for API calls
- Environment-based configuration

Features:
- Weather lookup tool (OpenWeatherMap simulation)
- Generic HTTP GET tool
- JSON parsing and validation
- Rate limiting awareness
- API authentication patterns
"""
import sys
import os
import json
import requests
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from utils import get_config


# Tool Input Schemas
class WeatherInput(BaseModel):
    """Input for weather lookup tool."""
    city: str = Field(description="City name to get weather for")


class HTTPGetInput(BaseModel):
    """Input for HTTP GET request tool."""
    url: str = Field(description="URL to fetch")
    headers: Optional[dict] = Field(default=None, description="Optional HTTP headers")


class JSONParseInput(BaseModel):
    """Input for JSON parsing tool."""
    json_string: str = Field(description="JSON string to parse")


def create_weather_tool() -> StructuredTool:
    """
    Create a weather lookup tool.
    
    Note: This is a simulated version. In production, use a real API like OpenWeatherMap.
    """
    def get_weather(city: str) -> str:
        """
        Get weather information for a city.
        
        In production, this would call a real weather API:
        ```
        api_key = os.getenv("WEATHER_API_KEY")
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        response = requests.get(url)
        return response.json()
        ```
        """
        # Simulated weather data
        weather_data = {
            "london": {"temp": 15, "condition": "Cloudy", "humidity": 75},
            "new york": {"temp": 22, "condition": "Sunny", "humidity": 60},
            "tokyo": {"temp": 18, "condition": "Rainy", "humidity": 85},
            "paris": {"temp": 17, "condition": "Partly cloudy", "humidity": 70},
            "sydney": {"temp": 25, "condition": "Sunny", "humidity": 55},
        }
        
        city_lower = city.lower()
        
        if city_lower in weather_data:
            data = weather_data[city_lower]
            return json.dumps({
                "city": city,
                "temperature": f"{data['temp']}°C",
                "condition": data['condition'],
                "humidity": f"{data['humidity']}%",
                "source": "simulated_api"
            }, indent=2)
        else:
            return json.dumps({
                "error": f"Weather data not available for {city}",
                "suggestion": "Try: London, New York, Tokyo, Paris, or Sydney"
            })
    
    return StructuredTool.from_function(
        func=get_weather,
        name="weather_lookup",
        description="Get current weather information for a city. Returns temperature, condition, and humidity.",
        args_schema=WeatherInput
    )


def create_http_get_tool() -> StructuredTool:
    """
    Create an HTTP GET request tool.
    
    Note: Be careful with this in production - validate URLs and implement rate limiting.
    """
    def http_get(url: str, headers: Optional[dict] = None) -> str:
        """
        Perform HTTP GET request.
        
        Args:
            url: URL to fetch
            headers: Optional HTTP headers
            
        Returns:
            Response content or error message
        """
        try:
            # In production, add timeout and validate URL
            # Also implement rate limiting and caching
            
            # For safety, only allow specific domains in production
            allowed_domains = ["api.github.com", "jsonplaceholder.typicode.com", "httpbin.org"]
            
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            
            if domain not in allowed_domains:
                return json.dumps({
                    "error": "Domain not in allowlist",
                    "allowed_domains": allowed_domains,
                    "note": "This is for security. Modify allowed_domains for your use case."
                })
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Try to return JSON if possible
            try:
                return json.dumps(response.json(), indent=2)
            except:
                return response.text[:500]  # Limit response size
                
        except requests.RequestException as e:
            return json.dumps({"error": str(e)})
    
    return StructuredTool.from_function(
        func=http_get,
        name="http_get",
        description="Perform HTTP GET request to fetch data from APIs. Only works with allowed domains for security.",
        args_schema=HTTPGetInput
    )


def create_json_parser_tool() -> StructuredTool:
    """Create a JSON parsing and validation tool."""
    def parse_json(json_string: str) -> str:
        """Parse and validate JSON string."""
        try:
            parsed = json.loads(json_string)
            
            # Return formatted and validated JSON
            return json.dumps({
                "success": True,
                "type": type(parsed).__name__,
                "keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                "length": len(parsed) if isinstance(parsed, (list, dict)) else None,
                "data": parsed
            }, indent=2)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": "Invalid JSON format"
            })
    
    return StructuredTool.from_function(
        func=parse_json,
        name="json_parser",
        description="Parse and validate JSON strings. Returns structured information about the JSON data.",
        args_schema=JSONParseInput
    )


def create_github_info_tool() -> Tool:
    """Create a GitHub repository info tool."""
    def get_github_repo_info(repo: str) -> str:
        """
        Get information about a GitHub repository.
        
        Args:
            repo: Repository in format 'owner/repo'
            
        Returns:
            Repository information
        """
        try:
            url = f"https://api.github.com/repos/{repo}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return json.dumps({
                "name": data.get("name"),
                "description": data.get("description"),
                "stars": data.get("stargazers_count"),
                "forks": data.get("forks_count"),
                "language": data.get("language"),
                "open_issues": data.get("open_issues_count"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return Tool(
        name="github_repo_info",
        func=get_github_repo_info,
        description="Get information about a GitHub repository. Input should be 'owner/repo' format (e.g., 'langchain-ai/langchain')."
    )


def create_api_agent():
    """
    Create an agent with external API tools.
    
    Returns:
        AgentExecutor: Configured agent with API tools
    """
    config = get_config()
    
    if not config.validate():
        print("❌ Configuration validation failed.")
        sys.exit(1)
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        openai_api_key=config.openai_api_key
    )
    
    # Create tools
    tools = [
        create_weather_tool(),
        create_http_get_tool(),
        create_json_parser_tool(),
        create_github_info_tool(),
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to external API tools.

Available tools:
- weather_lookup: Get weather information for cities
- http_get: Fetch data from allowed APIs
- json_parser: Parse and validate JSON data
- github_repo_info: Get GitHub repository information

When using APIs:
1. Check for errors in responses
2. Parse JSON data appropriately
3. Provide clear summaries of the information
4. Handle rate limits gracefully

Be helpful and informative in your responses."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=config.verbose,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent_executor


def run_interactive_agent():
    """Run interactive session with API agent."""
    print("=" * 60)
    print("🌐 Agent with External API Tools")
    print("=" * 60)
    print("\nThis agent can:")
    print("  ☀️  Check weather for cities")
    print("  🌐 Fetch data from APIs")
    print("  📦 Get GitHub repository info")
    print("  📄 Parse JSON data")
    print("\nType 'exit', 'quit', or 'bye' to end.\n")
    print("=" * 60)
    
    agent_executor = create_api_agent()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            print("\n🤖 Agent: ", end="", flush=True)
            response = agent_executor.invoke({"input": user_input})
            print(response['output'])
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def run_example_tasks():
    """Run pre-defined example tasks."""
    print("=" * 60)
    print("📝 Running Example API Tasks")
    print("=" * 60)
    
    agent_executor = create_api_agent()
    
    tasks = [
        "What's the weather like in London?",
        "Get information about the langchain-ai/langchain GitHub repository",
        "Compare the weather in Tokyo and Sydney",
        "Parse this JSON: {\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}",
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
    
    parser = argparse.ArgumentParser(description="External API Tools Agent Example")
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
