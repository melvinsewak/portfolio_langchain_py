#!/usr/bin/env python3
"""
Validation script to check the structure and configuration of examples.
This runs without requiring langchain dependencies to be installed.
"""
import os
import sys


def check_file_exists(filepath, description):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists


def check_directory_structure():
    """Validate the directory structure."""
    print("=" * 60)
    print("🔍 Checking Directory Structure")
    print("=" * 60)
    
    checks = [
        ("README.md", "README file"),
        ("requirements.txt", "Requirements file"),
        (".env.example", "Environment example file"),
        (".gitignore", "Git ignore file"),
        ("examples/", "Examples directory"),
        ("utils/", "Utils directory"),
        ("examples/__init__.py", "Examples package init"),
        ("utils/__init__.py", "Utils package init"),
        ("utils/config.py", "Configuration module"),
        ("examples/01_conversational_chatbot.py", "Conversational chatbot example"),
        ("examples/02_rag_qa_system.py", "RAG Q&A system example"),
        ("examples/03_multi_tool_agent.py", "Multi-tool agent example"),
        ("examples/04_streaming_responses.py", "Streaming responses example"),
    ]
    
    all_passed = True
    for filepath, description in checks:
        if not check_file_exists(filepath, description):
            all_passed = False
    
    print()
    return all_passed


def check_env_example():
    """Check .env.example file contents."""
    print("=" * 60)
    print("🔧 Checking Environment Configuration")
    print("=" * 60)
    
    required_vars = [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "TEMPERATURE",
        "MAX_TOKENS",
        "VERBOSE",
        "CHROMA_PERSIST_DIRECTORY",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
    ]
    
    try:
        with open(".env.example", "r") as f:
            content = f.read()
        
        all_present = True
        for var in required_vars:
            present = var in content
            status = "✅" if present else "❌"
            print(f"{status} {var}")
            if not present:
                all_present = False
        
        print()
        return all_present
    except Exception as e:
        print(f"❌ Error reading .env.example: {e}\n")
        return False


def check_requirements():
    """Check requirements.txt file contents."""
    print("=" * 60)
    print("📦 Checking Requirements")
    print("=" * 60)
    
    required_packages = [
        "langchain",
        "langchain-community",
        "langchain-openai",
        "chromadb",
        "faiss-cpu",
        "pypdf",
        "python-dotenv",
        "tiktoken",
    ]
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        all_present = True
        for package in required_packages:
            present = package in content
            status = "✅" if present else "❌"
            print(f"{status} {package}")
            if not present:
                all_present = False
        
        print()
        return all_present
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}\n")
        return False


def check_example_structure(filepath):
    """Check if example file has proper structure."""
    try:
        with open(filepath, "r") as f:
            content = f.read()
        
        checks = {
            "Has docstring": '"""' in content[:500],
            "Imports utils": "from utils import" in content or "import utils" in content,
            "Has main block": 'if __name__ == "__main__"' in content,
            "Has argparse": "import argparse" in content or "argparse" in content,
            "Has interactive mode": "interactive" in content.lower(),
        }
        
        all_passed = True
        for check, passed in checks.items():
            if not passed:
                all_passed = False
            status = "  ✅" if passed else "  ❌"
            print(f"{status} {check}")
        
        return all_passed
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False


def check_all_examples():
    """Check structure of all example files."""
    print("=" * 60)
    print("📝 Checking Example Files Structure")
    print("=" * 60)
    
    examples = [
        "examples/01_conversational_chatbot.py",
        "examples/02_rag_qa_system.py",
        "examples/03_multi_tool_agent.py",
        "examples/04_streaming_responses.py",
    ]
    
    all_passed = True
    for example in examples:
        print(f"\n📄 {os.path.basename(example)}:")
        if not check_example_structure(example):
            all_passed = False
    
    print()
    return all_passed


def check_readme():
    """Check README.md has necessary sections."""
    print("=" * 60)
    print("📖 Checking README")
    print("=" * 60)
    
    required_sections = [
        "Features",
        "Getting Started",
        "Installation",
        "Running Examples",
        "Configuration",
        "Example Usage",
        "Project Structure",
    ]
    
    try:
        with open("README.md", "r") as f:
            content = f.read()
        
        all_present = True
        for section in required_sections:
            present = section in content
            status = "✅" if present else "❌"
            print(f"{status} {section} section")
            if not present:
                all_present = False
        
        print()
        return all_present
    except Exception as e:
        print(f"❌ Error reading README.md: {e}\n")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("🧪 Portfolio LangChain Python - Structure Validation")
    print("=" * 60)
    print()
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Environment Variables", check_env_example),
        ("Requirements", check_requirements),
        ("Example Files", check_all_examples),
        ("README Documentation", check_readme),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            passed = check_func()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ Error in {name}: {e}\n")
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("📊 Validation Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All validation checks passed!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env")
        print("2. Add your OPENAI_API_KEY to .env")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Run examples: python examples/01_conversational_chatbot.py --mode example")
        return 0
    else:
        print("\n⚠️  Some validation checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
