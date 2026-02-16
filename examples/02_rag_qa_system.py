"""
RAG (Retrieval Augmented Generation) Example using LangChain.

This example demonstrates a production-ready RAG system with:
- Document loading and text splitting
- Vector store (Chroma) for efficient similarity search
- OpenAI embeddings for document vectorization
- Question-answering over documents
- Environment-based configuration

Features:
- Load and process documents from various sources
- Store embeddings in a persistent vector database
- Answer questions based on document context
- Configurable chunk size and overlap
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from utils import get_config


def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration."""
    documents = [
        Document(
            page_content="""
            LangChain is a framework for developing applications powered by language models.
            It provides tools and abstractions for working with LLMs, including chains, agents,
            and memory components. LangChain makes it easy to build complex LLM applications
            by providing reusable components and patterns.
            """,
            metadata={"source": "langchain_intro", "topic": "framework"}
        ),
        Document(
            page_content="""
            Retrieval Augmented Generation (RAG) is a technique that enhances language models
            by retrieving relevant information from a knowledge base before generating responses.
            This allows the model to access up-to-date information and domain-specific knowledge
            that wasn't part of its training data. RAG systems typically use vector databases
            for efficient similarity search.
            """,
            metadata={"source": "rag_explanation", "topic": "technique"}
        ),
        Document(
            page_content="""
            Vector databases store embeddings - numerical representations of text that capture
            semantic meaning. They enable fast similarity search, which is crucial for RAG systems.
            Popular vector databases include Chroma, Pinecone, Weaviate, and FAISS. These databases
            use algorithms like HNSW or IVF for approximate nearest neighbor search.
            """,
            metadata={"source": "vector_db_info", "topic": "database"}
        ),
        Document(
            page_content="""
            OpenAI provides powerful embedding models like text-embedding-ada-002, which convert
            text into high-dimensional vectors. These embeddings capture semantic relationships
            between words and concepts, making them ideal for similarity search and retrieval tasks.
            The embeddings are used in RAG systems to find relevant documents.
            """,
            metadata={"source": "embeddings_info", "topic": "embeddings"}
        ),
        Document(
            page_content="""
            Production RAG systems should consider several factors: chunk size affects context
            quality, overlap ensures continuity between chunks, and the number of retrieved
            documents balances context and cost. Monitoring retrieval quality and implementing
            fallback strategies are also important for production deployments.
            """,
            metadata={"source": "production_tips", "topic": "best_practices"}
        ),
    ]
    return documents


def setup_rag_system(documents: List[Document] = None):
    """
    Set up a RAG system with vector store and retrieval chain.
    
    Args:
        documents: Optional list of documents. If None, uses sample documents.
    
    Returns:
        RetrievalQA: Configured RAG chain ready for question answering
    """
    # Load configuration
    config = get_config()
    
    if not config.validate():
        print("❌ Configuration validation failed. Please check your .env file.")
        sys.exit(1)
    
    # Use sample documents if none provided
    if documents is None:
        documents = create_sample_documents()
    
    print(f"\n📄 Processing {len(documents)} documents...")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"✂️  Split into {len(split_docs)} chunks")
    
    # Create embeddings
    print("🔢 Creating embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
    
    # Create vector store
    print(f"💾 Creating vector store at {config.chroma_persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=config.chroma_persist_directory
    )
    
    # Create LLM
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        openai_api_key=config.openai_api_key
    )
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        ),
        return_source_documents=True,
        verbose=config.verbose
    )
    
    print("✅ RAG system ready!\n")
    
    return qa_chain


def run_interactive_qa():
    """Run an interactive Q&A session with the RAG system."""
    print("=" * 60)
    print("📚 RAG Question-Answering System")
    print("=" * 60)
    print("\nThis system can answer questions based on the provided documents.")
    print("Type 'exit', 'quit', or 'bye' to end the session.\n")
    print("=" * 60)
    
    # Set up RAG system
    qa_chain = setup_rag_system()
    
    # Interactive Q&A loop
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            # Get answer
            print("\n🔍 Searching documents...")
            result = qa_chain.invoke({"query": question})
            
            # Print answer
            print(f"\n💡 Answer: {result['result']}")
            
            # Print sources
            if result.get('source_documents'):
                print("\n📎 Sources:")
                for i, doc in enumerate(result['source_documents'], 1):
                    source = doc.metadata.get('source', 'unknown')
                    topic = doc.metadata.get('topic', 'N/A')
                    print(f"  {i}. Source: {source} | Topic: {topic}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'exit' to quit.")


def run_example_queries():
    """Run pre-defined example queries to demonstrate RAG capabilities."""
    print("=" * 60)
    print("📝 Running Example Queries")
    print("=" * 60)
    
    qa_chain = setup_rag_system()
    
    # Example queries
    queries = [
        "What is LangChain?",
        "How does RAG work?",
        "What are vector databases used for?",
        "What should I consider for production RAG systems?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print(f"{'='*60}")
        
        try:
            result = qa_chain.invoke({"query": query})
            print(f"\n💡 Answer: {result['result']}")
            
            if result.get('source_documents'):
                print("\n📎 Sources:")
                for j, doc in enumerate(result['source_documents'], 1):
                    source = doc.metadata.get('source', 'unknown')
                    print(f"  {j}. {source}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Example queries completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Question-Answering Example")
    parser.add_argument(
        "--mode",
        choices=["interactive", "example"],
        default="interactive",
        help="Run mode: 'interactive' for Q&A session, 'example' for demo"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        run_interactive_qa()
    else:
        run_example_queries()
