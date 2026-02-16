"""
External Document Loading Example using LangChain.

This example demonstrates loading and processing documents from external sources:
- PDF files from local storage
- Web pages from URLs
- Document processing and chunking
- Integration with RAG system
- Environment-based configuration

Features:
- Load PDF documents using pypdf
- Scrape web content using BeautifulSoup
- Process mixed document sources
- Store in vector database
- Query across all sources
"""
import sys
import os
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from utils import get_config


def load_pdf_documents(pdf_paths: List[str]) -> List[Document]:
    """
    Load documents from PDF files.
    
    Args:
        pdf_paths: List of paths to PDF files
        
    Returns:
        List of loaded documents
    """
    documents = []
    
    for pdf_path in pdf_paths:
        try:
            print(f"📄 Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"  ✅ Loaded {len(docs)} pages")
        except Exception as e:
            print(f"  ❌ Error loading {pdf_path}: {str(e)}")
    
    return documents


def load_web_documents(urls: List[str]) -> List[Document]:
    """
    Load documents from web URLs.
    
    Args:
        urls: List of URLs to scrape
        
    Returns:
        List of loaded documents
    """
    documents = []
    
    for url in urls:
        try:
            print(f"🌐 Loading web page: {url}")
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
            print(f"  ✅ Loaded {len(docs)} document(s)")
        except Exception as e:
            print(f"  ❌ Error loading {url}: {str(e)}")
    
    return documents


def create_sample_pdf_content():
    """
    Create a sample PDF for demonstration (if no PDFs provided).
    
    Returns:
        List of sample documents
    """
    documents = [
        Document(
            page_content="""
            LangChain Document Loading Best Practices
            
            When loading external documents, consider:
            1. File format compatibility (PDF, DOCX, TXT, etc.)
            2. Character encoding handling
            3. Metadata preservation
            4. Error handling for corrupted files
            5. Chunking strategy for large documents
            
            PDF Loading:
            - Use PyPDFLoader for standard PDFs
            - Handle password-protected files appropriately
            - Extract metadata (page numbers, author, etc.)
            - Consider OCR for scanned PDFs
            """,
            metadata={"source": "sample_pdf_page_1", "page": 1, "type": "pdf"}
        ),
        Document(
            page_content="""
            Web Scraping with LangChain
            
            Best practices for web scraping:
            1. Respect robots.txt
            2. Rate limiting to avoid overwhelming servers
            3. Handle dynamic content (JavaScript-rendered pages)
            4. Clean HTML to extract meaningful text
            5. Preserve important metadata (URL, title, date)
            
            Common patterns:
            - Use WebBaseLoader for simple HTML pages
            - Use Selenium or Playwright for JavaScript-heavy sites
            - Implement retry logic for failed requests
            - Cache downloaded content when appropriate
            """,
            metadata={"source": "sample_web_page", "url": "https://example.com/docs", "type": "web"}
        ),
    ]
    return documents


def setup_document_qa_system(pdf_paths: List[str] = None, urls: List[str] = None):
    """
    Set up a QA system with external document sources.
    
    Args:
        pdf_paths: Optional list of PDF file paths
        urls: Optional list of web URLs
        
    Returns:
        RetrievalQA: Configured QA chain
    """
    config = get_config()
    
    if not config.validate():
        print("❌ Configuration validation failed. Please check your .env file.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("📚 Loading External Documents")
    print("=" * 60)
    
    all_documents = []
    
    # Load PDFs if provided
    if pdf_paths:
        pdf_docs = load_pdf_documents(pdf_paths)
        all_documents.extend(pdf_docs)
    
    # Load web pages if provided
    if urls:
        web_docs = load_web_documents(urls)
        all_documents.extend(web_docs)
    
    # Use sample content if no external sources
    if not all_documents:
        print("\n⚠️  No external sources provided, using sample content")
        all_documents = create_sample_pdf_content()
    
    print(f"\n📊 Total documents loaded: {len(all_documents)}")
    
    # Split documents
    print("\n✂️  Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
    )
    
    split_docs = text_splitter.split_documents(all_documents)
    print(f"  Created {len(split_docs)} chunks")
    
    # Create embeddings and vector store
    print("\n🔢 Creating embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
    
    print(f"💾 Building vector store...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=config.chroma_persist_directory + "_external"
    )
    
    # Create LLM
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        openai_api_key=config.openai_api_key
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=config.verbose
    )
    
    print("✅ Document QA system ready!\n")
    
    return qa_chain


def run_interactive_qa():
    """Run interactive Q&A session with external documents."""
    print("=" * 60)
    print("📚 External Document QA System")
    print("=" * 60)
    print("\nThis system can process:")
    print("  📄 PDF files (local)")
    print("  🌐 Web pages (URLs)")
    print("\nType 'exit', 'quit', or 'bye' to end.\n")
    print("=" * 60)
    
    # Example usage - in real scenario, user would provide paths/URLs
    print("\n💡 Tip: Edit this file to add your PDF paths or URLs")
    print("   Example: pdf_paths=['/path/to/file.pdf']")
    print("   Example: urls=['https://example.com/page']")
    
    # For demo, use sample content
    qa_chain = setup_document_qa_system()
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            print("\n🔍 Searching documents...")
            result = qa_chain.invoke({"query": question})
            
            print(f"\n💡 Answer: {result['result']}")
            
            if result.get('source_documents'):
                print("\n📎 Sources:")
                for i, doc in enumerate(result['source_documents'], 1):
                    source = doc.metadata.get('source', 'unknown')
                    doc_type = doc.metadata.get('type', 'N/A')
                    page = doc.metadata.get('page', 'N/A')
                    print(f"  {i}. {source} (type: {doc_type}, page: {page})")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def run_example_with_urls():
    """Run example with web URLs."""
    print("=" * 60)
    print("📝 Example: Loading from Web URLs")
    print("=" * 60)
    
    # Example URLs (educational/documentation sites)
    urls = [
        "https://python.langchain.com/docs/get_started/introduction",
    ]
    
    print("\n🌐 This example demonstrates loading from web pages")
    print("⚠️  Note: Requires internet connection\n")
    
    try:
        qa_chain = setup_document_qa_system(urls=urls)
        
        queries = [
            "What is LangChain?",
            "How do I get started with LangChain?",
        ]
        
        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            result = qa_chain.invoke({"query": query})
            print(f"\n💡 Answer: {result['result']}")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("⚠️  Falling back to sample content\n")
        run_example_with_samples()


def run_example_with_samples():
    """Run example with sample content."""
    print("=" * 60)
    print("📝 Example: Sample Document Content")
    print("=" * 60)
    
    qa_chain = setup_document_qa_system()
    
    queries = [
        "What are best practices for loading PDF documents?",
        "How should I handle web scraping with LangChain?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        try:
            result = qa_chain.invoke({"query": query})
            print(f"\n💡 Answer: {result['result']}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="External Document Loading Example")
    parser.add_argument(
        "--mode",
        choices=["interactive", "example", "web"],
        default="example",
        help="Run mode: 'interactive' for Q&A, 'example' for sample demo, 'web' to load from URLs"
    )
    parser.add_argument(
        "--pdf",
        nargs="+",
        help="PDF file paths to load"
    )
    parser.add_argument(
        "--url",
        nargs="+",
        help="Web URLs to load"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        if args.pdf or args.url:
            qa_chain = setup_document_qa_system(pdf_paths=args.pdf, urls=args.url)
            run_interactive_qa()
        else:
            run_interactive_qa()
    elif args.mode == "web":
        run_example_with_urls()
    else:
        run_example_with_samples()
