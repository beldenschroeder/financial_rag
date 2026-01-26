"""
Financial RAG Pipeline - Local Version
A RAG system for querying personal financial PDF documents using:
- LangChain for orchestration
- ChromaDB for vector storage (local)
- HuggingFace embeddings (free, runs locally)
- Claude for generation
"""

import os
import re
from pathlib import Path
from typing import TypedDict, cast

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()


class StatsDict(TypedDict):
    """Type definition for get_stats() return value."""

    total_chunks: int
    persist_directory: str


class QueryResponseDict(TypedDict):
    """Type definition for query() return value."""

    answer: str
    sources: list[dict[str, str]]


class FinancialRAG:
    """
    A RAG (Retrieval-Augmented Generation) system for personal financial documents.

    This class handles:
    1. Loading and processing PDF documents
    2. Storing document embeddings in a local ChromaDB vector store
    3. Querying the documents using Claude as the LLM

    Attributes:
        embeddings: HuggingFace embedding model (free, runs locally)
        vectorstore: ChromaDB instance for storing/retrieving document embeddings
        llm: Claude model for generating answers
        retriever: LangChain retriever for finding relevant documents
        persist_directory: Local directory where ChromaDB stores the vector database
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        embedding_model: str | None = None,
        claude_model: str | None = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            persist_directory: Where to store the ChromaDB database locally
                              (env: CHROMA_DB_PATH, default: ./chroma_db)
            embedding_model: HuggingFace model for creating embeddings
                            (env: EMBEDDING_MODEL, default: sentence-transformers/all-MiniLM-L6-v2)
            claude_model: Which Claude model to use for generation
                         (env: CLAUDE_MODEL, default: claude-sonnet-4-20250514)
        """
        print("🚀 Initializing Financial RAG Pipeline...")

        # Load from environment or use defaults
        persist_directory = persist_directory or os.getenv("CHROMA_DB_PATH", "./chroma_db")
        embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        claude_model = claude_model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

        # Initialize embeddings (free, runs locally on your Mac)
        print(f"   Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},  # Use 'mps' for M1/M2 Mac GPU acceleration
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize or load existing vector store
        print(f"   Setting up vector store at: {persist_directory}")
        self.persist_directory = persist_directory
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="financial_documents",
        )

        # Initialize Claude LLM
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ ANTHROPIC_API_KEY not found!\n"
                "   Please create a .env file with your API key:\n"
                "   ANTHROPIC_API_KEY=your-key-here"
            )

        print(f"   Connecting to Claude ({claude_model})")
        self.llm = ChatAnthropic(model=claude_model, api_key=api_key, max_tokens=4096)  # type: ignore

        # Create retriever (finds relevant documents for a query)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},  # Return top 5 most relevant chunks
        )

        print("✅ RAG Pipeline initialized successfully!\n")

    def ingest_documents(self, documents_path: str, force_reingest: bool = False) -> int:
        """
        Load all PDFs from a directory into the vector store.

        This method:
        1. Finds all PDF files in the directory (recursively)
        2. Extracts text from each PDF
        3. Splits text into chunks for better retrieval
        4. Stores chunks with embeddings in ChromaDB

        Args:
            documents_path: Path to directory containing PDFs
                           (e.g., "~/OneDrive/Documents/Finance/Financial History")
            force_reingest: If True, re-process all documents even if already ingested

        Returns:
            Number of chunks added to the vector store
        """
        doc_path = Path(documents_path).expanduser()

        if not doc_path.exists():
            raise FileNotFoundError(f"❌ Directory not found: {doc_path}")

        print(f"📂 Scanning for PDFs in: {doc_path}")

        # Find all PDF files
        pdf_files = list(doc_path.rglob("*.pdf"))
        print(f"   Found {len(pdf_files)} PDF files")

        if not pdf_files:
            print("   ⚠️  No PDF files found!")
            return 0

        # Text splitter for chunking documents
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Characters per chunk
            chunk_overlap=200,  # Overlap between chunks for context continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Split priorities
        )

        all_chunks = []
        processed_count = 0

        for pdf_path in pdf_files:
            try:
                print(f"   Processing: {pdf_path.name}")

                # Load PDF
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()

                # Extract metadata from filename
                # Expected format: "Family Expenses 2024-01-01.pdf" or "Income Statement 2024-01-01.pdf"
                metadata = self._extract_metadata(pdf_path)

                # Add metadata to each page
                for page in pages:
                    page.metadata.update(metadata)  # type: ignore

                # Split into chunks
                chunks = text_splitter.split_documents(pages)  # type: ignore
                all_chunks.extend(chunks)  # type: ignore
                processed_count += 1

            except Exception as e:
                print(f"   ⚠️  Error processing {pdf_path.name}: {e}")
                continue

        if all_chunks:
            print(f"\n📥 Adding {len(all_chunks)} chunks to vector store...")  # type: ignore
            self.vectorstore.add_documents(all_chunks)  # type: ignore
            print(
                f"✅ Successfully ingested {processed_count} documents ({len(all_chunks)} chunks)"  # type: ignore
            )

        return len(all_chunks)  # type: ignore

    def ingest_single_document(self, pdf_path: str) -> int:
        """
        Ingest a single PDF file (useful for incremental updates).

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Number of chunks added
        """
        doc_path = Path(pdf_path).expanduser()

        if not doc_path.exists():
            raise FileNotFoundError(f"❌ File not found: {doc_path}")

        print(f"📄 Ingesting: {doc_path.name}")

        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        loader = PyPDFLoader(str(doc_path))
        pages = loader.load()

        metadata = self._extract_metadata(doc_path)
        for page in pages:
            page.metadata.update(metadata)  # type: ignore

        chunks = text_splitter.split_documents(pages)  # type: ignore

        if chunks:
            self.vectorstore.add_documents(chunks)
            print(f"✅ Added {len(chunks)} chunks from {doc_path.name}")

        return len(chunks)

    def _extract_metadata(self, pdf_path: Path) -> dict[str, str]:
        """
        Extract metadata from filename and path.

        Handles formats like:
        - "Family Expenses 2024-01-01.pdf"
        - "Income Statement 2024-01-01.pdf"
        - Parent folders like "Financial History (2024)"

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with extracted metadata
        """
        filename = pdf_path.stem  # Filename without extension

        metadata: dict[str, str] = {
            "source_file": pdf_path.name,
            "source_path": str(pdf_path),
            "document_type": "unknown",
            "document_category": "unknown",  # "family" or "personal"
            "year": "unknown",
            "month": "unknown",
            "date": "unknown",
        }

        # Determine document category (family vs personal) - default to personal if not specified
        filename_lower = filename.lower()
        if "family" in filename_lower:
            metadata["document_category"] = "family"
        else:
            metadata["document_category"] = "personal"

        # Determine document type(s)
        # Note: "Income Statement" contains both income (gains) and expenses (losses)
        has_income = "income" in filename_lower
        has_expense = "expense" in filename_lower or "loss" in filename_lower

        if has_income and has_expense:
            metadata["document_type"] = "income,expenses"
        elif has_income:
            metadata["document_type"] = "income"
        elif has_expense:
            metadata["document_type"] = "expenses"
        else:
            metadata["document_type"] = "unknown"

        # Extract date from filename (format: YYYY-MM-DD)
        date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
        if date_match:
            metadata["year"] = date_match.group(1)
            metadata["month"] = date_match.group(2)
            metadata["date"] = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

        # Try to extract year from parent folder (format: "Financial History (2024)")
        parent_folder = pdf_path.parent.name
        year_match = re.search(r"\((\d{4})\)", parent_folder)
        if year_match and metadata["year"] == "unknown":
            metadata["year"] = year_match.group(1)

        return metadata

    def query(self, question: str, include_sources: bool = True) -> QueryResponseDict:
        """
        Ask a question about your financial documents.

        This method:
        1. Finds the most relevant document chunks for your question
        2. Sends them to Claude along with your question
        3. Returns Claude's answer based on the documents

        Args:
            question: Your question (e.g., "What were my grocery expenses in March 2024?")
            include_sources: Whether to include source documents in the response

        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        print(f"🔍 Query: {question}")

        # System prompt that tells Claude how to behave
        system_prompt = """You are a helpful financial assistant analyzing personal
finance documents. Your job is to answer questions about the user's expenses,
income, and financial history based on the provided context.

IMPORTANT: Document Structure and Dates
- All documents are named with dates in YYYY-MM-DD format (e.g., "Expenses 2025-01-15.pdf")
- Parent folders contain years like "Financial History (2025)"
- Documents are categorized as either "family" or "personal" (defaults to personal if not specified)
- "Income Statement" documents contain both income (gains) and expenses (losses)
- Document types include: "expenses", "income", or "income,expenses"
- When answering about recent data, prioritize documents with more recent dates

Guidelines:
- Only use information from the provided context
- If the context doesn't contain enough information, say so clearly
- Be specific with numbers and dates when available
- Pay attention to document metadata (dates, categories) to give accurate timeframes
- Distinguish between family and personal finances when relevant
- Organize your response clearly, especially for comparisons
- If asked about trends, analyze patterns across multiple documents with attention to dates

Context from financial documents:
{context}
"""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(  # type: ignore
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )  # type: ignore

        # Retrieve relevant documents
        retrieved_docs = self.retriever.invoke(question)  # type: ignore

        # Format context from documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])  # type: ignore

        # Create and invoke the chain
        chain = (  # type: ignore
            {"context": lambda x: context, "question": lambda x: question}  # type: ignore
            | prompt
            | self.llm
            | StrOutputParser()
        )  # type: ignore

        answer = chain.invoke({})  # type: ignore

        sources: list[dict[str, str]] = []
        if include_sources:
            # Extract unique source files
            seen_files = set()  # type: ignore
            for doc in retrieved_docs:  # type: ignore
                source_file = doc.metadata.get("source_file", "Unknown")  # type: ignore
                if source_file not in seen_files:
                    sources.append(
                        {
                            "file": source_file,
                            "type": doc.metadata.get("document_type", "unknown"),  # type: ignore
                            "date": doc.metadata.get("date", "unknown"),  # type: ignore
                        }
                    )
                    seen_files.add(source_file)  # type: ignore

        response: QueryResponseDict = cast(
            QueryResponseDict,
            {"answer": answer, "sources": sources},  # type: ignore
        )  # type: ignore
        print(f"✅ Found {len(retrieved_docs)} relevant chunks")  # type: ignore
        return response

    def get_stats(self) -> StatsDict:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with collection statistics
        """
        try:
            results = self.vectorstore.get()
            count = len(results.get("ids", [])) if results else 0
        except Exception:
            count = 0

        return {"total_chunks": count, "persist_directory": self.persist_directory}

    def clear_database(self):
        """
        Clear all documents from the vector store.
        Use with caution!
        """
        print("⚠️  Clearing vector store...")
        self.vectorstore.delete_collection()
        # Reinitialize empty collection
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="financial_documents",
        )
        print("✅ Vector store cleared")


def main():
    """
    Example usage of the Financial RAG Pipeline.
    """
    # Initialize the RAG system
    rag = FinancialRAG()

    # Check if we have documents
    stats = rag.get_stats()
    print(f"📊 Current stats: {stats['total_chunks']} chunks in database\n")

    # If no documents, show instructions
    if stats["total_chunks"] == 0:
        print("=" * 60)
        print("📚 No documents ingested yet!")
        print("=" * 60)
        print("\nTo ingest your financial documents, run:")
        print()
        print('  rag.ingest_documents("~/OneDrive/Documents/Finance/Financial History")')
        print()
        print("Or for a specific folder:")
        print()
        print(
            '  rag.ingest_documents("~/OneDrive/Documents/Finance/Financial History/Financial History (2024)")'
        )
        print("=" * 60)
        return

    # Example queries
    example_queries = [
        "What were my total expenses last month?",
        "How much income did I receive in 2024?",
        "What are my biggest expense categories?",
        "Compare my January and February expenses",
    ]

    print("💡 Example queries you can try:")
    for i, q in enumerate(example_queries, 1):
        print(f"   {i}. {q}")
    print()

    # Interactive mode
    print("Enter your question (or 'quit' to exit):")
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue

        result = rag.query(question)
        print(f"\n📝 Answer:\n{result['answer']}")

        if result.get("sources"):
            print("\n📎 Sources:")
            for src in result["sources"]:
                print(f"   - {src['file']} ({src['type']}, {src['date']})")


if __name__ == "__main__":
    main()
