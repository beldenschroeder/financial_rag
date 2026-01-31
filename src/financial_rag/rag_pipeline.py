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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypedDict, cast

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


# ============================================================================
# QUESTION ANALYSIS - Single Responsibility: Parse user questions
# ============================================================================


class QuestionAnalyzer:
    """Extracts temporal and categorical information from user questions."""

    def analyze(self, question: str) -> dict[str, Any]:
        """Parse question to extract structured information."""
        return {
            "years": self._extract_years(question),
            "months_days": self._extract_months_days(question),
            "preferences": self._extract_preferences(question),
        }

    def _extract_years(self, question: str) -> list[str]:
        """Extract years mentioned in the question (e.g., "2025", "2024")."""
        years_found = re.findall(r"\b(20\d{2})\b", question)
        return list(set(years_found))

    def _extract_months_days(self, question: str) -> dict[str, list[int]]:
        """Extract months and days mentioned in the question."""
        question_lower = question.lower()
        month_names = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }

        months_found: list[int] = []
        for month_name, month_num in month_names.items():
            if month_name in question_lower:
                months_found.append(month_num)

        days_found: list[int] = []
        day_match = re.search(r"ending\s+(\d{1,2})(?:[a-z]{2})?\b", question_lower)
        if day_match:
            days_found.append(int(day_match.group(1)))

        for month_name in month_names:
            if re.search(rf"ending\s+{month_name}\b", question_lower):
                days_found.append(0)
                break

        if re.search(r"month\s+ending", question_lower):
            days_found.append(0)

        return {"months": list(set(months_found)), "days": list(set(days_found))}

    def _extract_preferences(self, question: str) -> dict[str, bool]:
        """Extract document type and category preferences."""
        question_lower = question.lower()
        return {
            "is_family": "family" in question_lower,
            "is_personal": "personal" in question_lower,
            "wants_expenses": any(
                word in question_lower for word in ["expense", "spent", "cost", "spending", "loss"]
            ),
            "wants_income": any(
                word in question_lower for word in ["income", "earn", "received", "revenue", "gain"]
            ),
        }


# ============================================================================
# DOCUMENT FILTERS - Single Responsibility: Filter documents by criteria
# ============================================================================


class DocumentFilter(ABC):
    """Abstract base for filtering documents."""

    @abstractmethod
    def matches(self, metadata: dict[str, Any]) -> bool:
        """Return True if document matches this filter."""
        pass


class YearFilter(DocumentFilter):
    """Filter documents by year."""

    def __init__(self, years: list[str]):
        self.years = years

    def matches(self, metadata: dict[str, Any]) -> bool:
        return metadata.get("year") in self.years


class MonthDayFilter(DocumentFilter):
    """Filter documents by month and day."""

    def __init__(self, months: list[int], days: list[int]):
        self.months = months
        self.days = days

    def matches(self, metadata: dict[str, Any]) -> bool:
        if not self.months and not self.days:
            return True

        doc_month = int(metadata.get("month", 0)) if metadata.get("month") != "unknown" else 0
        doc_date = cast(str, metadata.get("date", ""))
        doc_day = int(doc_date.split("-")[2]) if doc_date != "unknown" else 0

        if self.months and doc_month not in self.months:
            return False

        if self.days:
            if 0 in self.days:  # Month-ending documents
                return doc_day >= 28
            elif doc_day not in self.days:
                return False

        return True


class CategoryTypeFilter(DocumentFilter):
    """Filter documents by category (family/personal) and type (expenses/income)."""

    def __init__(self, preferences: dict[str, bool]):
        self.is_family = preferences.get("is_family", False)
        self.is_personal = preferences.get("is_personal", False)
        self.wants_expenses = preferences.get("wants_expenses", False)
        self.wants_income = preferences.get("wants_income", False)

    def matches(self, metadata: dict[str, Any]) -> bool:
        doc_category = metadata.get("document_category", "personal")
        doc_type = metadata.get("document_type", "unknown")

        # Family filter
        if self.is_family and doc_category != "family":
            return False

        if (self.is_personal or not self.is_family) and doc_category == "family":
            if not self.wants_expenses:
                return False

        # Type filter
        if self.wants_expenses and "expense" not in doc_type and doc_category != "family":
            if "income" not in doc_type:
                return False

        if self.wants_income and "income" not in doc_type:
            return False

        return True


class DocumentFilterChain:
    """Applies multiple filters in sequence (Chain of Responsibility pattern)."""

    def __init__(self, filters: list[DocumentFilter]):
        self.filters = filters

    def apply(self, metadata: dict[str, Any]) -> bool:
        """Return True if document passes all filters."""
        return all(f.matches(metadata) for f in self.filters)


# ============================================================================
# SOURCE EXTRACTION - Single Responsibility: Extract source citations
# ============================================================================


class SourceExtractor:
    """Extracts unique source citations from retrieved documents."""

    def extract(self, docs: list[Any]) -> list[dict[str, str]]:
        """Extract unique source files from documents."""
        sources: list[dict[str, str]] = []
        seen_files: set[str] = set()

        for doc in docs:
            source_file = doc.metadata.get("source_file", "Unknown")
            if source_file not in seen_files:
                sources.append(
                    {
                        "file": source_file,
                        "type": doc.metadata.get("document_type", "unknown"),
                        "date": doc.metadata.get("date", "unknown"),
                    }
                )
                seen_files.add(source_file)

        return sources


# ============================================================================
# PROMPT MANAGEMENT - Single Responsibility: Manage LLM prompts
# ============================================================================


class PromptManager:
    """Manages prompt templates for the LLM."""

    @staticmethod
    def get_system_prompt() -> str:
        """Return the system prompt for Claude."""
        return """You are a helpful financial assistant analyzing personal finance documents.
Answer questions about the user's expenses, income, and financial history based on the provided context.

Guidelines for Interpretation:
- Only use information from the provided context
- If the context doesn't contain enough information, say so clearly
- Be specific with numbers and dates when available
- Distinguish between family and personal finances when relevant
- Organize your response clearly, especially for comparisons
- If asked about trends, analyze patterns across multiple documents

Important Notes about Document Content:
- "Projected" columns contain estimates, not actual amounts. Use the untitled column for actual current amounts.
- Income Statement documents contain both income (gains) and expenses (losses)
- Column names and row names in the documents explain what data you are analyzing
- The documents you receive have already been filtered for the requested date range and category

Context from financial documents:
{context}
"""

    @staticmethod
    def create_prompt(system_prompt: str) -> ChatPromptTemplate:
        """Create a prompt template from system prompt."""
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )  # type: ignore


class FinancialRag:
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
        device = self._detect_device()
        print(f"   Using device: {device}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
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

    def _detect_device(self) -> str:
        """
        Auto-detect the best device for embeddings based on platform and environment.

        Priority order:
        1. Environment variable EMBEDDING_DEVICE (if set)
        2. Apple Silicon (M1/M2/M3+) Macs → "mps"
        3. All other systems → "cpu"

        Returns:
            Device string: "mps" for Apple Silicon, "cpu" for everything else
        """
        # Check for explicit environment variable first
        device_env = os.getenv("EMBEDDING_DEVICE")
        if device_env:
            print(f"   Using device from EMBEDDING_DEVICE: {device_env}")
            return device_env

        # Try to detect Apple Silicon
        try:
            import platform
            import sys

            if sys.platform == "darwin":  # macOS
                machine = platform.machine()
                # Apple Silicon uses arm64/aarch64 architecture
                if "arm64" in machine or "aarch64" in machine:
                    # Verify MPS is actually available
                    try:
                        import torch

                        if torch.backends.mps.is_available():
                            print("   Apple Silicon detected, using MPS acceleration")
                            return "mps"
                    except (ImportError, AttributeError):
                        # PyTorch not available or MPS not available
                        pass
        except Exception:
            pass

        # Default to CPU for all other cases
        print("   Using CPU (set EMBEDDING_DEVICE env var to override)")
        return "cpu"

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

        Simplified to focus on orchestration only (Single Responsibility).

        Args:
            question: Your question (e.g., "What were my grocery expenses in March 2024?")
            include_sources: Whether to include source documents in the response

        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        print(f"🔍 Query: {question}")

        # 1. PARSE THE QUESTION (delegated to QuestionAnalyzer)
        analyzer = QuestionAnalyzer()
        analysis = analyzer.analyze(question)

        # 2. RETRIEVE RELEVANT DOCUMENTS
        retrieved_docs = self._retrieve_with_priority(
            question,
            analysis["years"],
            analysis["months_days"],
            analysis["preferences"],
        )

        # 3. BUILD LLM PROMPT (delegated to PromptManager)
        prompt_manager = PromptManager()
        system_prompt = prompt_manager.get_system_prompt()
        prompt = prompt_manager.create_prompt(system_prompt)

        # 4. FORMAT CONTEXT FROM DOCUMENTS (with metadata for clarity)
        context_parts = []
        for doc in retrieved_docs:  # type: ignore
            metadata = doc.metadata if hasattr(doc, "metadata") else {}  # type: ignore
            source_info = f"[Source: {metadata.get('source', 'unknown')}]" if metadata else ""  # type: ignore
            context_parts.append(f"{source_info}\n{doc.page_content}")  # type: ignore
        context = "\n\n".join(context_parts)  # type: ignore

        # 5. INVOKE LLM
        chain = (  # type: ignore
            {"context": lambda x: context, "question": lambda x: question}  # type: ignore
            | prompt
            | self.llm
            | StrOutputParser()
        )  # type: ignore
        answer = chain.invoke({})  # type: ignore

        # 6. EXTRACT SOURCES (delegated to SourceExtractor)
        sources: list[dict[str, str]] = []
        if include_sources:
            extractor = SourceExtractor()
            sources = extractor.extract(retrieved_docs)  # type: ignore

        response: QueryResponseDict = cast(
            QueryResponseDict,
            {"answer": answer, "sources": sources},  # type: ignore
        )  # type: ignore
        print(f"✅ Found {len(retrieved_docs)} relevant chunks")  # type: ignore
        return response

    def _extract_years_from_question(self, question: str) -> list[str]:
        """Extract years from question. DEPRECATED: Use QuestionAnalyzer instead."""
        analyzer = QuestionAnalyzer()
        return analyzer._extract_years(question)

    def _extract_months_days_from_question(self, question: str) -> dict[str, list[int]]:
        """Extract months/days from question. DEPRECATED: Use QuestionAnalyzer instead."""
        analyzer = QuestionAnalyzer()
        return analyzer._extract_months_days(question)

    def _extract_document_preferences(self, question: str) -> dict[str, bool]:
        """Extract preferences from question. DEPRECATED: Use QuestionAnalyzer instead."""
        analyzer = QuestionAnalyzer()
        return analyzer._extract_preferences(question)

    def _retrieve_with_priority(
        self,
        question: str,
        years: list[str],
        months_days: dict[str, list[int]],
        preferences: dict[str, bool],
    ) -> list[Any]:
        """
        Retrieve documents using hybrid retrieval (metadata-first + semantic search).

        This implements best practice for RAG with structured metadata:
        1. Filter by metadata (year, month, category, type) - reduces search space
        2. Embed query and search filtered results - semantic relevance
        3. Sort by similarity and return top 5 - ranking by relevance

        Args:
            question: The user's question
            years: List of years extracted from the question
            months_days: Dict with 'months' and 'days' lists
            preferences: Dict with document type preferences

        Returns:
            List of relevant document chunks
        """
        import chromadb
        from langchain_core.documents import Document

        if not years:
            # No specific year mentioned, return standard semantic results
            return self.retriever.invoke(question)  # type: ignore

        # BUILD FILTER CHAIN (Single Responsibility: Each filter has one job)
        filters: list[DocumentFilter] = [
            YearFilter(years),
            MonthDayFilter(months_days.get("months", []), months_days.get("days", [])),  # type: ignore
            CategoryTypeFilter(preferences),
        ]
        filter_chain = DocumentFilterChain(filters)

        # GET ALL DOCUMENTS FROM VECTOR STORE
        db = chromadb.PersistentClient(path=self.persist_directory)
        collection = db.get_collection("financial_documents")
        query_embedding = self.embeddings.embed_query(question)

        # METADATA-FIRST APPROACH: Get more results, we'll filter aggressively
        all_docs_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=300,
            include=["metadatas", "documents", "distances"],
        )

        # APPLY METADATA FILTERS (reduces search space)
        metadatas = all_docs_result["metadatas"][0] if all_docs_result["metadatas"] else []  # type: ignore
        documents = all_docs_result["documents"][0] if all_docs_result["documents"] else []  # type: ignore
        distances = all_docs_result["distances"][0] if all_docs_result["distances"] else []  # type: ignore

        filtered_docs: list[tuple[Any, Any]] = []
        for meta, doc_text, distance in zip(metadatas, documents, distances):  # type: ignore
            if filter_chain.apply(meta):  # type: ignore
                doc = Document(page_content=doc_text, metadata=meta)  # type: ignore
                filtered_docs.append((doc, distance))

        # SORT BY SEMANTIC RELEVANCE (ranking by similarity)
        if filtered_docs:
            filtered_docs.sort(key=lambda x: x[1])  # type: ignore
            return [doc for doc, _ in filtered_docs[:5]]

        # FALLBACK: If filters are too restrictive, return year-matched results only
        fallback_docs: list[Any] = []
        for meta, doc_text in zip(metadatas, documents):  # type: ignore
            if meta.get("year") in years:  # type: ignore
                doc = Document(page_content=doc_text, metadata=meta)  # type: ignore
                fallback_docs.append(doc)

        return (
            fallback_docs[:5]
            if fallback_docs
            else (self.retriever.invoke(question) if self.retriever else [])
        )

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
    rag = FinancialRag()

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
        "What were my total, personal expenses for months ending Nov. 30, 2025 and Dec. 31, 2025?",
        "How much income did I receive in on month ending Dec. 31, 2025?",
        "What are my biggest family expense categories for the month endingDec. 31, 2025?",
        "Compare my January 2025 and February 2025 expenses.",
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
