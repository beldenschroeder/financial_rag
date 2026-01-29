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
- All documents are named with dates in YYYY-MM-DD format (e.g., "Family Expenses 2025-11-30.pdf" and "Income Statement 2025-01-31.pdf)
- Each document only reflects data for the month ending in the year, month, and day in the filename
- Parent folders contain years like "Financial History (2025)"
- Parent folders show the year the documents belong to of when year the data pertains to
- Documents are categorized as either "family" or "personal" (defaults to personal if file name does not specify, "Personal")
- "Family" documents have the word "Family" in the filename
- An example of "family" document: "Family Expenses 2025-11-30.pdf"
- An example of "personal" document: "Income Statement 2025-01-31.pdf"
- "Income Statement" documents contain both income (gains) and expenses (losses)
- "Income Statement" and "Family Expenses" documents have columns with titles containing "Projected". These are not actual amounts. Actual amount, current ammounts are found in the column with no title.
- Document types include: "expenses", "income", or "income,expenses"
- When answering about recent data, prioritize documents with more recent dates (see filenames and metadata)

Guidelines:
- Only use information from the provided context
- If the context doesn't contain enough information, say so clearly
- Be specific with numbers and dates when available
- Pay attention to document metadata (dates, categories) to give accurate timeframes
- Distinguish between family and personal finances when relevant
- Organize your response clearly, especially for comparisons
- If asked about trends, analyze patterns across multiple documents with attention to dates
- Some calculations asked may require gathering data from multiple documents
- Don't use the creation or modification date of the PDF files to determine the data the financial data pertains to.
- Use only the dates found in the parent folder names and the PDF filenames to determine the date the financial data pertains to.
- The PDF files give column names and row names to explain what data you are analyzing.

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

        # Extract temporal and categorical information from the question
        years_in_question = self._extract_years_from_question(question)
        months_days = self._extract_months_days_from_question(question)
        doc_preferences = self._extract_document_preferences(question)

        # Retrieve relevant documents with temporal and category-aware filtering
        retrieved_docs = self._retrieve_with_priority(
            question, years_in_question, months_days, doc_preferences
        )

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

    def _extract_years_from_question(self, question: str) -> list[str]:
        """
        Extract years mentioned in the question (e.g., "2025", "2024").

        Args:
            question: The user's question

        Returns:
            List of years found in the question
        """
        # Find all 4-digit numbers that look like years (2000-2099)
        years_found = re.findall(r"\b(20\d{2})\b", question)
        return list(set(years_found))  # Remove duplicates

    def _extract_months_days_from_question(self, question: str) -> dict:  # type: ignore
        """
        Extract months and days mentioned in the question.
        Looks for month names (January, Feb, etc.) and day numbers.
        Each file represents data for the month ending on that specific day.

        Args:
            question: The user's question

        Returns:
            Dictionary with 'months' and 'days' keys containing lists of integers
        """
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

        months_found = []
        for month_name, month_num in month_names.items():
            if month_name in question_lower:
                months_found.append(month_num)

        # Look for day numbers (1-31) in patterns like "ending 31" or "month ending"
        days_found = []

        # Look for "ending <day>" pattern (e.g., "ending 31")
        day_match = re.search(r"ending\s+(\d{1,2})(?:[a-z]{2})?\b", question_lower)
        if day_match:
            days_found.append(int(day_match.group(1)))

        # Look for "ending <month name>" pattern and infer last day of month
        for month_name, month_num in month_names.items():
            if re.search(rf"ending\s+{month_name}\b", question_lower):
                # Add a flag for month-ending (days 28-31)
                days_found.append(0)  # 0 means "month-ending documents"
                break

        # Also match "month ending" with implicit last day (28-31)
        if re.search(r"month\s+ending", question_lower):
            # This means they want month-end data, which would be days 28-31
            # We'll use this as a flag to prioritize documents that represent month-end
            days_found.append(0)  # 0 means "month-ending documents"

        return {"months": list(set(months_found)), "days": list(set(days_found))}  # type: ignore

    def _extract_document_preferences(self, question: str) -> dict:  # type: ignore
        """
        Extract preferences about document type and category from the question.
        Determines if looking for family/personal, expenses/income, etc.

        Important:
        - If "family" in question, prioritize family documents
        - If "personal" in question OR not "family", also include income statements
        - Income statements contain both gains (income) and losses (expenses)
        - Personal expenses should match non-family documents

        Args:
            question: The user's question

        Returns:
            Dictionary with preferences: is_family, is_personal, wants_expenses, wants_income
        """
        question_lower = question.lower()

        return {  # type: ignore
            "is_family": "family" in question_lower,
            "is_personal": "personal" in question_lower,
            "wants_expenses": any(
                word in question_lower for word in ["expense", "spent", "cost", "spending", "loss"]
            ),
            "wants_income": any(
                word in question_lower for word in ["income", "earn", "received", "revenue", "gain"]
            ),
        }  # type: ignore

    def _retrieve_with_priority(  # type: ignore
        self,
        question: str,
        years: list[str],
        months_days: dict,  # type: ignore
        preferences: dict,  # type: ignore
    ) -> list:  # type: ignore
        """
        Retrieve documents with priority given to years, months, days, and document type preferences.

        Matching strategy:
        1. Year filter (required if years specified)
        2. Month/day filter (prioritize month-end documents if requested)
        3. Category filter (family vs personal, expenses vs income)
        4. Fallback to semantic search if no matches

        Args:
            question: The user's question
            years: List of years extracted from the question
            months_days: Dict with 'months' and 'days' lists
            preferences: Dict with document type preferences (is_family, wants_expenses, etc.)

        Returns:
            List of relevant document chunks
        """
        import chromadb
        from langchain_core.documents import Document

        if not years:
            # No specific year mentioned, return standard results
            return self.retriever.invoke(question)  # type: ignore

        # Get all documents and apply filters
        db = chromadb.PersistentClient(path=self.persist_directory)
        collection = db.get_collection("financial_documents")

        # Embed the question for semantic search
        query_embedding = self.embeddings.embed_query(question)

        # Get more initial results to filter through
        # Get up to 300 to ensure documents with specified year/month/day are included
        # (semantic search may not rank them highly since it doesn't understand dates well)
        all_docs_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=300,
            include=["metadatas", "documents", "distances"],
        )

        # Extract data from results
        metadatas = all_docs_result["metadatas"][0] if all_docs_result["metadatas"] else []  # type: ignore
        documents = all_docs_result["documents"][0] if all_docs_result["documents"] else []  # type: ignore
        distances = all_docs_result["distances"][0] if all_docs_result["distances"] else []  # type: ignore

        filtered_docs: list = []  # type: ignore
        for meta, doc_text, distance in zip(metadatas, documents, distances):  # type: ignore
            # Filter by year (required)
            if meta.get("year") not in years:  # type: ignore
                continue

            # Check month/day filters (only apply if specified)
            months_specified = months_days.get("months", [])  # type: ignore
            days_specified = months_days.get("days", [])  # type: ignore

            # Only apply month/day filtering if they were actually mentioned in the question
            if months_specified or days_specified:
                doc_month = int(meta.get("month", 0)) if meta.get("month") != "unknown" else 0  # type: ignore
                doc_day = (
                    int(meta.get("date", "").split("-")[2]) if meta.get("date") != "unknown" else 0
                )  # type: ignore

                # If months are specified, document month must match
                if months_specified and doc_month not in months_specified:  # type: ignore
                    continue

                # If days are specified, check if document day matches
                if days_specified:
                    # 0 means "month-ending documents" (days 28-31)
                    if 0 in days_specified:  # type: ignore
                        if doc_day < 28:  # type: ignore
                            continue
                    elif doc_day not in days_specified:  # type: ignore
                        # Exact day match required if specific day(s) mentioned
                        continue

            # Check document category preferences
            is_family = preferences.get("is_family", False)  # type: ignore
            is_personal = preferences.get("is_personal", False)  # type: ignore
            wants_expenses = preferences.get("wants_expenses", False)  # type: ignore
            wants_income = preferences.get("wants_income", False)  # type: ignore

            doc_category = meta.get("document_category", "personal")  # type: ignore
            doc_type = meta.get("document_type", "unknown")  # type: ignore

            # Family documents only have "Family Expenses" in the name
            # Personal items include both Income Statements (which have income AND expenses)
            # If asking for family, exclude non-family documents
            if is_family and doc_category != "family":  # type: ignore
                continue

            # If asking for personal and not asking for family, exclude family docs
            # OR if not mentioning family at all, include income statements too
            if (is_personal or not is_family) and doc_category == "family":  # type: ignore
                # Allow family if also wanting expenses (they may still be relevant)
                if not wants_expenses:  # type: ignore
                    continue

            # Type matching for expenses vs income
            # Income statements contain both, so they match either query
            # Family Expenses only have expenses
            if wants_expenses and "expense" not in doc_type and doc_category != "family":  # type: ignore
                # Income statements are OK for expense queries (contain losses/expenses)
                if "income" not in doc_type:  # type: ignore
                    continue

            if wants_income and "income" not in doc_type:  # type: ignore
                continue

            # Document passed all filters, add it
            doc = Document(page_content=doc_text, metadata=meta)  # type: ignore
            filtered_docs.append((doc, distance))  # type: ignore

        # If we found matching documents, return top 5 sorted by relevance
        if filtered_docs:
            filtered_docs.sort(key=lambda x: x[1])  # type: ignore
            return [doc for doc, _ in filtered_docs[:5]]  # type: ignore

        # Fallback: return top semantic matches if no filters matched anything
        # This prevents empty results when filters are too restrictive
        fallback_docs: list = []  # type: ignore
        for meta, doc_text in zip(metadatas, documents):  # type: ignore
            # At least match the year
            if meta.get("year") in years:  # type: ignore
                doc = Document(page_content=doc_text, metadata=meta)  # type: ignore
                fallback_docs.append(doc)  # type: ignore

        return (
            fallback_docs[:5]
            if fallback_docs
            else (self.retriever.invoke(question) if self.retriever else [])
        )  # type: ignore

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
