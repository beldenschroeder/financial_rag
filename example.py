#!/usr/bin/env python3
"""
Quick Start Example
Run this script to test your Financial RAG setup.

Usage:
    python example.py
"""

import os
from pathlib import Path


def main():
    print("=" * 60)
    print("🏦 Financial RAG - Quick Start Example")
    print("=" * 60)
    print()

    # Check for API key
    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ERROR: ANTHROPIC_API_KEY not found!")
        print()
        print("Please create a .env file with your API key:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your Anthropic API key")
        print()
        print("Get your key at: https://console.anthropic.com/")
        return

    print("✅ API key found")
    print()

    # Initialize RAG
    print("🔄 Initializing RAG pipeline...")
    from financial_rag.rag_pipeline import FinancialRAG

    rag = FinancialRAG()

    # Check stats
    stats = rag.get_stats()
    print(f"📊 Database contains {stats['total_chunks']} document chunks")
    print()

    if stats["total_chunks"] == 0:
        print("=" * 60)
        print("📚 No documents indexed yet!")
        print("=" * 60)
        print()
        print("Let's ingest some documents. You have a few options:")
        print()

        # Check for common OneDrive paths
        home = Path.home()
        possible_paths = [
            home / "OneDrive" / "Documents" / "Finance" / "Financial History",
            home / "OneDrive - Personal" / "Documents" / "Finance" / "Financial History",
            home / "Documents" / "Finance" / "Financial History",
        ]

        found_path = None
        for path in possible_paths:
            if path.exists():
                found_path = path
                pdf_count = len(list(path.rglob("*.pdf")))
                print("✅ Found your documents at:")
                print(f"   {path}")
                print(f"   ({pdf_count} PDF files)")
                print()
                break

        if found_path:
            print("To ingest these documents, run in Python:")
            print()
            print(f'  rag.ingest_documents("{found_path}")')
            print()

            response = input("Would you like to ingest these documents now? (y/n): ")
            if response.lower() == "y":
                print()
                print("🔄 Ingesting documents...")
                chunks = rag.ingest_documents(str(found_path))
                print(f"✅ Done! Indexed {chunks} chunks.")
                print()
        else:
            print("Could not find your financial documents folder.")
            print()
            print("To ingest documents, run in Python:")
            print()
            print("  from rag_pipeline import FinancialRAG")
            print("  rag = FinancialRAG()")
            print('  rag.ingest_documents("/path/to/your/pdfs")')
            return

    # Demo queries if we have documents
    if rag.get_stats()["total_chunks"] > 0:
        print("=" * 60)
        print("💡 Let's try some example queries!")
        print("=" * 60)
        print()

        example_queries = [
            "What are the most recent personal expenses recorded?",
            "Summarize my personal income sources",
            "What categories do my personal expenses fall into?",
            "What categories do my family expenses fall into?",
        ]

        for i, query in enumerate(example_queries, 1):
            print(f"📝 Query {i}: {query}")
            print("-" * 40)

            try:
                result = rag.query(query, include_sources=True)
                print(result["answer"][:500])  # type: ignore # Truncate long answers
                if len(result["answer"]) > 500:  # type: ignore
                    print("...")
                print()

                if result.get("sources"):
                    print("📎 Sources:", ", ".join([s["file"] for s in result["sources"][:3]]))  # type: ignore
                print()
            except Exception as e:
                print(f"Error: {e}")
                print()

            if i < len(example_queries):
                input("Press Enter to continue to next query...")
                print()

        print("=" * 60)
        print("🎉 Setup complete! You can now:")
        print("=" * 60)
        print()
        print("1. Query from Python:")
        print('   result = rag.query("What were my expenses last month?")')
        print()
        print("2. Use the interactive mode:")
        print("   python rag_pipeline.py")
        print()
        print("3. Connect to Claude Desktop:")
        print("   See README.md for MCP setup instructions")
        print()
        print("4. Auto-sync new documents:")
        print("   python file_watcher.py")


if __name__ == "__main__":
    main()
