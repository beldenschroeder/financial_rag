"""
Financial RAG MCP Server
Connects your financial RAG system to Claude Desktop via Model Context Protocol.

This server exposes tools that Claude Desktop can use to:
- Query your financial documents
- Get expense summaries by month/year
- Compare income vs expenses
- Search for specific transactions

To run as an MCP server:
    python -m financial_rag.mcp_server
"""

from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Initialize the MCP server
mcp = FastMCP("Financial RAG Server")

# Lazy initialization of RAG (to avoid slow startup)
_rag_instance = None


def get_rag():
    """Get or initialize the RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        from financial_rag.rag_pipeline import FinancialRag

        _rag_instance = FinancialRag()
    return _rag_instance


@mcp.tool()
def query_finances(question: str) -> str:
    """
    Ask any question about your financial documents.

    Use this tool to query your personal financial records including
    expenses, income statements, and other financial documents.

    Args:
        question: Your question about finances. Be specific about dates
                 and categories when possible.
                 Examples:
                 - "What were my grocery expenses in March 2024?"
                 - "How much did I spend on utilities last year?"
                 - "What was my total income in Q1 2024?"

    Returns:
        A detailed answer based on the financial documents, including
        relevant source citations.
    """
    try:
        rag = get_rag()
        result = rag.query(question)

        answer = result["answer"]

        # Add source citations if available
        if result.get("sources"):
            sources_text = "\n\n📎 Sources consulted:\n"
            for src in result["sources"]:
                sources_text += f"  • {src['file']} ({src['type']}, {src['date']})\n"
            answer += sources_text

        return answer

    except Exception as e:
        return f"Error querying finances: {str(e)}"


@mcp.tool()
def get_monthly_expenses(year: int, month: int) -> str:
    """
    Get a detailed breakdown of expenses for a specific month.

    Args:
        year: The year (e.g., 2024)
        month: The month number (1-12, where 1=January, 12=December)

    Returns:
        A summary of all expenses for that month, organized by category
        if available, with totals.
    """
    month_names = [
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    if not 1 <= month <= 12:
        return "Invalid month. Please provide a number between 1 and 12."

    month_name = month_names[month]

    question = f"""Please provide a detailed breakdown of all expenses for {month_name} {year}.
    
    Include:
    1. All expense categories and their totals
    2. Any notable individual expenses
    3. The total sum of all expenses for the month
    
    Format the response clearly with categories and amounts."""

    try:
        rag = get_rag()
        result = rag.query(question)
        return result["answer"]
    except Exception as e:
        return f"Error getting monthly expenses: {str(e)}"


@mcp.tool()
def get_monthly_income(year: int, month: int) -> str:
    """
    Get income information for a specific month.

    Args:
        year: The year (e.g., 2024)
        month: The month number (1-12)

    Returns:
        A summary of all income sources for that month.
    """
    month_names = [
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    if not 1 <= month <= 12:
        return "Invalid month. Please provide a number between 1 and 12."

    month_name = month_names[month]

    question = f"""What was my total income for {month_name} {year}?
    
    Please include:
    1. All income sources
    2. Individual amounts from each source
    3. Total income for the month"""

    try:
        rag = get_rag()
        result = rag.query(question)
        return result["answer"]
    except Exception as e:
        return f"Error getting monthly income: {str(e)}"


@mcp.tool()
def compare_income_expenses(year: int, month: Optional[int] = None) -> str:
    """
    Compare total income versus total expenses for a period.

    Args:
        year: The year to analyze (e.g., 2024)
        month: Optional month number (1-12). If not provided,
               compares for the entire year.

    Returns:
        A comparison showing:
        - Total income
        - Total expenses
        - Net difference (savings or deficit)
        - Savings rate percentage
    """
    if month:
        month_names = [
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        if not 1 <= month <= 12:
            return "Invalid month. Please provide a number between 1 and 12."
        period = f"{month_names[month]} {year}"
    else:
        period = str(year)

    question = f"""Compare my total income versus total expenses for {period}.
    
    Please calculate and show:
    1. Total income
    2. Total expenses  
    3. Net difference (income minus expenses)
    4. Savings rate as a percentage (if income > expenses)
    5. Brief analysis of the financial health for this period"""

    try:
        rag = get_rag()
        result = rag.query(question)
        return result["answer"]
    except Exception as e:
        return f"Error comparing income and expenses: {str(e)}"


@mcp.tool()
def search_transactions(
    search_term: str, year: Optional[int] = None, transaction_type: Optional[str] = None
) -> str:
    """
    Search for specific transactions or expense categories.

    Args:
        search_term: What to search for (e.g., "Amazon", "groceries",
                    "electricity", "rent")
        year: Optional year to limit search (e.g., 2024)
        transaction_type: Optional filter - "expense", "income", or None for both

    Returns:
        All matching transactions with dates and amounts.
    """
    query_parts = [f'Find all transactions related to "{search_term}"']

    if year:
        query_parts.append(f"from {year}")

    if transaction_type:
        query_parts.append(f"(looking only at {transaction_type} documents)")

    question = (
        " ".join(query_parts)
        + """
    
    For each match, please show:
    - Date
    - Description
    - Amount
    - Category (if available)
    
    Also provide a total sum of all matching transactions."""
    )

    try:
        rag = get_rag()
        result = rag.query(question)
        return result["answer"]
    except Exception as e:
        return f"Error searching transactions: {str(e)}"


@mcp.tool()
def get_expense_trends(category: str, num_months: int = 6) -> str:
    """
    Analyze spending trends for a specific category over time.

    Args:
        category: The expense category to analyze (e.g., "groceries",
                 "utilities", "entertainment", "dining out")
        num_months: Number of recent months to analyze (default: 6)

    Returns:
        A trend analysis showing how spending in this category has
        changed over time, with insights and patterns.
    """
    question = f"""Analyze my spending on "{category}" over the last {num_months} months.
    
    Please provide:
    1. Monthly spending amounts for each month
    2. The trend direction (increasing, decreasing, or stable)
    3. Average monthly spending
    4. Highest and lowest months
    5. Any notable patterns or insights
    
    Present the data in a clear, easy-to-understand format."""

    try:
        rag = get_rag()
        result = rag.query(question)
        return result["answer"]
    except Exception as e:
        return f"Error analyzing expense trends: {str(e)}"


@mcp.tool()
def get_financial_summary(year: int) -> str:
    """
    Get a comprehensive financial summary for an entire year.

    Args:
        year: The year to summarize (e.g., 2024)

    Returns:
        A complete financial overview including income, expenses,
        savings, and key insights.
    """
    question = f"""Provide a comprehensive financial summary for {year}.
    
    Include:
    1. Total annual income (with breakdown by source if available)
    2. Total annual expenses (with breakdown by major categories)
    3. Net savings/deficit for the year
    4. Monthly average income and expenses
    5. Best and worst months financially
    6. Top 3 expense categories
    7. Year-over-year comparison if data is available
    8. Key observations and recommendations
    
    Format this as an executive summary that gives a clear picture of 
    the year's financial health."""

    try:
        rag = get_rag()
        result = rag.query(question)
        return result["answer"]
    except Exception as e:
        return f"Error getting financial summary: {str(e)}"


@mcp.tool()
def get_database_stats() -> str:
    """
    Get information about the financial documents database.

    Returns:
        Statistics about how many documents are indexed and
        available for querying.
    """
    try:
        rag = get_rag()
        stats = rag.get_stats()

        return f"""📊 Financial RAG Database Statistics:
        
• Total document chunks indexed: {stats["total_chunks"]}
• Database location: {stats["persist_directory"]}

The database contains processed chunks from your financial PDFs.
Each chunk is a searchable segment of your documents that can be
retrieved to answer your questions."""

    except Exception as e:
        return f"Error getting database stats: {str(e)}"


# Main entry point
if __name__ == "__main__":
    print("🚀 Starting Financial RAG MCP Server...")
    print("   This server provides tools for querying financial documents.")
    print("   Available tools:")
    print("   - query_finances: Ask any question about your finances")
    print("   - get_monthly_expenses: Get expenses for a specific month")
    print("   - get_monthly_income: Get income for a specific month")
    print("   - compare_income_expenses: Compare income vs expenses")
    print("   - search_transactions: Search for specific transactions")
    print("   - get_expense_trends: Analyze spending trends")
    print("   - get_financial_summary: Get yearly financial summary")
    print("   - get_database_stats: Check database status")
    print()
    mcp.run()
