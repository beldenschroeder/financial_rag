"""
Financial Document Watcher
Automatically detects new or modified PDF files in your OneDrive folder
and ingests them into the RAG vector store.

This script monitors your financial documents directory and:
1. Detects when new PDFs are added
2. Detects when existing PDFs are modified
3. Automatically processes and indexes them
4. Logs all activity

Usage:
    python -m financial_rag.file_watcher                    # Uses default path
    python -m financial_rag.file_watcher /path/to/folder    # Uses custom path
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("file_watcher.log")],
)
logger = logging.getLogger(__name__)


class FinancialDocumentHandler(FileSystemEventHandler):
    """
    Handles file system events for financial PDF documents.

    When a new PDF is created or an existing one is modified,
    this handler automatically ingests it into the RAG system.
    """

    def __init__(self, rag_instance=None, debounce_seconds: float = 2.0):
        """
        Initialize the document handler.

        Args:
            rag_instance: Optional pre-initialized RAG instance.
                         If None, will be initialized on first use.
            debounce_seconds: Time to wait before processing a file
                            (helps with files that trigger multiple events)
        """
        super().__init__()
        self._rag = rag_instance
        self._debounce_seconds = debounce_seconds
        self._pending_files = {}  # Track files being processed

    @property
    def rag(self):
        """Lazy initialization of RAG instance."""
        if self._rag is None:
            logger.info("Initializing RAG pipeline...")
            from financial_rag.rag_pipeline import FinancialRAG

            self._rag = FinancialRAG()
        return self._rag

    def _should_process(self, path: str) -> bool:
        """Check if a file should be processed."""
        path_obj = Path(path)

        # Only process PDF files
        if path_obj.suffix.lower() != ".pdf":
            return False

        # Ignore hidden files and temporary files
        if path_obj.name.startswith(".") or path_obj.name.startswith("~"):
            return False

        # Ignore files in hidden directories
        for part in path_obj.parts:
            if part.startswith("."):
                return False

        return True

    def _process_file(self, path: str, event_type: str):
        """Process a single PDF file."""
        if not self._should_process(path):
            return

        # Debounce: wait a bit to ensure file is fully written
        current_time = time.time()
        last_event_time = self._pending_files.get(path, 0)

        if current_time - last_event_time < self._debounce_seconds:
            return

        self._pending_files[path] = current_time

        logger.info(f"📄 {event_type}: {path}")

        try:
            # Wait for file to be fully written
            time.sleep(1)

            # Check if file still exists and is readable
            if not os.path.exists(path):
                logger.warning(f"File no longer exists: {path}")
                return

            # Ingest the document
            chunks_added = self.rag.ingest_single_document(path)
            logger.info(f"✅ Indexed {chunks_added} chunks from: {Path(path).name}")

        except Exception as e:
            logger.error(f"❌ Error processing {path}: {e}")

        finally:
            # Clean up pending files
            if path in self._pending_files:
                del self._pending_files[path]

    def on_created(self, event):
        """Handle file creation events."""
        if isinstance(event, FileCreatedEvent):
            self._process_file(event.src_path, "New file")

    def on_modified(self, event):
        """Handle file modification events."""
        if isinstance(event, FileModifiedEvent):
            self._process_file(event.src_path, "Modified file")


def get_default_watch_path() -> str:
    """
    Get the default path to watch based on common OneDrive locations.

    Returns:
        Path string if found, otherwise raises an error with helpful message.
    """
    home = Path.home()

    # Common OneDrive paths on Mac
    possible_paths = [
        home / "OneDrive" / "Documents" / "Finance" / "Financial History",
        home / "OneDrive - Personal" / "Documents" / "Finance" / "Financial History",
        home
        / "Library"
        / "CloudStorage"
        / "OneDrive-Personal"
        / "Documents"
        / "Finance"
        / "Financial History",
        home / "Documents" / "Finance" / "Financial History",  # Local fallback
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    # Return a reasonable default even if it doesn't exist yet
    return str(home / "OneDrive" / "Documents" / "Finance" / "Financial History")


def watch_directory(watch_path: Optional[str] = None, recursive: bool = True):
    """
    Start watching a directory for new/modified PDF files.

    Args:
        watch_path: Path to watch. If None, uses default OneDrive path.
        recursive: Whether to watch subdirectories (default: True)
    """
    if watch_path is None:
        watch_path = get_default_watch_path()

    watch_path = Path(watch_path).expanduser()

    # Validate path
    if not watch_path.exists():
        logger.error(f"❌ Watch path does not exist: {watch_path}")
        logger.info("\nTo create this directory structure, run:")
        logger.info(f"  mkdir -p '{watch_path}'")
        logger.info("\nOr specify a different path:")
        logger.info("  python -m financial_rag.file_watcher /your/path/here")
        return

    logger.info("=" * 60)
    logger.info("🔍 Financial Document Watcher")
    logger.info("=" * 60)
    logger.info(f"📂 Watching: {watch_path}")
    logger.info(f"   Recursive: {recursive}")
    logger.info("   Watching for: *.pdf files")
    logger.info("")
    logger.info("When you add or modify PDF files in this folder,")
    logger.info("they will be automatically indexed for RAG queries.")
    logger.info("")
    logger.info("Press Ctrl+C to stop watching.")
    logger.info("=" * 60)

    # Create event handler and observer
    event_handler = FinancialDocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, str(watch_path), recursive=recursive)

    # Start watching
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n👋 Stopping file watcher...")
        observer.stop()

    observer.join()
    logger.info("✅ File watcher stopped.")


def main():
    """Main entry point."""
    # Get path from command line argument if provided
    watch_path = sys.argv[1] if len(sys.argv) > 1 else None

    watch_directory(watch_path)


if __name__ == "__main__":
    main()
