"""
Setup Script for RAG School Data Chatbot
Run this script to initialize the database and vector store.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_files():
    """Check if required files exist"""
    required_files = [
        "2025 CMAS Performance_ELA.xlsx",
        "2025 CMAS Performance_Math.xlsx"
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"   - {file}")
        return False

    return True


def check_api_key():
    """Check if API key is configured"""
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("⚠️  No ANTHROPIC_API_KEY found in environment")
        print("   Please add it to .env file or set as environment variable")
        print("   Get your key from: https://console.anthropic.com/settings/keys")
        return False

    print("✅ API key found")
    return True


def setup_database():
    """Initialize DuckDB database"""
    print("\n📊 Setting up database...")

    from data_processor import DataProcessor

    processor = DataProcessor()
    success = processor.load_and_process(
        "2025 CMAS Performance_ELA.xlsx",
        "2025 CMAS Performance_Math.xlsx"
    )

    if success:
        print("✅ Database created: school_data.duckdb")
        processor.close()
        return True
    else:
        print("❌ Database setup failed")
        return False


def setup_vector_store():
    """Initialize vector store"""
    print("\n🔍 Setting up vector store...")
    print("   (This may take a few minutes to generate embeddings)")

    from vector_store import VectorStore

    store = VectorStore()
    success = store.build_index()

    if success:
        print("✅ Vector store created in ./chroma_db")
        return True
    else:
        print("❌ Vector store setup failed")
        return False


def test_system():
    """Run basic system tests"""
    print("\n🧪 Testing system...")

    try:
        # Test database
        import duckdb
        conn = duckdb.connect("school_data.duckdb")
        result = conn.execute("SELECT COUNT(*) FROM schools").fetchone()
        school_count = result[0]
        print(f"✅ Database: {school_count} schools loaded")
        conn.close()

        # Test vector store
        from vector_store import VectorStore
        store = VectorStore()
        results = store.search("high performing schools", top_k=3)
        print(f"✅ Vector store: {len(results)} results from test search")

        # Test Claude connection
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            from claude_interface import ClaudeInterface
            claude = ClaudeInterface(api_key)
            print("✅ Claude API: Connection ready")

        print("\n✨ All systems operational!")
        return True

    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False


def main():
    """Main setup process"""
    print("="*60)
    print("🎓 School Data RAG Chatbot - Setup")
    print("="*60)

    # Step 1: Check files
    print("\n1️⃣  Checking required files...")
    if not check_files():
        print("\n⚠️  Please ensure Excel files are in the current directory")
        sys.exit(1)

    print("✅ All required files found")

    # Step 2: Check API key (optional for data setup)
    print("\n2️⃣  Checking API configuration...")
    api_configured = check_api_key()
    if not api_configured:
        print("\n⚠️  API key not configured. You can still set up the database.")
        print("   But you'll need the API key to run the chatbot.")

        response = input("\n   Continue setup without API key? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    # Step 3: Setup database
    print("\n3️⃣  Setting up database...")
    if not setup_database():
        print("\n❌ Setup failed at database creation")
        sys.exit(1)

    # Step 4: Setup vector store
    print("\n4️⃣  Setting up vector store...")
    if not setup_vector_store():
        print("\n⚠️  Vector store setup failed, but you can still use the chatbot")
        print("   (Semantic search will be disabled)")

    # Step 5: Test system
    print("\n5️⃣  Testing system...")
    test_system()

    # Done!
    print("\n" + "="*60)
    print("✨ Setup Complete!")
    print("="*60)
    print("\nYou can now run the chatbot with:")
    print("   streamlit run app_rag.py")
    print("\nOr test individual components:")
    print("   python data_processor.py")
    print("   python vector_store.py")
    print("   python query_engine.py")
    print("\nFor help, see README_RAG.md")
    print("="*60)


if __name__ == "__main__":
    main()
