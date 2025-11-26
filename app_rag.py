"""
RAG-Powered School Data Chatbot
Complete implementation with Claude API, DuckDB, and ChromaDB
"""

import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
import time

# Import our modules
from data_processor import DataProcessor
from vector_store import VectorStore
from query_engine import QueryEngine
from claude_interface import ClaudeInterface
from cache_manager import CacheManager
from utils import (
    format_school_list,
    create_scatter_plot,
    create_bar_chart,
    detect_visualization_intent,
    export_to_csv,
    generate_example_questions,
    format_conversation_for_export
)

# Page configuration
st.set_page_config(
    page_title="School Data AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()


@st.cache_resource
def initialize_system():
    """Initialize all components (cached for performance)"""

    with st.spinner("🔧 Initializing AI system..."):
        # Get API key
        api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")

        if not api_key:
            st.error("⚠️ No Anthropic API key found. Please set ANTHROPIC_API_KEY in .env or Streamlit secrets.")
            return None

        # Check if database exists
        db_path = "school_data.duckdb"
        if not Path(db_path).exists():
            st.info("📊 No database found. Please run setup first.")
            return None

        # Initialize components
        processor = DataProcessor(db_path)
        schema = processor.get_schema()

        # Try to initialize vector store (optional)
        vector_store = None
        try:
            from vector_store import VectorStore
            vector_store = VectorStore(db_path)

            # Check if vector index exists
            try:
                vector_store.collection = vector_store.client.get_collection("schools")
            except:
                st.warning("⚠️ Vector index not found. Semantic search disabled. (Run setup_rag.py to enable)")
                vector_store = None
        except ImportError:
            st.info("ℹ️ Running in SQL-only mode (vector search not available)")
            vector_store = None

        query_engine = QueryEngine(db_path, api_key)
        claude = ClaudeInterface(api_key)
        cache = CacheManager()

        return {
            'processor': processor,
            'schema': schema,
            'vector_store': vector_store,
            'query_engine': query_engine,
            'claude': claude,
            'cache': cache
        }


def process_query(user_question: str, system: dict, conversation_history: list):
    """
    Main query processing pipeline

    Args:
        user_question: User's question
        system: Dictionary of system components
        conversation_history: Previous messages

    Returns:
        Response string
    """

    # 1. Check cache first
    cached_response = system['cache'].get(user_question)
    if cached_response:
        st.info(f"💾 Retrieved from cache (hit #{cached_response['hit_count']})")
        return cached_response['response']

    # 2. Classify query
    classification = system['query_engine'].classify_query(user_question)
    query_type = classification['type']

    # Show query type
    with st.expander("🔍 Query Analysis", expanded=False):
        st.write(f"**Type**: {query_type}")
        st.write(f"**Requires SQL**: {classification['requires_sql']}")
        st.write(f"**Requires Reasoning**: {classification['requires_reasoning']}")

    # 3. Route based on query type
    if query_type == 'factual_lookup' and classification['requires_sql']:
        # Direct SQL approach (fastest, cheapest)
        st.info("🔎 Using SQL query...")

        success, answer, data = system['query_engine'].query_to_answer(
            user_question,
            system['schema']
        )

        if success:
            # Cache the response
            system['cache'].set(
                user_question,
                answer,
                metadata={'type': 'sql_direct', 'data_points': len(data) if data else 0},
                ttl=3600
            )

            # Check if visualization would be helpful
            if data and len(data) > 1:
                viz_type = detect_visualization_intent(user_question, data)
                if viz_type:
                    st.session_state['last_viz_data'] = data
                    st.session_state['last_viz_type'] = viz_type

            return answer
        else:
            st.warning("SQL query failed, using full RAG pipeline...")

    # 4. Full RAG pipeline for analytical/complex questions
    st.info("🧠 Using AI analysis...")

    # Get relevant context (hybrid retrieval)
    context_parts = []

    # A. Try SQL query for structured data
    if classification['requires_sql']:
        sql = system['query_engine'].generate_sql(user_question, system['schema'])
        if sql:
            success, results, error = system['query_engine'].execute_query(sql)
            if success and results:
                context_parts.append(f"SQL QUERY RESULTS:\n{results[:20]}")  # Limit to 20 results

    # B. Semantic search for similar schools/insights
    if system['vector_store']:
        similar = system['vector_store'].search(user_question, top_k=5)
        if similar:
            context_parts.append(f"SIMILAR SCHOOLS:\n{similar}")

    # C. Combine context
    combined_context = "\n\n".join(context_parts) if context_parts else "No specific data retrieved. Answer based on general knowledge of the dataset."

    # 5. Get Claude's response
    response = system['claude'].chat(
        user_message=user_question,
        context_data=combined_context,
        schema=system['schema'],
        conversation_history=conversation_history,
        stream=False
    )

    # Cache the response
    system['cache'].set(
        user_question,
        response,
        metadata={'type': 'rag_full'},
        ttl=1800  # 30 minutes for analytical responses
    )

    return response


def main():
    """Main application"""

    # Initialize system
    system = initialize_system()

    if not system:
        st.error("❌ System initialization failed. Please check setup.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.title("🎓 School Data AI")
        st.markdown("---")

        # Example questions
        st.subheader("💡 Example Questions")
        examples = generate_example_questions()

        for i, example in enumerate(examples[:5], 1):
            if st.button(f"📝 {example[:40]}...", key=f"ex_{i}", use_container_width=True):
                st.session_state['selected_example'] = example

        st.markdown("---")

        # Settings
        with st.expander("⚙️ Settings"):
            show_debug = st.checkbox("Show debug info", value=False)
            enable_cache = st.checkbox("Enable response caching", value=True)

            if not enable_cache:
                system['cache'].clear_all()
                st.success("Cache cleared!")

        # Cache stats
        with st.expander("📊 Cache Statistics"):
            stats = system['cache'].get_stats()
            st.metric("Cached Queries", stats['valid_entries'])
            st.metric("Total Cache Hits", stats['total_hits'])
            st.metric("Avg Hits/Query", stats['avg_hits_per_query'])

            if st.button("🗑️ Clear Cache"):
                cleared = system['cache'].clear_all()
                st.success(f"Cleared {cleared} entries")

        # Export conversation
        if st.session_state.get('messages') and len(st.session_state['messages']) > 0:
            st.markdown("---")
            if st.button("📥 Export Conversation"):
                export_md = format_conversation_for_export(st.session_state['messages'])
                st.download_button(
                    label="Download as Markdown",
                    data=export_md,
                    file_name=f"conversation_{int(time.time())}.md",
                    mime="text/markdown"
                )

    # Main area
    st.title("🤖 School Data AI Assistant")
    st.caption("Ask questions about Colorado charter school CMAS performance data")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
        # Add welcome message
        st.session_state['messages'].append({
            'role': 'assistant',
            'content': """👋 Welcome! I'm your AI assistant for Colorado charter school data.

I can help you:
- Look up specific schools and districts
- Compare performance across networks
- Analyze relationships between demographics and performance
- Find schools matching specific criteria
- Provide insights and recommendations

Try asking a question or click an example on the left!"""
        })

    # Handle example selection
    if 'selected_example' in st.session_state:
        example = st.session_state['selected_example']
        st.session_state['messages'].append({'role': 'user', 'content': example})
        del st.session_state['selected_example']
        st.rerun()

    # Display chat history
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input
    if prompt := st.chat_input("Ask a question about the schools..."):
        # Add user message
        st.session_state['messages'].append({'role': 'user', 'content': prompt})

        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)

        # Generate response
        with st.chat_message('assistant'):
            with st.spinner("🤔 Thinking..."):
                # Get response
                response = process_query(
                    prompt,
                    system,
                    st.session_state['messages']
                )

                # Display response
                st.markdown(response)

                # Add to history
                st.session_state['messages'].append({
                    'role': 'assistant',
                    'content': response
                })

                # Show visualization if available
                if 'last_viz_data' in st.session_state:
                    data = st.session_state['last_viz_data']
                    viz_type = st.session_state['last_viz_type']

                    if viz_type == 'scatter' and len(data) > 1:
                        st.subheader("📊 Visualization")
                        # Try to create scatter plot
                        # This is simplified - in production, auto-detect best columns
                        st.info("Scatter plot visualization would appear here")

                    # Clear visualization state
                    del st.session_state['last_viz_data']
                    del st.session_state['last_viz_type']

        # Generate follow-up questions
        if len(st.session_state['messages']) >= 2:
            with st.expander("💡 Suggested follow-up questions"):
                follow_ups = system['claude'].generate_follow_up_questions(
                    question=prompt,
                    answer=response,
                    context="Recent conversation"
                )

                for i, fq in enumerate(follow_ups, 1):
                    if st.button(fq, key=f"fq_{i}"):
                        st.session_state['selected_example'] = fq
                        st.rerun()


if __name__ == "__main__":
    main()
