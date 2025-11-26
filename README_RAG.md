# 🎓 School Data RAG Chatbot

Complete Retrieval-Augmented Generation (RAG) system for analyzing Colorado charter school CMAS performance data using Claude AI.

## ✨ Features

### Core Capabilities
- **Natural Language Queries**: Ask questions in plain English
- **Intelligent Routing**: Automatically determines best approach (SQL vs. AI analysis)
- **Hybrid Retrieval**: Combines SQL queries + semantic search for accurate answers
- **Prompt Caching**: 90% cost reduction through Claude prompt caching
- **Response Caching**: Instant answers for repeated questions
- **Conversation Memory**: Maintains context across multiple questions
- **Follow-up Suggestions**: AI-generated relevant next questions

### Data Access
- **Complete Dataset**: All 1,265 schools with full metadata
- **Fast SQL Queries**: Sub-second responses for factual lookups
- **Semantic Search**: Find schools by characteristics, not just exact matches
- **Pre-computed Statistics**: Instant access to common aggregations

## 🏗️ Architecture

```
User Question
     ↓
Query Classifier (determines type)
     ↓
┌────────────────┬─────────────────┐
│   Factual      │   Analytical    │
│   Lookup       │   Question      │
├────────────────┼─────────────────┤
│ SQL Generation │ Hybrid Retrieval│
│ (Claude)       │ (SQL + Vectors) │
│     ↓          │        ↓        │
│ DuckDB Query   │ Context Builder │
│     ↓          │        ↓        │
│ Format Results │ Claude Analysis │
└────────────────┴─────────────────┘
          ↓
    Response Cache
          ↓
    User Response
```

## 📦 Components

### 1. Data Layer (`data_processor.py`)
- **DuckDB Database**: Optimized for analytical queries
- **Indexed Tables**: Fast lookups on school_name, network, district, FRL%
- **Materialized Views**: Pre-aggregated network and FRL band statistics
- **Tercile Calculations**: Performance relative to demographics

### 2. Vector Store (`vector_store.py`)
- **ChromaDB**: Persistent vector database
- **SentenceTransformers**: Local embedding generation (no API costs)
- **Rich School Profiles**: Combines name, network, performance, demographics
- **Semantic Search**: Find schools by meaning, not just keywords

### 3. Query Engine (`query_engine.py`)
- **SQL Generation**: Claude generates DuckDB queries
- **Query Classification**: Routes queries efficiently
- **Safe Execution**: Prevents SQL injection, allows SELECT only
- **Result Formatting**: Human-readable responses

### 4. Claude Interface (`claude_interface.py`)
- **Prompt Caching**: Caches schema + guidelines (saves 90% tokens)
- **Conversation Management**: Maintains context across turns
- **Error Handling**: Graceful degradation on API failures
- **Streaming Support**: Real-time response generation

### 5. Cache Manager (`cache_manager.py`)
- **SQLite Cache**: Fast local caching
- **TTL Management**: Automatic expiration
- **Hit Tracking**: Monitor cache effectiveness
- **Query Normalization**: Matches similar questions

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
# pip installed
```

### Installation

1. **Clone/Download the project**

2. **Install dependencies**
```bash
pip install -r requirements_rag.txt
```

3. **Configure API Key**

Create `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

Or use Streamlit secrets (for deployment):
```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "your_key_here"
```

4. **Place Data Files**

Ensure these files are in the project directory:
- `2025 CMAS Performance_ELA.xlsx`
- `2025 CMAS Performance_Math.xlsx`

5. **Run Setup**
```bash
python setup_rag.py
```

This will:
- Create DuckDB database (`school_data.duckdb`)
- Generate vector embeddings (`./chroma_db/`)
- Create statistical summaries (`data_summaries.json`)
- Run system tests

6. **Launch Chatbot**
```bash
streamlit run app_rag.py
```

## 💬 Example Questions

### Factual Lookups (Fast SQL)
```
"Which district is Vega Collegiate Academy from?"
"How many charter schools serve more than 70% FRL?"
"List schools in KIPP Colorado network"
"Show me all schools in Denver County 1"
```

### Analytical (AI Analysis)
```
"Why do some high-poverty schools outperform?"
"What's the correlation between FRL and Math performance?"
"Compare KIPP Colorado to DSST Public Schools"
"Which elementary charter schools are beating expectations?"
```

### Complex Queries (Hybrid RAG)
```
"Find single site charter schools with FRL > 50% in top third for ELA"
"Show me high-performing schools similar to Rocky Mountain Prep"
"What strategies work for schools with 80%+ FRL?"
"Identify outliers: low-poverty schools performing poorly"
```

## 📊 System Files

### Created by Setup
```
school_data.duckdb       # Main database (DuckDB)
chroma_db/              # Vector embeddings (ChromaDB)
data_summaries.json     # Pre-computed statistics
query_cache.db          # Response cache (SQLite)
```

### Configuration
```
.env                    # API keys (not committed)
.env.example           # Template for .env
requirements_rag.txt   # Python dependencies
```

### Core Modules
```
data_processor.py      # Data loading & database creation
vector_store.py        # Embeddings & semantic search
query_engine.py        # SQL generation & execution
claude_interface.py    # Claude API wrapper
cache_manager.py       # Response caching
utils.py              # Helper functions
app_rag.py            # Main Streamlit app
setup_rag.py          # Setup script
```

## 🔧 Configuration

### Caching Settings

**Response Cache TTL**:
- Factual queries: 1 hour (3600s)
- Analytical queries: 30 minutes (1800s)

Modify in `cache_manager.py`:
```python
cache = CacheManager(default_ttl=3600)
```

### Embedding Model

Default: `all-MiniLM-L6-v2` (fast, good quality, runs locally)

Change in `vector_store.py`:
```python
self.embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Better quality, slower
```

### Claude Model

Default: `claude-3-5-sonnet-20241022` (latest with caching)

Change in `claude_interface.py`:
```python
self.model = "claude-3-haiku-20240307"  # Faster, cheaper
```

## 💰 Cost Optimization

### Prompt Caching (Biggest Savings)
- Schema + guidelines cached: ~10K tokens
- Cache hit: $0.30 per 1M tokens (90% cheaper)
- Cache miss: $3.00 per 1M tokens

**Estimated savings**: ~90% on repeated queries

### Smart Routing
- Factual queries → SQL only (no Claude API call)
- ~60-70% of queries can skip AI entirely

### Response Caching
- Common questions cached locally
- Zero API cost for cache hits
- TTL prevents stale data

### Cost Examples

**Without optimization**:
- 100 queries/day × 20K tokens each = 2M tokens/day
- Cost: 2M × $3.00 / 1M = **$6.00/day**

**With RAG system**:
- 60 queries via SQL (free)
- 40 queries via Claude:
  - 10K cached tokens (90% hit rate)
  - Cost: 40 × 10K × $0.30 / 1M = **$0.12/day**

**Savings: ~98%**

## 🧪 Testing Components

### Test Database
```bash
python data_processor.py
```

### Test Vector Store
```bash
python vector_store.py
```

### Test Query Engine
```bash
# Requires API key
python query_engine.py
```

### Test Claude Interface
```bash
# Requires API key
python claude_interface.py
```

### Test Cache
```bash
python cache_manager.py
```

## 📈 Monitoring

### Cache Statistics

View in app sidebar:
- Cached queries count
- Total cache hits
- Average hits per query
- Most popular queries

Or programmatically:
```python
from cache_manager import CacheManager
cache = CacheManager()
stats = cache.get_stats()
print(stats)
```

### Database Queries

```python
import duckdb
conn = duckdb.connect("school_data.duckdb")

# Total schools
conn.execute("SELECT COUNT(*) FROM schools").fetchone()

# Charter schools
conn.execute("SELECT COUNT(*) FROM charter_schools").fetchone()

# Top networks
conn.execute("SELECT * FROM network_summary LIMIT 10").fetchall()
```

## 🔒 Security

### API Key Protection
- Never commit `.env` file
- Use Streamlit secrets for deployment
- Validate API key exists before calls

### SQL Injection Prevention
- Only SELECT queries allowed
- Parameterized queries when possible
- Dangerous keywords blocked

### Rate Limiting
- Claude API: Built-in rate limits
- Cache reduces API calls
- Graceful error handling

## 🚀 Deployment (Streamlit Cloud)

1. **Push to GitHub**
```bash
git add .
git commit -m "Add RAG chatbot system"
git push
```

2. **Deploy on Streamlit Cloud**
- Go to https://share.streamlit.io
- Select repository
- **Main file**: `app_rag.py`
- Add secrets:
  ```toml
  ANTHROPIC_API_KEY = "your_key_here"
  ```

3. **Upload Data Files**
- Include Excel files in repository OR
- Load from Google Drive/S3

4. **First Deploy**
- App will run `setup_rag.py` automatically
- Database will be created on first run
- Vector store will initialize

## 🛠️ Troubleshooting

### "No database found"
```bash
python setup_rag.py
```

### "Vector index not found"
- Vector store is optional
- Run setup to create it
- App works without it (SQL-only mode)

### "API key not found"
- Check `.env` file exists
- Verify `ANTHROPIC_API_KEY` is set
- For Streamlit Cloud, check secrets

### "Model not found (404)"
- Billing not set up on Anthropic account
- Visit: https://console.anthropic.com/settings/billing
- Add payment method

### Slow performance
- Check cache hit rate (should be >50%)
- Verify database indexes exist
- Consider using Haiku model for speed

## 📚 Advanced Usage

### Custom Queries

```python
from query_engine import QueryEngine

engine = QueryEngine(api_key="your_key")
engine.connect()

# Direct SQL
sql = "SELECT * FROM schools WHERE frl_percent > 80"
success, results, error = engine.execute_query(sql)

# AI-generated SQL
success, answer, data = engine.query_to_answer(
    "Find high-poverty schools in top third",
    schema
)
```

### Semantic Search

```python
from vector_store import VectorStore

store = VectorStore()
results = store.search(
    "high performing schools with high poverty",
    top_k=10
)

for school in results:
    print(f"{school['school_name']}: {school['similarity_score']:.3f}")
```

### Direct Claude Access

```python
from claude_interface import ClaudeInterface

claude = ClaudeInterface(api_key="your_key")

response = claude.chat(
    user_message="Analyze KIPP schools",
    context_data="[relevant data here]",
    schema=schema
)
```

## 🤝 Contributing

To extend the system:

1. **Add new data sources**: Modify `data_processor.py`
2. **Custom embeddings**: Update `vector_store.py`
3. **Query templates**: Extend `query_engine.py`
4. **UI features**: Modify `app_rag.py`

## 📄 License

Educational use. Customize as needed.

## 🙏 Acknowledgments

Built with:
- **Anthropic Claude**: AI analysis and SQL generation
- **DuckDB**: High-performance analytics database
- **ChromaDB**: Vector database
- **Sentence Transformers**: Local embeddings
- **Streamlit**: Web interface

---

**Questions?** Check the troubleshooting section or review individual module docstrings.

**Ready to start?** Run `python setup_rag.py` then `streamlit run app_rag.py`!
