# Windows Installation Guide

You hit a Windows Long Path limitation. Here are **3 solutions**:

## ✅ **SOLUTION 1: Use Batch Installer (Easiest)**

Simply run:
```bash
install_rag.bat
```

This installs packages in smaller groups to avoid the path issue.

---

## ✅ **SOLUTION 2: Use Minimal Requirements**

Install core packages only (skip optional vector search):

```bash
pip install -r requirements_rag_minimal.txt
```

This installs everything except ChromaDB and sentence-transformers.
**The system will still work!** It just won't have semantic search (uses SQL only).

---

## ✅ **SOLUTION 3: Enable Windows Long Paths (Recommended)**

### Option A: Registry Edit (Admin Required)

1. Press `Win + R`, type `regedit`, press Enter
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Find or create: `LongPathsEnabled` (DWORD)
4. Set value to: `1`
5. Restart your computer
6. Run: `pip install -r requirements_rag.txt`

### Option B: Group Policy (Windows 10/11 Pro)

1. Press `Win + R`, type `gpedit.msc`, press Enter
2. Navigate to: `Computer Configuration > Administrative Templates > System > Filesystem`
3. Enable: "Enable Win32 long paths"
4. Restart your computer
5. Run: `pip install -r requirements_rag.txt`

---

## 🚀 **Quick Start After Installation**

Once packages are installed:

### 1. Create `.env` file
```
ANTHROPIC_API_KEY=your_key_here
```

### 2. Run setup
```bash
python setup_rag.py
```

This creates:
- `school_data.duckdb` (database)
- `chroma_db/` (vector store - if chromadb installed)
- `data_summaries.json` (statistics)

### 3. Launch chatbot
```bash
streamlit run app_rag.py
```

---

## ❓ **Troubleshooting**

### "ModuleNotFoundError: No module named 'dotenv'"

Installation didn't complete. Try Solution 1 or 2 above.

### "No module named 'chromadb'"

This is optional! The system works without it. You'll just see:
```
ℹ️ Running in SQL-only mode (vector search not available)
```

To install later (if you want semantic search):
```bash
pip install chromadb sentence-transformers
python setup_rag.py  # Re-run to build vector index
```

### "Database creation failed"

Make sure Excel files are in the same directory:
- `2025 CMAS Performance_ELA.xlsx`
- `2025 CMAS Performance_Math.xlsx`

### "API key not found"

Create `.env` file with:
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 📦 **What Gets Installed**

### Core (Required)
- `streamlit` - Web interface
- `anthropic` - Claude API
- `pandas` - Data processing
- `duckdb` - Database
- `python-dotenv` - Environment variables

### Optional (Nice to Have)
- `chromadb` - Vector database (for semantic search)
- `sentence-transformers` - Embeddings (for semantic search)

**The system works great with just the core packages!**

---

## 💡 **Recommended Installation Order**

If you're having issues, install in this order:

```bash
# Step 1: Core packages
pip install streamlit anthropic pandas openpyxl python-dotenv

# Step 2: Database
pip install duckdb sqlalchemy

# Step 3: Analysis
pip install plotly numpy scipy

# Step 4: Optional vector search (skip if problems)
pip install chromadb sentence-transformers
```

Then run:
```bash
python setup_rag.py
streamlit run app_rag.py
```

---

## ✅ **You're Ready When You See:**

```
✨ Setup Complete!

You can now run the chatbot with:
   streamlit run app_rag.py
```

---

**Still having issues?** Try the batch installer: `install_rag.bat`
