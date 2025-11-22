# 💻 Local Setup Guide

Quick guide to run the CMAS Performance Dashboard on your local computer.

## 🔧 Prerequisites

1. **Python 3.9 or higher**
   - Check version: `python --version` or `python3 --version`
   - Download: https://www.python.org/downloads/

2. **pip (Python package manager)**
   - Usually comes with Python
   - Check: `pip --version`

## 📥 Installation Steps

### Step 1: Download/Clone the Project

**Option A: Download ZIP**
1. Download all project files to a folder
2. Extract if zipped

**Option B: Clone from GitHub**
```bash
git clone https://github.com/YOUR_USERNAME/cmas-performance-dashboard.git
cd cmas-performance-dashboard
```

### Step 2: Install Dependencies

Open terminal/command prompt in the project folder:

**Windows:**
```bash
cd "C:\Users\SudhanshuMalani\Documents\2025 2x2 Analysis"
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
cd ~/Documents/2025\ 2x2\ Analysis/
pip3 install -r requirements.txt
```

**Using Virtual Environment (Recommended):**
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Data Files

Ensure these files are in the same folder as `app.py`:
- ✅ `2025 CMAS Performance_ELA.xlsx`
- ✅ `2025 CMAS Performance_Math.xlsx`

### Step 4: Run the App

```bash
streamlit run app.py
```

The app should automatically open in your browser at `http://localhost:8501`

## 🎯 Quick Test

Once the app loads:
1. ✅ You should see two side-by-side scatter plots
2. ✅ Sidebar should show filters
3. ✅ Try selecting a school from the dropdown
4. ✅ Test the gradespan filter
5. ✅ Verify the charts update

## 🐛 Troubleshooting

### Issue: "Python not found"

**Windows:**
1. Download Python from https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Restart command prompt

**Mac:**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python
```

### Issue: "streamlit: command not found"

```bash
# Install Streamlit directly
pip install streamlit

# Or reinstall all requirements
pip install -r requirements.txt
```

### Issue: "ModuleNotFoundError: No module named 'openpyxl'"

```bash
pip install openpyxl
```

### Issue: "File not found" for Excel files

Make sure the Excel files are in the **exact same folder** as `app.py`:

```bash
# Check current directory
ls    # Mac/Linux
dir   # Windows

# You should see:
# - app.py
# - 2025 CMAS Performance_ELA.xlsx
# - 2025 CMAS Performance_Math.xlsx
```

### Issue: App won't stop running

Press `Ctrl+C` in the terminal to stop the app

## 🔄 Making Changes

1. Edit `app.py` in your favorite code editor
2. Save the file
3. Streamlit will auto-detect changes and prompt you to rerun
4. Click "Rerun" or press `R` in the terminal

## 📁 Project Structure

```
2025 2x2 Analysis/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── README.md                           # Full documentation
├── DEPLOYMENT_GUIDE.md                 # Streamlit Cloud deployment
├── LOCAL_SETUP_GUIDE.md               # This file
├── .gitignore                          # Git ignore rules
├── .streamlit/
│   └── config.toml                     # Streamlit configuration
├── 2025 CMAS Performance_ELA.xlsx     # ELA performance data
└── 2025 CMAS Performance_Math.xlsx    # Math performance data
```

## 🎨 Customization Tips

### Change Colors
Edit `app.py`, search for color definitions:
- Line ~197: Tercile colors
- Line ~250-280: Highlight colors

### Modify Filters
Edit sidebar section (lines ~460-520)

### Adjust Layout
Change column ratios in line ~580:
```python
col1, col2 = st.columns(2)  # Equal width
# or
col1, col2 = st.columns([2, 1])  # 2:1 ratio
```

## 📊 Testing with Sample Data

To test without real data, you can create sample Excel files:

**Python script to generate sample data:**
```python
import pandas as pd
import numpy as np

# Create sample data
n_schools = 50
data = {
    'School Name': [f'School {i}' for i in range(n_schools)],
    'Network': np.random.choice(['Network A', 'Network B', 'Network C'], n_schools),
    'Percent FRL': np.random.uniform(20, 90, n_schools),
    'School Performance Value': np.random.uniform(30, 85, n_schools),
    'Gradespan': np.random.choice(['K-5', '6-8', '9-12', 'K-8'], n_schools),
    'CSF Portfolio': np.random.choice(['CSF', 'Not CSF'], n_schools),
    'School Type': np.random.choice(['Charter', 'District'], n_schools)
}

df = pd.DataFrame(data)
df.to_excel('2025 CMAS Performance_ELA.xlsx', index=False)
df['School Performance Value'] = np.random.uniform(30, 85, n_schools)
df.to_excel('2025 CMAS Performance_Math.xlsx', index=False)

print("Sample data created!")
```

## 🚀 Performance Tips

### Speed up loading
- Data is cached automatically with `@st.cache_data`
- First load may be slow, subsequent loads are fast

### Clear cache
If data seems stale:
```bash
streamlit cache clear
```

Or click the "Clear cache" button in the app's hamburger menu (top right)

## 📝 Common Commands

```bash
# Run the app
streamlit run app.py

# Run on different port
streamlit run app.py --server.port 8502

# Run without auto-opening browser
streamlit run app.py --server.headless true

# Clear cache
streamlit cache clear

# Show Streamlit version
streamlit --version

# Get help
streamlit --help
```

## 🔗 Useful Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Plotly Docs**: https://plotly.com/python/
- **Pandas Docs**: https://pandas.pydata.org/docs/

## ✅ Success Checklist

- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Excel files in project folder
- [ ] App runs with `streamlit run app.py`
- [ ] Charts display correctly
- [ ] All filters work
- [ ] Data exports successfully

## 🎓 Next Steps

1. **Familiarize yourself** with all filters and features
2. **Test with your data** to ensure compatibility
3. **Customize colors** to match your branding
4. **Deploy to Streamlit Cloud** (see DEPLOYMENT_GUIDE.md)
5. **Share with stakeholders** and gather feedback

---

**Need help?** Check the main README.md or open an issue on GitHub.

**Happy coding!** 🎉
