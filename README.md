# Colorado School CMAS Performance Dashboard 📊

An interactive web-based dashboard visualizing Colorado school CMAS (Colorado Measures of Academic Success) performance data for the 2024-25 academic year.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🌟 Features

### Interactive Visualizations
- **Side-by-side scatterplots** comparing ELA and Math performance
- **Color-coded performance terciles** (top third, middle third, bottom third)
- **Dynamic best-fit regression lines** with R² values
- **Performance reference bands** showing tercile boundaries

### Advanced Filtering
- 🏫 **School Selector**: Highlight individual schools
- 🌐 **Network Selector**: Highlight all schools in a CMO/network
- 📚 **Gradespan Filter**: Filter by Elementary, Middle, High, or Multiple
- ⭐ **CSF Portfolio Filter**: Highlight Colorado Schools Fund schools
- 🎓 **Charter Schools Filter**: Show only charter schools

### Data Insights
- Real-time summary statistics
- Hover tooltips with detailed school information
- Performance tercile reference lines
- Correlation analysis via regression lines
- CSV data export functionality

## 📋 Prerequisites

- Python 3.9 or higher
- Excel files with CMAS performance data:
  - `2025 CMAS Performance_ELA.xlsx`
  - `2025 CMAS Performance_Math.xlsx`

## 🚀 Quick Start

### Local Development

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your data files**
   - Ensure both Excel files are in the same directory as `app.py`

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

## ☁️ Deploying to Streamlit Community Cloud (FREE)

Streamlit Community Cloud offers **completely free hosting** with shareable links!

### Step 1: Prepare Your GitHub Repository

1. **Create a new GitHub repository**
   - Go to https://github.com/new
   - Name it (e.g., `cmas-performance-dashboard`)
   - Make it public or private (both work)

2. **Upload your files to GitHub**
   ```bash
   # Initialize git (if not already done)
   git init

   # Add all files
   git add app.py requirements.txt README.md
   git add "2025 CMAS Performance_ELA.xlsx"
   git add "2025 CMAS Performance_Math.xlsx"

   # Commit
   git commit -m "Initial commit: CMAS Performance Dashboard"

   # Add remote and push
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Sign in with GitHub**
   - Click "Sign in with GitHub"
   - Authorize Streamlit

3. **Deploy your app**
   - Click "New app"
   - Select your repository
   - Select the branch (usually `main`)
   - Set main file path: `app.py`
   - Click "Deploy!"

4. **Wait for deployment** (2-3 minutes)
   - Streamlit will install dependencies and launch your app

5. **Share your app!**
   - You'll get a URL like: `https://YOUR-APP-NAME.streamlit.app`
   - Share this link with anyone!

### Updating Your Deployed App

Any push to your GitHub repository will automatically update your Streamlit app!

```bash
# Make changes to app.py
git add .
git commit -m "Update dashboard features"
git push
```

## 📊 Data File Format

The dashboard expects the following columns in each Excel file:

### ELA and Math Files
- **School Name**: Name of the school
- **Network**: School network or CMO affiliation
- **Percent FRL**: Percentage of Free and Reduced Lunch students (0-100)
- **School Performance Value**: CMAS Met or Exceeded Expectations percentage (0-100)
- **Gradespan**: Grade levels served (e.g., "K-5", "6-8", "9-12", "K-8")
- **CSF Portfolio**: "CSF" or "Not CSF"
- **School Type**: "Charter" or "District"

### Example Data Format
```
| School Name       | Network    | Percent FRL | School Performance Value | Gradespan | CSF Portfolio | School Type |
|-------------------|------------|-------------|--------------------------|-----------|---------------|-------------|
| Example School 1  | Network A  | 45.2        | 67.8                     | K-5       | CSF           | Charter     |
| Example School 2  | Network B  | 78.5        | 54.3                     | 6-8       | Not CSF       | District    |
```

## 🎨 Visual Design

### Color Coding
- 🔵 **Dark Blue** (#1f4788): Top third performers
- 🔷 **Teal** (#4cb5c4): Middle third performers
- ⚪ **Light Gray** (#D3D3D3): Bottom third performers
- 🟠 **Orange** (#FF6B35): Selected school highlight
- 🟡 **Gold**: CSF Portfolio schools (when filter enabled)
- 🟣 **Purple**: Network schools (when filter active)

### Plot Features
- **Reference bands**: Subtle background shading for terciles
- **Percentile lines**: Dotted lines at 33rd and 67th percentiles
- **Regression line**: Red dashed line with R² value
- **Hover tooltips**: Detailed information on mouse-over

## 🔧 Customization

### Adjusting Performance Categories

To change from terciles to quartiles, modify the `calculate_terciles()` function:

```python
def calculate_quartiles(performance_values):
    """Calculate performance quartiles"""
    valid_values = performance_values[~pd.isna(performance_values)]
    q25 = np.percentile(valid_values, 25)
    q50 = np.percentile(valid_values, 50)
    q75 = np.percentile(valid_values, 75)
    return q25, q50, q75
```

### Changing Color Scheme

Colors are defined in the `assign_tercile_color()` function and scatter plot marker properties. Modify these values to match your brand colors.

## 📈 Features Breakdown

### 1. School Selector
Select individual schools from a searchable dropdown to highlight them in bright orange on both ELA and Math plots.

### 2. Network Selector
Choose a school network/CMO to highlight all affiliated schools in purple.

### 3. Gradespan Filter
Filter schools by grade level:
- **Elementary**: K-5
- **Middle**: 6-8
- **High**: 9-12
- **Multiple**: Schools serving multiple levels
- **All**: Show all schools

**Important**: The regression line updates dynamically based on the selected gradespan.

### 4. CSF Portfolio Highlight
Enable to highlight all Colorado Schools Fund portfolio schools with a gold marker.

### 5. Charter Schools Filter
Toggle to show only charter schools in the dataset.

### 6. Summary Statistics
Real-time display of:
- Number of schools currently displayed
- Average FRL percentage
- Average ELA performance
- Average Math performance

### 7. Data Export
Download filtered or complete datasets as CSV files for further analysis.

## 🛠️ Technical Details

### Built With
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **SciPy**: Statistical analysis (linear regression)
- **OpenPyXL**: Excel file reading

### Performance Optimization
- Data caching with `@st.cache_data` for faster load times
- Efficient data filtering using pandas
- Responsive layout for various screen sizes

## 🐛 Troubleshooting

### App won't start locally
```bash
# Ensure all dependencies are installed
pip install --upgrade -r requirements.txt

# Check Python version (3.9+ required)
python --version
```

### Data not loading
- Verify Excel files are in the same directory as `app.py`
- Check that column names match exactly (case-sensitive)
- Ensure no corrupt Excel files

### Streamlit Cloud deployment fails
- Check that `requirements.txt` is in the root directory
- Verify all file paths are relative (not absolute)
- Ensure Excel files are pushed to GitHub
- Check Streamlit Cloud logs for specific errors

### Charts not displaying correctly
- Clear browser cache
- Try a different browser
- Check that data contains valid numeric values for FRL and Performance

## 📝 License

This project is open source and available for educational and analytical purposes.

## 🤝 Contributing

Suggestions and improvements are welcome! Feel free to:
- Open an issue for bugs or feature requests
- Submit pull requests with enhancements
- Share feedback on the dashboard design

## 📧 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review Streamlit documentation: https://docs.streamlit.io
3. Check Plotly documentation: https://plotly.com/python/

## 🎯 Future Enhancements

Potential features for future versions:
- [ ] Year-over-year comparison (when historical data available)
- [ ] Side-by-side school comparison tool
- [ ] Additional demographic filters
- [ ] Growth metrics visualization
- [ ] Downloadable chart images
- [ ] Custom color theme selector
- [ ] District-level aggregation views

---

**Powered by Streamlit** | **Data Source: 2024-25 CMAS Assessments**

Made with ❤️ for Colorado educators and education advocates
