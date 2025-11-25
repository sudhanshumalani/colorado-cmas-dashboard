"""
Colorado School CMAS Performance Dashboard
Interactive visualization of 2024-25 CMAS performance data for ELA and Math
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Colorado CMAS Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    """Load and preprocess CMAS performance data"""
    try:
        # Load the data files
        ela_df = pd.read_excel('2025 CMAS Performance_ELA.xlsx')
        math_df = pd.read_excel('2025 CMAS Performance_Math.xlsx')

        # Standardize column names for consistency
        # ELA has 'School type' (lowercase), Math has 'School Type' (capital)
        if 'School type' in ela_df.columns:
            ela_df = ela_df.rename(columns={'School type': 'School_Type'})
        if 'School Type' in ela_df.columns:
            ela_df = ela_df.rename(columns={'School Type': 'School_Type'})
        if 'School Type' in math_df.columns:
            math_df = math_df.rename(columns={'School Type': 'School_Type'})

        # Rename other columns for consistency
        ela_df = ela_df.rename(columns={
            'School Performance Value': 'ELA_Performance',
            'Percent FRL': 'FRL_Percent',
            'Gradespan Level': 'Gradespan'
        })

        math_df = math_df.rename(columns={
            'School Performance Value': 'Math_Performance',
            'Percent FRL': 'FRL_Percent',
            'Gradespan Level': 'Gradespan'
        })

        # Select columns for merge (CSF Portfolio only in ELA file)
        ela_columns = ['School Name', 'Network', 'FRL_Percent', 'ELA_Performance', 'Gradespan']

        # Add CSF Portfolio if it exists
        if 'CSF Portfolio' in ela_df.columns:
            ela_columns.append('CSF Portfolio')
        else:
            ela_df['CSF Portfolio'] = 'Unknown'
            ela_columns.append('CSF Portfolio')

        # Add School Type / Charter info
        if 'School_Type' in ela_df.columns:
            ela_columns.append('School_Type')
        elif 'Charter Y/N' in ela_df.columns:
            ela_df['School_Type'] = ela_df['Charter Y/N'].apply(lambda x: 'Charter' if str(x).upper() == 'Y' else 'District')
            ela_columns.append('School_Type')
        else:
            ela_df['School_Type'] = 'Unknown'
            ela_columns.append('School_Type')

        # Merge ELA and Math data on School Name
        df = pd.merge(
            ela_df[ela_columns],
            math_df[['School Name', 'Math_Performance']],
            on='School Name',
            how='outer'
        )

        # Clean and validate data
        df['FRL_Percent'] = pd.to_numeric(df['FRL_Percent'], errors='coerce')
        df['ELA_Performance'] = pd.to_numeric(df['ELA_Performance'], errors='coerce')
        df['Math_Performance'] = pd.to_numeric(df['Math_Performance'], errors='coerce')

        # Convert FRL from decimal (0-1) to percentage (0-100)
        # If FRL values are between 0-1, multiply by 100
        if df['FRL_Percent'].max() <= 1.0:
            df['FRL_Percent'] = df['FRL_Percent'] * 100

        # Ensure values are within 0-100% range
        df['FRL_Percent'] = df['FRL_Percent'].clip(0, 100)
        df['ELA_Performance'] = df['ELA_Performance'].clip(0, 100)
        df['Math_Performance'] = df['Math_Performance'].clip(0, 100)

        # Handle missing values
        df = df.dropna(subset=['School Name'])

        # Standardize gradespan categories
        df['Gradespan_Category'] = df['Gradespan'].apply(categorize_gradespan)

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def categorize_gradespan(gradespan):
    """Categorize gradespan into Elementary, Middle, High, or Multiple"""
    if pd.isna(gradespan):
        return 'Unknown'

    gradespan_str = str(gradespan).upper()

    # Check for simple format: "Elementary School", "Middle School", "High School"
    if 'ELEMENTARY' in gradespan_str:
        return 'Elementary'
    elif 'MIDDLE' in gradespan_str:
        return 'Middle'
    elif 'HIGH' in gradespan_str:
        return 'High'

    # Fallback: Check for grade ranges like K-5, 6-8, 9-12
    # Check for elementary grades (K-5)
    has_elementary = any(g in gradespan_str for g in ['K', '1', '2', '3', '4', '5'])
    # Check for middle grades (6-8)
    has_middle = any(g in gradespan_str for g in ['6', '7', '8'])
    # Check for high grades (9-12)
    has_high = any(g in gradespan_str for g in ['9', '10', '11', '12'])

    categories = []
    if has_elementary:
        categories.append('Elementary')
    if has_middle:
        categories.append('Middle')
    if has_high:
        categories.append('High')

    if len(categories) == 0:
        return 'Unknown'
    elif len(categories) == 1:
        return categories[0]
    else:
        return 'Multiple'

def calculate_terciles_from_residuals(residuals):
    """Calculate terciles based on residuals from regression line"""
    valid_residuals = residuals[~pd.isna(residuals)]
    if len(valid_residuals) < 3:
        return None, None

    # Calculate terciles: top third = most positive residuals, bottom third = most negative
    tercile_33 = np.percentile(valid_residuals, 33.33)
    tercile_67 = np.percentile(valid_residuals, 66.67)

    return tercile_33, tercile_67

def assign_tercile_color_from_residual(residual, tercile_33, tercile_67):
    """Assign color based on residual tercile (distance from trend line)"""
    if pd.isna(residual):
        return '#D3D3D3'  # Light gray for missing data
    elif residual >= tercile_67:
        return '#1f4788'  # Dark blue - top third (performing above trend)
    elif residual >= tercile_33:
        return '#4cb5c4'  # Teal - middle third (near trend)
    else:
        return '#D3D3D3'  # Light gray - bottom third (performing below trend)

def calculate_regression(x, y):
    """Calculate linear regression and return slope, intercept, and R²"""
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return None, None, None

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    r_squared = r_value ** 2

    return slope, intercept, r_squared

def create_scatter_plot(df_filtered, df_for_trendline, subject, selected_school, selected_network, highlight_csf):
    """Create scatter plot for ELA or Math performance with residual-based terciles

    Args:
        df_filtered: Data to display on the plot
        df_for_trendline: Data to use for calculating the regression line (may be same or all schools)
        subject: 'ELA' or 'Math'
        selected_school: School name to highlight
        selected_network: Network to highlight
        highlight_csf: Whether to highlight CSF schools
    """

    performance_col = f'{subject}_Performance'

    # Prepare data for display
    x = df_filtered['FRL_Percent'].values
    y = df_filtered[performance_col].values

    # Calculate regression line based on trendline dataset (could be all schools or filtered)
    x_trend = df_for_trendline['FRL_Percent'].values
    y_trend = df_for_trendline[performance_col].values
    slope, intercept, r_squared = calculate_regression(x_trend, y_trend)

    # Calculate residuals for displayed schools based on the trendline
    residuals = np.full(len(y), np.nan)
    if slope is not None and intercept is not None:
        # Predicted values based on regression from trendline dataset
        y_predicted = slope * x + intercept
        # Residuals = actual - predicted (positive = above trend, negative = below trend)
        residuals = y - y_predicted

    # Calculate terciles based on residuals of the TRENDLINE dataset
    residuals_trend = np.full(len(y_trend), np.nan)
    if slope is not None and intercept is not None:
        y_predicted_trend = slope * x_trend + intercept
        residuals_trend = y_trend - y_predicted_trend

    tercile_33, tercile_67 = calculate_terciles_from_residuals(residuals_trend)

    # Assign colors based on residual terciles
    colors = [assign_tercile_color_from_residual(res, tercile_33, tercile_67) for res in residuals]

    # Create base scatter plot
    fig = go.Figure()

    # Plot all schools
    hover_text = []
    for idx, row in df_filtered.iterrows():
        text = f"<b>{row['School Name']}</b><br>"
        text += f"Network: {row['Network']}<br>"
        text += f"Gradespan: {row['Gradespan']}<br>"
        text += f"FRL: {row['FRL_Percent']:.1f}%<br>"
        text += f"Performance: {row[performance_col]:.1f}%<br>"
        text += f"CSF: {row['CSF Portfolio']}<br>"
        text += f"Type: {row['School_Type']}"
        hover_text.append(text)

    # Base scatter for all schools
    base_trace = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            opacity=0.7,
            line=dict(width=0.5, color='white')
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    )
    fig.add_trace(base_trace)

    # Highlight CSF schools if requested
    if highlight_csf:
        csf_schools = df_filtered[df_filtered['CSF Portfolio'].str.upper() == 'CSF']
        if not csf_schools.empty:
            fig.add_trace(go.Scatter(
                x=csf_schools['FRL_Percent'],
                y=csf_schools[performance_col],
                mode='markers',
                marker=dict(
                    size=12,
                    color='rgba(255, 215, 0, 0.8)',  # Gold
                    line=dict(width=2, color='orange')
                ),
                name='CSF Portfolio',
                hovertemplate='<b>CSF School</b><extra></extra>',
                showlegend=True
            ))

    # Highlight selected network
    if selected_network and selected_network != 'All':
        network_schools = df_filtered[df_filtered['Network'] == selected_network]
        if not network_schools.empty:
            fig.add_trace(go.Scatter(
                x=network_schools['FRL_Percent'],
                y=network_schools[performance_col],
                mode='markers',
                marker=dict(
                    size=12,
                    color='rgba(138, 43, 226, 0.7)',  # Purple
                    line=dict(width=2, color='purple')
                ),
                name=f'{selected_network} Network',
                hovertemplate='<b>Network School</b><extra></extra>',
                showlegend=True
            ))

    # Highlight selected school
    if selected_school and selected_school != 'None':
        school_data = df_filtered[df_filtered['School Name'] == selected_school]
        if not school_data.empty:
            fig.add_trace(go.Scatter(
                x=school_data['FRL_Percent'],
                y=school_data[performance_col],
                mode='markers+text',
                marker=dict(
                    size=18,
                    color='#FF6B35',  # Bright orange
                    line=dict(width=3, color='#FF4500')
                ),
                text=[selected_school],
                textposition='top center',
                textfont=dict(size=10, color='#FF4500', family='Arial Black'),
                name='Selected School',
                hovertemplate='<b>Selected School</b><extra></extra>',
                showlegend=True
            ))

    # Calculate and add regression line
    slope, intercept, r_squared = calculate_regression(x, y)
    if slope is not None:
        x_range = np.array([0, 100])
        y_pred = slope * x_range + intercept

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Trend (R²={r_squared:.3f})',
            showlegend=True,
            hovertemplate=f'Regression Line<br>R² = {r_squared:.3f}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>{subject}</b>',
            font=dict(size=18, color='#1f4788'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Percent FRL Students (%)',
            range=[0, 100],
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title='School Performance Value (%)',
            range=[0, 100],
            gridcolor='lightgray',
            showgrid=True
        ),
        plot_bgcolor='white',
        hovermode='closest',
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )

    return fig

def main():
    """Main application function"""

    # Header
    st.markdown('<p class="main-header">Colorado Schools CMAS - Similar Schools Performance Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">2024-25 Academic Year | English Language Arts & Mathematics</p>', unsafe_allow_html=True)

    # Load data
    with st.spinner('Loading data...'):
        df = load_data()

    if df.empty:
        st.error("Unable to load data. Please ensure the Excel files are in the same directory as this app.")
        st.info("Expected files: '2025 CMAS Performance_ELA.xlsx' and '2025 CMAS Performance_Math.xlsx'")
        return

    # Sidebar filters
    st.sidebar.header('🔍 Filters & Controls')

    # School selector
    school_options = ['None'] + sorted(df['School Name'].dropna().unique().tolist())
    selected_school = st.sidebar.selectbox(
        '🏫 Select School',
        options=school_options,
        index=0,
        help='Highlight a specific school on both plots'
    )

    # Network selector
    network_options = ['All'] + sorted(df['Network'].dropna().unique().tolist())
    selected_network = st.sidebar.selectbox(
        '🌐 Select Network',
        options=network_options,
        index=0,
        help='Highlight all schools in a specific network'
    )

    # Gradespan filter
    gradespan_options = ['All', 'Elementary', 'Middle', 'High', 'Multiple']
    selected_gradespan = st.sidebar.radio(
        '📚 Gradespan Filter',
        options=gradespan_options,
        index=0,
        help='Filter schools by grade level (updates trend line)'
    )

    # CSF Portfolio filter
    highlight_csf = st.sidebar.checkbox(
        '⭐ Highlight CSF Portfolio Schools',
        value=False,
        help='Highlight schools in the Colorado Schools Fund portfolio'
    )

    # Charter schools filter
    show_only_charter = st.sidebar.checkbox(
        '🎓 Show Only Charter Schools',
        value=False,
        help='Filter to display only charter schools (trend line remains based on all schools in selected gradespan)'
    )

    # Debug: Show available gradespan categories
    st.sidebar.markdown('---')
    st.sidebar.markdown('**📊 Data Info**')
    st.sidebar.text(f'Total schools: {len(df)}')
    if 'Gradespan_Category' in df.columns:
        gradespan_counts = df['Gradespan_Category'].value_counts()
        for category, count in gradespan_counts.items():
            st.sidebar.text(f'{category}: {count}')

    # Apply filters in stages
    # Stage 1: Apply gradespan filter - this dataset will be used for trendline
    df_for_trendline = df.copy()
    if selected_gradespan != 'All':
        df_for_trendline = df_for_trendline[df_for_trendline['Gradespan_Category'] == selected_gradespan]

    # Stage 2: Apply charter filter to get final display dataset
    df_filtered = df_for_trendline.copy()
    if show_only_charter:
        df_filtered = df_filtered[df_filtered['School_Type'].str.upper().str.contains('CHARTER', na=False)]

    # Debug: Show filtered count
    st.sidebar.text(f'Displaying: {len(df_filtered)} schools')

    # Create two-column layout for plots
    col1, col2 = st.columns(2)

    with col1:
        ela_fig = create_scatter_plot(
            df_filtered, df_for_trendline, 'ELA', selected_school, selected_network,
            highlight_csf
        )
        st.plotly_chart(ela_fig, use_container_width=True)

    with col2:
        math_fig = create_scatter_plot(
            df_filtered, df_for_trendline, 'Math', selected_school, selected_network,
            highlight_csf
        )
        st.plotly_chart(math_fig, use_container_width=True)

    # Legend explanation
    st.markdown('---')
    with st.expander('📖 Color Coding Legend', expanded=False):
        st.markdown("""
        **Performance Terciles (Relative to Trend Line):**
        - 🔵 **Dark Blue**: Top third - Schools performing above expectations for their FRL%
        - 🔷 **Teal**: Middle third - Schools performing near the trend line
        - ⚪ **Light Gray**: Bottom third - Schools performing below expectations for their FRL%

        **Note:** Terciles are calculated based on each school's distance from the trend line, not absolute performance. This helps identify schools that outperform or underperform relative to similar demographics.

        **Highlights:**
        - 🟠 **Orange**: Selected school
        - 🟡 **Gold**: CSF Portfolio schools (when enabled)
        - 🟣 **Purple**: Selected network schools

        **Trend Line:**
        - 🔴 **Red Dashed Line**: Best-fit regression line with R² value
        - Updates when gradespan filter changes
        - Remains constant when charter filter is applied (shows performance vs. all schools)
        """)

    # Data export
    st.markdown('---')
    col_export1, col_export2 = st.columns(2)

    with col_export1:
        # Download filtered data
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label='📥 Download Filtered Data (CSV)',
            data=csv,
            file_name='cmas_filtered_data.csv',
            mime='text/csv'
        )

    with col_export2:
        # Download all data
        csv_all = df.to_csv(index=False)
        st.download_button(
            label='📥 Download All Data (CSV)',
            data=csv_all,
            file_name='cmas_all_data.csv',
            mime='text/csv'
        )

    # Footer
    st.markdown('---')
    st.markdown(
        '<p style="text-align: center; color: gray; font-size: 0.9rem;">'
        'Colorado CMAS Performance Dashboard | Data Source: 2024-25 CMAS Assessments'
        '</p>',
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
