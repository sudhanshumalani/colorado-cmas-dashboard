"""
Colorado Schools CMAS - Similar Schools Performance Dashboard
Enhanced Interactive Version with AI, Comparisons, Peer Groups, and Advanced Analytics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import warnings
import json
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
    .insight-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f4788;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    .comparison-table {
        font-size: 0.9rem;
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

        # Select columns for merge
        ela_columns = ['School Name', 'Network', 'FRL_Percent', 'ELA_Performance', 'Gradespan']

        if 'CSF Portfolio' in ela_df.columns:
            ela_columns.append('CSF Portfolio')
        else:
            ela_df['CSF Portfolio'] = 'Unknown'
            ela_columns.append('CSF Portfolio')

        if 'School_Type' in ela_df.columns:
            ela_columns.append('School_Type')
        elif 'Charter Y/N' in ela_df.columns:
            ela_df['School_Type'] = ela_df['Charter Y/N'].apply(lambda x: 'Charter' if str(x).upper() == 'Y' else 'District')
            ela_columns.append('School_Type')
        else:
            ela_df['School_Type'] = 'Unknown'
            ela_columns.append('School_Type')

        # Merge ELA and Math data
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

        # Convert FRL from decimal to percentage if needed
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

    # Check for simple format
    if 'ELEMENTARY' in gradespan_str:
        return 'Elementary'
    elif 'MIDDLE' in gradespan_str:
        return 'Middle'
    elif 'HIGH' in gradespan_str:
        return 'High'

    # Fallback to grade ranges
    has_elementary = any(g in gradespan_str for g in ['K', '1', '2', '3', '4', '5'])
    has_middle = any(g in gradespan_str for g in ['6', '7', '8'])
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

    tercile_33 = np.percentile(valid_residuals, 33.33)
    tercile_67 = np.percentile(valid_residuals, 66.67)

    return tercile_33, tercile_67

def assign_tercile_color_from_residual(residual, tercile_33, tercile_67):
    """Assign color based on residual tercile"""
    if pd.isna(residual):
        return '#D3D3D3'
    elif residual >= tercile_67:
        return '#1f4788'  # Dark blue - top third
    elif residual >= tercile_33:
        return '#4cb5c4'  # Teal - middle third
    else:
        return '#D3D3D3'  # Light gray - bottom third

def calculate_regression(x, y):
    """Calculate linear regression"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return None, None, None

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    r_squared = r_value ** 2

    return slope, intercept, r_squared

def find_peer_schools(df, school_name, frl_tolerance=10, n_peers=10):
    """Find peer schools based on FRL% and gradespan"""
    school = df[df['School Name'] == school_name]
    if school.empty:
        return pd.DataFrame()

    school_frl = school.iloc[0]['FRL_Percent']
    school_gradespan = school.iloc[0]['Gradespan_Category']

    # Find schools with similar FRL (±tolerance) and same gradespan
    peers = df[
        (df['School Name'] != school_name) &
        (df['FRL_Percent'] >= school_frl - frl_tolerance) &
        (df['FRL_Percent'] <= school_frl + frl_tolerance) &
        (df['Gradespan_Category'] == school_gradespan)
    ].copy()

    # Calculate distance from target school's FRL
    peers['FRL_Distance'] = abs(peers['FRL_Percent'] - school_frl)
    peers = peers.sort_values('FRL_Distance').head(n_peers)

    return peers

def identify_outliers(df, subject='ELA', std_threshold=1.5):
    """Identify schools that are outliers (significantly above/below trend)"""
    performance_col = f'{subject}_Performance'

    x = df['FRL_Percent'].values
    y = df[performance_col].values

    slope, intercept, r_squared = calculate_regression(x, y)

    if slope is None:
        return pd.DataFrame(), pd.DataFrame()

    y_predicted = slope * x + intercept
    residuals = y - y_predicted

    std_residual = np.nanstd(residuals)

    # High performers: residual > threshold * std
    high_performers = df[residuals > std_threshold * std_residual].copy()
    high_performers['Residual'] = residuals[residuals > std_threshold * std_residual]
    high_performers = high_performers.sort_values('Residual', ascending=False)

    # Low performers: residual < -threshold * std
    low_performers = df[residuals < -std_threshold * std_residual].copy()
    low_performers['Residual'] = residuals[residuals < -std_threshold * std_residual]
    low_performers = low_performers.sort_values('Residual')

    return high_performers, low_performers

def calculate_network_stats(df, network_name):
    """Calculate aggregate statistics for a network"""
    network_schools = df[df['Network'] == network_name]

    if network_schools.empty:
        return {}

    stats_dict = {
        'total_schools': len(network_schools),
        'avg_frl': network_schools['FRL_Percent'].mean(),
        'avg_ela': network_schools['ELA_Performance'].mean(),
        'avg_math': network_schools['Math_Performance'].mean(),
        'gradespan_dist': network_schools['Gradespan_Category'].value_counts().to_dict()
    }

    return stats_dict

def create_scatter_plot(df_filtered, df_for_trendline, subject, selected_school, selected_network,
                       highlight_csf, comparison_schools=None, peer_schools=None):
    """Create enhanced scatter plot with all features"""

    performance_col = f'{subject}_Performance'

    x = df_filtered['FRL_Percent'].values
    y = df_filtered[performance_col].values

    # Calculate regression
    x_trend = df_for_trendline['FRL_Percent'].values
    y_trend = df_for_trendline[performance_col].values
    slope, intercept, r_squared = calculate_regression(x_trend, y_trend)

    # Calculate residuals
    residuals = np.full(len(y), np.nan)
    if slope is not None and intercept is not None:
        y_predicted = slope * x + intercept
        residuals = y - y_predicted

    residuals_trend = np.full(len(y_trend), np.nan)
    if slope is not None and intercept is not None:
        y_predicted_trend = slope * x_trend + intercept
        residuals_trend = y_trend - y_predicted_trend

    tercile_33, tercile_67 = calculate_terciles_from_residuals(residuals_trend)

    colors = [assign_tercile_color_from_residual(res, tercile_33, tercile_67) for res in residuals]

    fig = go.Figure()

    # Hover text
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

    # Base scatter
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
        showlegend=False,
        name='Schools'
    )
    fig.add_trace(base_trace)

    # Peer schools highlight
    if peer_schools is not None and not peer_schools.empty:
        fig.add_trace(go.Scatter(
            x=peer_schools['FRL_Percent'],
            y=peer_schools[performance_col],
            mode='markers',
            marker=dict(
                size=14,
                color='rgba(255, 165, 0, 0.6)',
                line=dict(width=3, color='orange')
            ),
            name='Peer Schools',
            showlegend=True
        ))

    # Comparison schools highlight
    if comparison_schools is not None and len(comparison_schools) > 0:
        comp_df = df_filtered[df_filtered['School Name'].isin(comparison_schools)]
        if not comp_df.empty:
            fig.add_trace(go.Scatter(
                x=comp_df['FRL_Percent'],
                y=comp_df[performance_col],
                mode='markers+text',
                marker=dict(
                    size=16,
                    color='rgba(255, 0, 255, 0.7)',
                    line=dict(width=3, color='purple')
                ),
                text=comp_df['School Name'],
                textposition='top center',
                textfont=dict(size=9, color='purple'),
                name='Comparison',
                showlegend=True
            ))

    # CSF schools
    if highlight_csf:
        csf_schools = df_filtered[df_filtered['CSF Portfolio'].str.upper() == 'CSF']
        if not csf_schools.empty:
            fig.add_trace(go.Scatter(
                x=csf_schools['FRL_Percent'],
                y=csf_schools[performance_col],
                mode='markers',
                marker=dict(
                    size=12,
                    color='rgba(255, 215, 0, 0.8)',
                    line=dict(width=2, color='orange')
                ),
                name='CSF Portfolio',
                showlegend=True
            ))

    # Network schools
    if selected_network and selected_network != 'All':
        network_schools = df_filtered[df_filtered['Network'] == selected_network]
        if not network_schools.empty:
            fig.add_trace(go.Scatter(
                x=network_schools['FRL_Percent'],
                y=network_schools[performance_col],
                mode='markers',
                marker=dict(
                    size=12,
                    color='rgba(138, 43, 226, 0.7)',
                    line=dict(width=2, color='purple')
                ),
                name=f'{selected_network}',
                showlegend=True
            ))

    # Selected school
    if selected_school and selected_school != 'None':
        school_data = df_filtered[df_filtered['School Name'] == selected_school]
        if not school_data.empty:
            fig.add_trace(go.Scatter(
                x=school_data['FRL_Percent'],
                y=school_data[performance_col],
                mode='markers+text',
                marker=dict(
                    size=18,
                    color='#FF6B35',
                    line=dict(width=3, color='#FF4500')
                ),
                text=[selected_school],
                textposition='top center',
                textfont=dict(size=10, color='#FF4500', family='Arial Black'),
                name='Selected School',
                showlegend=True
            ))

    # Regression line
    if slope is not None:
        x_range = np.array([0, 100])
        y_pred = slope * x_range + intercept

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Trend (R²={r_squared:.3f})',
            showlegend=True
        ))

    # Layout
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
    """Main application"""

    # Header
    st.markdown('<p class="main-header">Colorado Schools CMAS - Similar Schools Performance Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">2024-25 Academic Year | Enhanced Interactive Version</p>', unsafe_allow_html=True)

    # Load data
    with st.spinner('Loading data...'):
        df = load_data()

    if df.empty:
        st.error("Unable to load data. Please ensure the Excel files are in the same directory as this app.")
        return

    # Sidebar
    st.sidebar.header('🔍 Filters & Tools')

    # Create tabs for different features
    tab_main, tab_compare, tab_peers, tab_outliers, tab_networks, tab_ai = st.tabs([
        "📊 Main Dashboard",
        "🔄 Compare Schools",
        "👥 Peer Finder",
        "⭐ Outliers",
        "🏢 Network Reports",
        "🤖 AI Assistant"
    ])

    # Sidebar filters (apply to all tabs)
    selected_school = st.sidebar.selectbox(
        '🏫 Select School',
        options=['None'] + sorted(df['School Name'].dropna().unique().tolist()),
        index=0
    )

    selected_network = st.sidebar.selectbox(
        '🌐 Select Network',
        options=['All'] + sorted(df['Network'].dropna().unique().tolist()),
        index=0
    )

    gradespan_options = ['All', 'Elementary', 'Middle', 'High', 'Multiple']
    selected_gradespan = st.sidebar.radio(
        '📚 Gradespan Filter',
        options=gradespan_options,
        index=0
    )

    highlight_csf = st.sidebar.checkbox('⭐ Highlight CSF Portfolio Schools', value=False)
    show_only_charter = st.sidebar.checkbox('🎓 Show Only Charter Schools', value=False)

    # Apply filters
    df_for_trendline = df.copy()
    if selected_gradespan != 'All':
        df_for_trendline = df_for_trendline[df_for_trendline['Gradespan_Category'] == selected_gradespan]

    df_filtered = df_for_trendline.copy()
    if show_only_charter:
        df_filtered = df_filtered[df_filtered['School_Type'].str.upper().str.contains('CHARTER', na=False)]

    # TAB 1: MAIN DASHBOARD
    with tab_main:
        st.markdown("### 📊 Performance Overview")

        col1, col2 = st.columns(2)

        with col1:
            ela_fig = create_scatter_plot(
                df_filtered, df_for_trendline, 'ELA', selected_school, selected_network, highlight_csf
            )
            st.plotly_chart(ela_fig, use_container_width=True)

        with col2:
            math_fig = create_scatter_plot(
                df_filtered, df_for_trendline, 'Math', selected_school, selected_network, highlight_csf
            )
            st.plotly_chart(math_fig, use_container_width=True)

        # Quick Stats
        st.markdown("---")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Schools Shown", len(df_filtered))
        with col_s2:
            st.metric("Avg FRL%", f"{df_filtered['FRL_Percent'].mean():.1f}%")
        with col_s3:
            st.metric("Avg ELA", f"{df_filtered['ELA_Performance'].mean():.1f}%")
        with col_s4:
            st.metric("Avg Math", f"{df_filtered['Math_Performance'].mean():.1f}%")

    # TAB 2: COMPARE SCHOOLS
    with tab_compare:
        st.markdown("### 🔄 School Comparison Tool")
        st.markdown("Select 2-5 schools to compare side-by-side")

        comparison_schools = st.multiselect(
            "Select schools to compare:",
            options=sorted(df['School Name'].dropna().unique().tolist()),
            max_selections=5
        )

        if len(comparison_schools) >= 2:
            comp_df = df[df['School Name'].isin(comparison_schools)].copy()

            # Comparison plots
            col1, col2 = st.columns(2)

            with col1:
                ela_comp_fig = create_scatter_plot(
                    df_filtered, df_for_trendline, 'ELA', None, None, False,
                    comparison_schools=comparison_schools
                )
                st.plotly_chart(ela_comp_fig, use_container_width=True)

            with col2:
                math_comp_fig = create_scatter_plot(
                    df_filtered, df_for_trendline, 'Math', None, None, False,
                    comparison_schools=comparison_schools
                )
                st.plotly_chart(math_comp_fig, use_container_width=True)

            # Comparison table
            st.markdown("#### 📋 Detailed Comparison")
            comp_display = comp_df[['School Name', 'Network', 'Gradespan', 'FRL_Percent',
                                    'ELA_Performance', 'Math_Performance', 'School_Type', 'CSF Portfolio']]
            comp_display.columns = ['School', 'Network', 'Gradespan', 'FRL%', 'ELA%', 'Math%', 'Type', 'CSF']
            st.dataframe(comp_display, use_container_width=True)

            # Calculate differences
            st.markdown("#### 📊 Performance Analysis")
            avg_ela = comp_df['ELA_Performance'].mean()
            avg_math = comp_df['Math_Performance'].mean()
            best_ela = comp_df.loc[comp_df['ELA_Performance'].idxmax(), 'School Name']
            best_math = comp_df.loc[comp_df['Math_Performance'].idxmax(), 'School Name']

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Best ELA Performance:** {best_ela}")
            with col2:
                st.info(f"**Best Math Performance:** {best_math}")

        else:
            st.info("👆 Select at least 2 schools to start comparing")

    # TAB 3: PEER FINDER
    with tab_peers:
        st.markdown("### 👥 Find Similar Schools (Peer Groups)")
        st.markdown("Automatically find schools with similar demographics and grade levels")

        peer_school = st.selectbox(
            "Select a school to find its peers:",
            options=sorted(df['School Name'].dropna().unique().tolist()),
            key='peer_selector'
        )

        frl_tolerance = st.slider("FRL% tolerance (±)", 5, 20, 10)
        n_peers = st.slider("Number of peers to show", 5, 20, 10)

        if peer_school:
            peers = find_peer_schools(df, peer_school, frl_tolerance, n_peers)

            if not peers.empty:
                # Get the selected school data
                school_data = df[df['School Name'] == peer_school].iloc[0]

                st.markdown(f"#### 🎯 Peers for: **{peer_school}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("School FRL%", f"{school_data['FRL_Percent']:.1f}%")
                with col2:
                    st.metric("School ELA%", f"{school_data['ELA_Performance']:.1f}%")
                with col3:
                    st.metric("School Math%", f"{school_data['Math_Performance']:.1f}%")

                # Scatter plots with peers highlighted
                col1, col2 = st.columns(2)

                with col1:
                    ela_peer_fig = create_scatter_plot(
                        df_filtered, df_for_trendline, 'ELA', peer_school, None, False,
                        peer_schools=peers
                    )
                    st.plotly_chart(ela_peer_fig, use_container_width=True)

                with col2:
                    math_peer_fig = create_scatter_plot(
                        df_filtered, df_for_trendline, 'Math', peer_school, None, False,
                        peer_schools=peers
                    )
                    st.plotly_chart(math_peer_fig, use_container_width=True)

                # Peer comparison table
                st.markdown("#### 📋 Peer Schools List")
                peer_display = peers[['School Name', 'Network', 'FRL_Percent',
                                     'ELA_Performance', 'Math_Performance', 'School_Type']]
                peer_display.columns = ['School', 'Network', 'FRL%', 'ELA%', 'Math%', 'Type']
                st.dataframe(peer_display, use_container_width=True)

                # Insights
                st.markdown("#### 💡 Peer Group Insights")
                avg_peer_ela = peers['ELA_Performance'].mean()
                avg_peer_math = peers['Math_Performance'].mean()

                ela_diff = school_data['ELA_Performance'] - avg_peer_ela
                math_diff = school_data['Math_Performance'] - avg_peer_math

                col1, col2 = st.columns(2)
                with col1:
                    if ela_diff > 0:
                        st.success(f"📈 ELA is **{ela_diff:.1f} points ABOVE** peer average ({avg_peer_ela:.1f}%)")
                    else:
                        st.warning(f"📉 ELA is **{abs(ela_diff):.1f} points BELOW** peer average ({avg_peer_ela:.1f}%)")

                with col2:
                    if math_diff > 0:
                        st.success(f"📈 Math is **{math_diff:.1f} points ABOVE** peer average ({avg_peer_math:.1f}%)")
                    else:
                        st.warning(f"📉 Math is **{abs(math_diff):.1f} points BELOW** peer average ({avg_peer_math:.1f}%)")

            else:
                st.warning("No peer schools found with the selected criteria. Try increasing the FRL tolerance.")

    # TAB 4: OUTLIERS
    with tab_outliers:
        st.markdown("### ⭐ Outlier Analysis - Schools Beating/Missing Expectations")

        subject_outlier = st.radio("Select subject:", ['ELA', 'Math'], horizontal=True)
        std_threshold = st.slider("Sensitivity (standard deviations)", 1.0, 3.0, 1.5, 0.5)

        high_performers, low_performers = identify_outliers(df_filtered, subject_outlier, std_threshold)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### 🌟 Top Performers (n={len(high_performers)})")
            st.markdown("*Schools significantly ABOVE the trend line for their demographics*")

            if not high_performers.empty:
                top_display = high_performers[['School Name', 'Network', 'FRL_Percent',
                                               f'{subject_outlier}_Performance', 'Residual']].head(15)
                top_display.columns = ['School', 'Network', 'FRL%', f'{subject_outlier}%', 'Above Trend By']
                top_display['Above Trend By'] = top_display['Above Trend By'].round(1)
                st.dataframe(top_display, use_container_width=True, hide_index=True)

                # Download option
                csv_high = high_performers.to_csv(index=False)
                st.download_button(
                    "📥 Download Top Performers",
                    csv_high,
                    f"top_performers_{subject_outlier}.csv",
                    "text/csv"
                )
            else:
                st.info("No significant outperformers found with current threshold")

        with col2:
            st.markdown(f"#### ⚠️ Schools Needing Support (n={len(low_performers)})")
            st.markdown("*Schools significantly BELOW the trend line for their demographics*")

            if not low_performers.empty:
                low_display = low_performers[['School Name', 'Network', 'FRL_Percent',
                                              f'{subject_outlier}_Performance', 'Residual']].head(15)
                low_display.columns = ['School', 'Network', 'FRL%', f'{subject_outlier}%', 'Below Trend By']
                low_display['Below Trend By'] = low_display['Below Trend By'].round(1)
                st.dataframe(low_display, use_container_width=True, hide_index=True)

                # Download option
                csv_low = low_performers.to_csv(index=False)
                st.download_button(
                    "📥 Download Schools Needing Support",
                    csv_low,
                    f"schools_needing_support_{subject_outlier}.csv",
                    "text/csv"
                )
            else:
                st.info("No significant underperformers found with current threshold")

        # High-FRL Success Stories
        st.markdown("---")
        st.markdown("#### 🎯 High-Poverty Success Stories")
        st.markdown("*Schools with FRL > 70% performing above expectations*")

        success_stories = high_performers[high_performers['FRL_Percent'] > 70].head(10)
        if not success_stories.empty:
            success_display = success_stories[['School Name', 'Network', 'FRL_Percent',
                                               f'{subject_outlier}_Performance', 'School_Type']]
            success_display.columns = ['School', 'Network', 'FRL%', f'{subject_outlier}%', 'Type']
            st.dataframe(success_display, use_container_width=True, hide_index=True)
        else:
            st.info("No high-FRL success stories found")

    # TAB 5: NETWORK REPORTS
    with tab_networks:
        st.markdown("### 🏢 Network/CMO Performance Reports")

        networks = sorted(df['Network'].dropna().unique().tolist())
        selected_network_report = st.selectbox("Select Network for Detailed Report:", networks)

        if selected_network_report:
            network_data = df[df['Network'] == selected_network_report].copy()

            st.markdown(f"## {selected_network_report}")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Schools", len(network_data))
            with col2:
                st.metric("Avg FRL%", f"{network_data['FRL_Percent'].mean():.1f}%")
            with col3:
                st.metric("Avg ELA%", f"{network_data['ELA_Performance'].mean():.1f}%")
            with col4:
                st.metric("Avg Math%", f"{network_data['Math_Performance'].mean():.1f}%")

            # Gradespan distribution
            st.markdown("#### 📚 Schools by Grade Level")
            gradespan_dist = network_data['Gradespan_Category'].value_counts()
            col1, col2 = st.columns([1, 2])
            with col1:
                for gs, count in gradespan_dist.items():
                    st.write(f"**{gs}:** {count} schools")
            with col2:
                import plotly.express as px
                fig_pie = px.pie(values=gradespan_dist.values, names=gradespan_dist.index,
                                title="Gradespan Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)

            # Performance distribution
            st.markdown("#### 📊 Performance Distribution")

            # Calculate terciles for network
            network_ela_terciles = network_data['ELA_Performance'].quantile([0.33, 0.67]).values
            network_math_terciles = network_data['Math_Performance'].quantile([0.33, 0.67]).values

            ela_top = len(network_data[network_data['ELA_Performance'] >= network_ela_terciles[1]])
            ela_mid = len(network_data[(network_data['ELA_Performance'] >= network_ela_terciles[0]) &
                                       (network_data['ELA_Performance'] < network_ela_terciles[1])])
            ela_bot = len(network_data[network_data['ELA_Performance'] < network_ela_terciles[0]])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**ELA Performance Tiers:**")
                st.write(f"🔵 Top Third: {ela_top} schools ({ela_top/len(network_data)*100:.0f}%)")
                st.write(f"🔷 Middle Third: {ela_mid} schools ({ela_mid/len(network_data)*100:.0f}%)")
                st.write(f"⚪ Bottom Third: {ela_bot} schools ({ela_bot/len(network_data)*100:.0f}%)")

            with col2:
                math_top = len(network_data[network_data['Math_Performance'] >= network_math_terciles[1]])
                math_mid = len(network_data[(network_data['Math_Performance'] >= network_math_terciles[0]) &
                                           (network_data['Math_Performance'] < network_math_terciles[1])])
                math_bot = len(network_data[network_data['Math_Performance'] < network_math_terciles[0]])

                st.markdown("**Math Performance Tiers:**")
                st.write(f"🔵 Top Third: {math_top} schools ({math_top/len(network_data)*100:.0f}%)")
                st.write(f"🔷 Middle Third: {math_mid} schools ({math_mid/len(network_data)*100:.0f}%)")
                st.write(f"⚪ Bottom Third: {math_bot} schools ({math_bot/len(network_data)*100:.0f}%)")

            # School list
            st.markdown("#### 📋 All Network Schools")
            network_display = network_data[['School Name', 'Gradespan', 'FRL_Percent',
                                           'ELA_Performance', 'Math_Performance', 'School_Type']]
            network_display.columns = ['School', 'Gradespan', 'FRL%', 'ELA%', 'Math%', 'Type']
            network_display = network_display.sort_values('ELA%', ascending=False)
            st.dataframe(network_display, use_container_width=True, hide_index=True)

            # Download network report
            csv_network = network_data.to_csv(index=False)
            st.download_button(
                "📥 Download Complete Network Report",
                csv_network,
                f"{selected_network_report}_report.csv",
                "text/csv"
            )

    # TAB 6: AI ASSISTANT
    with tab_ai:
        st.markdown("### 🤖 AI-Powered Data Assistant")
        st.markdown("Ask questions about the data in natural language")

        # Check if API key is configured
        api_key_configured = False
        api_key = None

        try:
            if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
                api_key = st.secrets['ANTHROPIC_API_KEY']
                api_key_configured = True
            elif hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets['OPENAI_API_KEY']
                api_key_configured = True
        except Exception:
            api_key_configured = False

        if not api_key_configured:
            st.info("🔑 **Setup Required**: To enable the AI Assistant, you need to configure an API key.")

            with st.expander("📖 How to Enable AI Assistant"):
                st.markdown("""
                ### Setup Instructions:

                1. **Get an API Key:**
                   - **Option A (Recommended):** Anthropic Claude API
                     - Sign up at: https://console.anthropic.com/
                     - Create an API key
                     - Cost: ~$0.01-0.03 per conversation

                   - **Option B:** OpenAI GPT-4
                     - Sign up at: https://platform.openai.com/
                     - Create an API key
                     - Cost: ~$0.02-0.05 per conversation

                2. **Add API Key to Streamlit:**
                   - Go to your Streamlit Cloud dashboard
                   - Click on your app → Settings → Secrets
                   - Add: `ANTHROPIC_API_KEY = "your-key-here"`
                   - Or: `OPENAI_API_KEY = "your-key-here"`

                3. **The AI Assistant will then be able to:**
                   - Answer questions about school performance
                   - Find schools matching specific criteria
                   - Explain trends and patterns
                   - Provide data-driven recommendations
                   - Compare schools intelligently

                ### Example Questions:
                - "Which charter elementary schools are outperforming expectations?"
                - "Show me schools similar to [School Name]"
                - "What's the correlation between FRL and Math performance?"
                - "Find high-FRL schools in the top third"
                - "Compare KIPP schools to district schools"
                """)

            # Demo mode
            st.markdown("---")
            st.markdown("#### 💬 Try Demo Mode (Simulated Responses)")

            demo_question = st.text_input("Ask a question about the data:", key='demo_ai')

            if demo_question:
                with st.spinner("Analyzing data..."):
                    # Simulate AI response with actual data insights
                    st.markdown("**🤖 AI Response:**")

                    if 'charter' in demo_question.lower() and 'elementary' in demo_question.lower():
                        charter_elem = df[(df['School_Type'].str.contains('Charter', na=False)) &
                                         (df['Gradespan_Category'] == 'Elementary')]
                        st.write(f"I found **{len(charter_elem)}** charter elementary schools in the dataset.")
                        st.write(f"Average ELA Performance: **{charter_elem['ELA_Performance'].mean():.1f}%**")
                        st.write(f"Average Math Performance: **{charter_elem['Math_Performance'].mean():.1f}%**")
                        st.write(f"Average FRL: **{charter_elem['FRL_Percent'].mean():.1f}%**")

                        # Show top performers
                        top_charter = charter_elem.nlargest(5, 'ELA_Performance')[['School Name', 'Network', 'ELA_Performance']]
                        st.write("\n**Top 5 Charter Elementary Schools (ELA):**")
                        st.dataframe(top_charter, hide_index=True)

                    elif 'correlation' in demo_question.lower():
                        corr_ela = df['FRL_Percent'].corr(df['ELA_Performance'])
                        corr_math = df['FRL_Percent'].corr(df['Math_Performance'])

                        st.write(f"**Correlation Analysis:**")
                        st.write(f"- FRL vs ELA: **{corr_ela:.3f}** (negative correlation)")
                        st.write(f"- FRL vs Math: **{corr_math:.3f}** (negative correlation)")
                        st.write("\nThis means schools with higher FRL percentages tend to have lower performance scores, which is a well-documented achievement gap pattern.")

                    elif 'high' in demo_question.lower() and 'frl' in demo_question.lower():
                        high_frl = df[df['FRL_Percent'] > 70]
                        st.write(f"Found **{len(high_frl)}** schools with FRL > 70%")
                        st.write(f"Average ELA: **{high_frl['ELA_Performance'].mean():.1f}%**")
                        st.write(f"Average Math: **{high_frl['Math_Performance'].mean():.1f}%**")

                        # Best high-FRL schools
                        best_high_frl = high_frl.nlargest(5, 'ELA_Performance')[['School Name', 'FRL_Percent', 'ELA_Performance']]
                        st.write("\n**Top 5 High-FRL Schools (ELA):**")
                        st.dataframe(best_high_frl, hide_index=True)

                    else:
                        st.write("🤖 This is a demo response. Enable the full AI assistant to get intelligent, context-aware answers to any question about your school data.")
                        st.write(f"\n**Quick Stats for Your Question:**")
                        st.write(f"- Total schools: {len(df)}")
                        st.write(f"- Average ELA: {df['ELA_Performance'].mean():.1f}%")
                        st.write(f"- Average Math: {df['Math_Performance'].mean():.1f}%")

                st.info("💡 **Tip:** Enable the full AI assistant with an API key for much more sophisticated analysis and natural language understanding!")

        else:
            # Full AI implementation
            st.success("✅ AI Assistant is configured and ready!")

            # Initialize chat history in session state
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Display chat history
            if st.session_state.chat_history:
                st.markdown("#### 💬 Conversation History")
                for i, (q, a) in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q{i+1}: {q[:50]}...", expanded=(i == len(st.session_state.chat_history)-1)):
                        st.markdown(f"**You:** {q}")
                        st.markdown(f"**AI:** {a}")

            # New question input
            user_question = st.text_area("Ask your question about the school data:", height=100, key='ai_question')

            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("🚀 Ask AI", type="primary")
            with col2:
                if st.button("🗑️ Clear History"):
                    st.session_state.chat_history = []
                    st.rerun()

            if ask_button and user_question:
                with st.spinner("🤖 AI is analyzing your data..."):
                    try:
                        # Prepare data context
                        context = f"""You are a helpful education data analyst assistant analyzing Colorado CMAS school performance data.

Current Dataset Overview:
- Total schools: {len(df)}
- Total displayed (after filters): {len(df_filtered)}
- Current filters: Gradespan={selected_gradespan}, Charter Only={show_only_charter}

Key Statistics:
- Average FRL%: {df['FRL_Percent'].mean():.1f}%
- Average ELA Performance: {df['ELA_Performance'].mean():.1f}%
- Average Math Performance: {df['Math_Performance'].mean():.1f}%
- FRL vs ELA Correlation: {df['FRL_Percent'].corr(df['ELA_Performance']):.3f}
- FRL vs Math Correlation: {df['FRL_Percent'].corr(df['Math_Performance']):.3f}

Gradespan Distribution:
{df['Gradespan_Category'].value_counts().to_dict()}

Sample of schools (top 5 by ELA performance):
{df.nlargest(5, 'ELA_Performance')[['School Name', 'Network', 'FRL_Percent', 'ELA_Performance', 'Math_Performance', 'Gradespan_Category']].to_string()}

The data shows performance terciles based on residuals from a trend line (schools performing above/below expectations for their demographics).

Answer the user's question with specific data insights, school names, and statistics when relevant. Be concise but informative.
"""

                        # Call Claude API
                        if 'ANTHROPIC_API_KEY' in st.secrets:
                            try:
                                import anthropic
                                client = anthropic.Anthropic(api_key=api_key)

                                message = client.messages.create(
                                    model="claude-3-5-sonnet-20241022",
                                    max_tokens=1500,
                                    messages=[
                                        {"role": "user", "content": f"{context}\n\nUser Question: {user_question}"}
                                    ]
                                )

                                ai_response = message.content[0].text

                            except ImportError:
                                ai_response = "⚠️ The 'anthropic' library is not installed. Please add it to requirements.txt:\n\n`anthropic>=0.18.0`\n\nThen redeploy your app."
                            except Exception as e:
                                ai_response = f"⚠️ Error calling Claude API: {str(e)}\n\nPlease check your API key in Streamlit Secrets."

                        elif 'OPENAI_API_KEY' in st.secrets:
                            try:
                                import openai
                                client = openai.OpenAI(api_key=api_key)

                                response = client.chat.completions.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": context},
                                        {"role": "user", "content": user_question}
                                    ],
                                    max_tokens=1500
                                )

                                ai_response = response.choices[0].message.content

                            except ImportError:
                                ai_response = "⚠️ The 'openai' library is not installed. Please add it to requirements.txt:\n\n`openai>=1.12.0`\n\nThen redeploy your app."
                            except Exception as e:
                                ai_response = f"⚠️ Error calling OpenAI API: {str(e)}\n\nPlease check your API key in Streamlit Secrets."

                        else:
                            ai_response = "API key found but couldn't determine which service to use."

                        # Display response
                        st.markdown("---")
                        st.markdown("### 🤖 AI Response:")
                        st.markdown(ai_response)

                        # Save to history
                        st.session_state.chat_history.append((user_question, ai_response))

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("💡 Make sure you have the correct API library installed. Add to requirements.txt:\n\n`anthropic>=0.18.0` or `openai>=1.12.0`")

            # Example questions
            st.markdown("---")
            st.markdown("#### 💡 Example Questions to Try:")
            example_questions = [
                "Which charter elementary schools are performing above expectations?",
                "Find schools with FRL > 70% that are in the top performance tercile",
                "What's the correlation between FRL and Math performance for middle schools?",
                "Compare KIPP schools to Achievement First schools",
                "Which networks have the highest average ELA performance?",
                "Show me success stories: high-FRL schools beating the odds",
                "What percentage of charter schools are in the top tercile?",
                "Find schools similar to [specific school name]"
            ]

            for i, eq in enumerate(example_questions):
                if st.button(f"📝 {eq}", key=f"example_{i}"):
                    st.session_state.example_question = eq
                    st.rerun()

    # Footer with legend
    st.markdown("---")
    with st.expander('📖 Color Coding Legend & Help', expanded=False):
        st.markdown("""
        **Performance Terciles (Relative to Trend Line):**
        - 🔵 **Dark Blue**: Top third - Schools performing above expectations for their FRL%
        - 🔷 **Teal**: Middle third - Schools performing near the trend line
        - ⚪ **Light Gray**: Bottom third - Schools performing below expectations for their FRL%

        **Note:** Terciles are calculated based on each school's distance from the trend line, not absolute performance.

        **Highlights:**
        - 🟠 **Orange**: Selected school or peer groups
        - 🟡 **Gold**: CSF Portfolio schools
        - 🟣 **Purple**: Network schools or comparison schools

        **Trend Line:**
        - 🔴 **Red Dashed Line**: Best-fit regression line with R² value
        - Updates when gradespan filter changes
        - Remains constant when charter filter is applied

        **Pro Tips:**
        - Use the **Compare** tab to analyze multiple schools side-by-side
        - Use **Peer Finder** to identify schools with similar demographics
        - Check **Outliers** to find success stories and schools needing support
        - Review **Network Reports** for portfolio-level insights
        """)

    # Export options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_filtered = df_filtered.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Data (CSV)",
            csv_filtered,
            "cmas_filtered_data.csv",
            "text/csv"
        )

    with col2:
        csv_all = df.to_csv(index=False)
        st.download_button(
            "📥 Download All Data (CSV)",
            csv_all,
            "cmas_all_data.csv",
            "text/csv"
        )

    with col3:
        # Create summary report
        summary = f"""Colorado CMAS Performance Summary Report

Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
Filters Applied: Gradespan={selected_gradespan}, Charter Only={show_only_charter}

Summary Statistics:
- Total Schools: {len(df_filtered)}
- Average FRL: {df_filtered['FRL_Percent'].mean():.1f}%
- Average ELA Performance: {df_filtered['ELA_Performance'].mean():.1f}%
- Average Math Performance: {df_filtered['Math_Performance'].mean():.1f}%

Grade Level Distribution:
{df_filtered['Gradespan_Category'].value_counts().to_string()}
"""
        st.download_button(
            "📄 Download Summary Report",
            summary,
            "cmas_summary_report.txt",
            "text/plain"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: gray; font-size: 0.9rem;">'
        'Colorado CMAS Performance Dashboard - Enhanced Interactive Version | '
        'Data Source: 2024-25 CMAS Assessments'
        '</p>',
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
