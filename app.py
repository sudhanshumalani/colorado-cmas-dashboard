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

        # Add District Name if available
        if 'District Name' in ela_df.columns:
            ela_df['District_Name'] = ela_df['District Name']
        else:
            ela_df['District_Name'] = 'Unknown'

        # Add Grade Config if available
        if 'Grade Config' in ela_df.columns:
            ela_df['Grade_Config'] = ela_df['Grade Config']
        else:
            ela_df['Grade_Config'] = 'Unknown'

        # Select columns for merge (now including more fields)
        ela_columns = ['School Name', 'Network', 'FRL_Percent', 'ELA_Performance', 'Gradespan', 'District_Name', 'Grade_Config']

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
                       highlight_csf, comparison_schools=None, peer_schools=None, tercile_filter='All Schools', frl_filter='All FRL Levels'):
    """Create enhanced scatter plot with all features including tercile filtering"""

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

    # Calculate tercile assignment for each school
    tercile_assignments = []
    for res in residuals:
        if np.isnan(res):
            tercile_assignments.append('Unknown')
        elif res >= tercile_67:
            tercile_assignments.append('Top Third')
        elif res >= tercile_33:
            tercile_assignments.append('Middle Third')
        else:
            tercile_assignments.append('Bottom Third')

    # Apply both tercile and FRL filters with AND logic
    # A school is highlighted only if it matches BOTH active filters
    frl_values = df_filtered['FRL_Percent'].values
    opacities = []

    for i, tercile in enumerate(tercile_assignments):
        should_highlight = True

        # Check tercile filter
        if tercile_filter == 'All Schools':
            # Tercile filter passes for all schools
            pass
        elif tercile_filter == 'Top Third Only' and tercile == 'Top Third':
            # Tercile filter passes
            pass
        elif tercile_filter == 'Middle Third Only' and tercile == 'Middle Third':
            # Tercile filter passes
            pass
        elif tercile_filter == 'Bottom Third Only' and tercile == 'Bottom Third':
            # Tercile filter passes
            pass
        else:
            # Tercile filter fails
            should_highlight = False

        # Check FRL filter (only if tercile passed)
        if should_highlight and frl_filter != 'All FRL Levels':
            frl = frl_values[i]
            matches_frl = False

            if frl_filter == 'High Poverty (>70% FRL)' and frl > 70:
                matches_frl = True
            elif frl_filter == 'Moderate Poverty (40-70% FRL)' and 40 <= frl <= 70:
                matches_frl = True
            elif frl_filter == 'Low Poverty (<40% FRL)' and frl < 40:
                matches_frl = True

            if not matches_frl:
                should_highlight = False

        # Set opacity based on final result
        if should_highlight:
            if tercile_filter == 'All Schools' and frl_filter == 'All FRL Levels':
                opacities.append(0.7)  # Default opacity when no filters active
            else:
                opacities.append(0.9)  # Highlighted
        else:
            opacities.append(0.35)  # Faded

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
            opacity=opacities,
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
    tab_main, tab_compare, tab_networks, tab_sourcing = st.tabs([
        "📊 Main Dashboard",
        "🔄 Compare Schools",
        "🏢 Network Reports",
        "🎯 School Sourcing"
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

    # AI CHATBOT IN SIDEBAR
    st.sidebar.markdown("---")

    # TAB 1: MAIN DASHBOARD
    with tab_main:
        st.markdown("### 📊 Performance Overview")

        # Tercile highlighting control
        st.markdown("**Highlight by Performance:**")
        tercile_filter = st.radio(
            "Select which schools to highlight:",
            options=['All Schools', 'Top Third Only', 'Middle Third Only', 'Bottom Third Only'],
            horizontal=True,
            help="Highlight specific tercile groups while fading others",
            key="tercile_filter"
        )

        # FRL highlighting control
        st.markdown("**Highlight by Poverty Level (FRL %):**")
        frl_filter = st.radio(
            "Select FRL range to highlight:",
            options=['All FRL Levels', 'High Poverty (>70% FRL)', 'Moderate Poverty (40-70% FRL)', 'Low Poverty (<40% FRL)'],
            horizontal=True,
            help="Highlight schools in specific poverty level ranges",
            key="frl_filter"
        )

        col1, col2 = st.columns(2)

        with col1:
            ela_fig = create_scatter_plot(
                df_filtered, df_for_trendline, 'ELA', selected_school, selected_network, highlight_csf,
                tercile_filter=tercile_filter, frl_filter=frl_filter
            )
            st.plotly_chart(ela_fig, use_container_width=True)

        with col2:
            math_fig = create_scatter_plot(
                df_filtered, df_for_trendline, 'Math', selected_school, selected_network, highlight_csf,
                tercile_filter=tercile_filter, frl_filter=frl_filter
            )
            st.plotly_chart(math_fig, use_container_width=True)


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

    # TAB 3: NETWORK REPORTS
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

    # TAB 5: SCHOOL SOURCING
    with tab_sourcing:
        st.markdown("### 🎯 School Sourcing Tool")
        st.info("Find schools based on FRL percentage and performance tercile criteria. Results show charter schools only.")

        # Calculate terciles and residuals for all schools
        x_all = df_filtered['FRL_Percent'].values
        y_ela_all = df_filtered['ELA_Performance'].values
        y_math_all = df_filtered['Math_Performance'].values

        slope_ela_all, intercept_ela_all, _ = calculate_regression(x_all, y_ela_all)
        slope_math_all, intercept_math_all, _ = calculate_regression(x_all, y_math_all)

        # Calculate residuals for filtering
        df_sourcing = df_filtered.copy()

        if slope_ela_all and intercept_ela_all:
            df_sourcing['ELA_Residual'] = y_ela_all - (slope_ela_all * x_all + intercept_ela_all)
            ela_tercile_67_all = df_sourcing['ELA_Residual'].quantile(0.67)
            ela_tercile_33_all = df_sourcing['ELA_Residual'].quantile(0.33)
            df_sourcing['ELA_Tercile'] = df_sourcing['ELA_Residual'].apply(
                lambda r: 'Top Third' if r >= ela_tercile_67_all else ('Middle Third' if r >= ela_tercile_33_all else 'Bottom Third')
            )

        if slope_math_all and intercept_math_all:
            df_sourcing['Math_Residual'] = y_math_all - (slope_math_all * x_all + intercept_math_all)
            math_tercile_67_all = df_sourcing['Math_Residual'].quantile(0.67)
            math_tercile_33_all = df_sourcing['Math_Residual'].quantile(0.33)
            df_sourcing['Math_Tercile'] = df_sourcing['Math_Residual'].apply(
                lambda r: 'Top Third' if r >= math_tercile_67_all else ('Middle Third' if r >= math_tercile_33_all else 'Bottom Third')
            )

        # Filter to charter schools only
        df_sourcing = df_sourcing[df_sourcing['School_Type'].str.upper().str.contains('CHARTER', na=False)].copy()

        # Filters in columns
        col1, col2 = st.columns(2)

        with col1:
            frl_threshold = st.selectbox(
                "📊 FRL Percentage Filter",
                options=[
                    "More than 90% FRL",
                    "More than 80% FRL",
                    "More than 70% FRL",
                    "More than 60% FRL",
                    "More than 50% FRL",
                    "More than 40% FRL",
                    "More than 30% FRL",
                    "More than 20% FRL",
                    "More than 10% FRL",
                    "All schools (no FRL filter)"
                ],
                index=2  # Default to >70%
            )

        with col2:
            performance_filter = st.selectbox(
                "🎯 Performance Filter",
                options=[
                    "Top Third",
                    "Middle Third",
                    "Bottom Third",
                    "Above Trendline (All schools above trendline)"
                ],
                index=0  # Default to Top Third
            )

        # Parse FRL threshold
        frl_min = 0
        if "90%" in frl_threshold:
            frl_min = 90
        elif "80%" in frl_threshold:
            frl_min = 80
        elif "70%" in frl_threshold:
            frl_min = 70
        elif "60%" in frl_threshold:
            frl_min = 60
        elif "50%" in frl_threshold:
            frl_min = 50
        elif "40%" in frl_threshold:
            frl_min = 40
        elif "30%" in frl_threshold:
            frl_min = 30
        elif "20%" in frl_threshold:
            frl_min = 20
        elif "10%" in frl_threshold:
            frl_min = 10

        # Apply FRL filter
        df_sourcing_filtered = df_sourcing[df_sourcing['FRL_Percent'] >= frl_min].copy()

        # Apply performance filter for ELA
        if performance_filter == "Above Trendline (All schools above trendline)":
            ela_results = df_sourcing_filtered[df_sourcing_filtered['ELA_Residual'] > 0].copy()
            math_results = df_sourcing_filtered[df_sourcing_filtered['Math_Residual'] > 0].copy()
        else:
            ela_results = df_sourcing_filtered[df_sourcing_filtered['ELA_Tercile'] == performance_filter].copy()
            math_results = df_sourcing_filtered[df_sourcing_filtered['Math_Tercile'] == performance_filter].copy()

        # Sort by performance (descending)
        ela_results = ela_results.sort_values('ELA_Performance', ascending=False)
        math_results = math_results.sort_values('Math_Performance', ascending=False)

        # Display results
        st.markdown("---")
        st.markdown(f"### 📋 Results: {frl_threshold} + {performance_filter}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### 📖 ELA Results ({len(ela_results)} schools)")

            if len(ela_results) > 0:
                ela_display = ela_results[['School Name', 'Network', 'FRL_Percent', 'ELA_Performance',
                                           'Gradespan_Category', 'ELA_Tercile']].copy()
                ela_display.columns = ['School', 'Network', 'FRL%', 'ELA%', 'Gradespan', 'Tercile']
                st.dataframe(ela_display, use_container_width=True, hide_index=True, height=400)

                # Download button for ELA
                csv_ela = ela_display.to_csv(index=False)
                st.download_button(
                    "📥 Download ELA List",
                    csv_ela,
                    f"ELA_sourcing_{frl_threshold.replace(' ', '_')}_{performance_filter.replace(' ', '_')}.csv",
                    "text/csv",
                    key="download_ela"
                )
            else:
                st.info("No schools match the selected criteria for ELA.")

        with col2:
            st.markdown(f"#### 🔢 Math Results ({len(math_results)} schools)")

            if len(math_results) > 0:
                math_display = math_results[['School Name', 'Network', 'FRL_Percent', 'Math_Performance',
                                             'Gradespan_Category', 'Math_Tercile']].copy()
                math_display.columns = ['School', 'Network', 'FRL%', 'Math%', 'Gradespan', 'Tercile']
                st.dataframe(math_display, use_container_width=True, hide_index=True, height=400)

                # Download button for Math
                csv_math = math_display.to_csv(index=False)
                st.download_button(
                    "📥 Download Math List",
                    csv_math,
                    f"Math_sourcing_{frl_threshold.replace(' ', '_')}_{performance_filter.replace(' ', '_')}.csv",
                    "text/csv",
                    key="download_math"
                )
            else:
                st.info("No schools match the selected criteria for Math.")

        # Summary statistics
        st.markdown("---")
        st.markdown("### 📊 Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ELA Schools Found", len(ela_results))
        with col2:
            st.metric("Math Schools Found", len(math_results))
        with col3:
            if len(ela_results) > 0:
                st.metric("Avg ELA Performance", f"{ela_results['ELA_Performance'].mean():.1f}%")
            else:
                st.metric("Avg ELA Performance", "N/A")
        with col4:
            if len(math_results) > 0:
                st.metric("Avg Math Performance", f"{math_results['Math_Performance'].mean():.1f}%")
            else:
                st.metric("Avg Math Performance", "N/A")

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
