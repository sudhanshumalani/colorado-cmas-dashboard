"""
Utilities Module
Helper functions for the RAG chatbot system.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import json


def format_school_list(schools: List[Dict], max_display: int = 10) -> str:
    """
    Format list of schools for display

    Args:
        schools: List of school dictionaries
        max_display: Maximum number to display

    Returns:
        Formatted string
    """
    if not schools:
        return "No schools found."

    lines = []
    for i, school in enumerate(schools[:max_display], 1):
        name = school.get('school_name', 'Unknown')
        network = school.get('network', 'N/A')
        frl = school.get('frl_percent', 0)
        ela = school.get('ela_performance', 0)
        math = school.get('math_performance', 0)

        lines.append(
            f"{i}. **{name}**\n"
            f"   - Network: {network}\n"
            f"   - FRL: {frl:.0f}%, ELA: {ela:.0f}%, Math: {math:.0f}%"
        )

    if len(schools) > max_display:
        lines.append(f"\n... and {len(schools) - max_display} more schools")

    return "\n\n".join(lines)


def create_scatter_plot(
    data: List[Dict],
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """
    Create interactive scatter plot

    Args:
        data: List of dictionaries with data
        x_col: Column name for X axis
        y_col: Column name for Y axis
        title: Plot title
        color_col: Optional column for color coding

    Returns:
        Plotly figure
    """
    if not data:
        return None

    df = pd.DataFrame(data)

    if color_col and color_col in df.columns:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            hover_data=df.columns.tolist(),
            title=title
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            hover_data=df.columns.tolist(),
            title=title
        )

    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        height=500
    )

    return fig


def create_bar_chart(
    data: List[Dict],
    x_col: str,
    y_col: str,
    title: str
) -> go.Figure:
    """Create bar chart from data"""

    if not data:
        return None

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title
    )

    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        height=400
    )

    return fig


def detect_visualization_intent(query: str, results: List[Dict]) -> Optional[str]:
    """
    Detect if query results should be visualized

    Args:
        query: User query
        results: Query results

    Returns:
        'scatter' | 'bar' | None
    """

    if not results or len(results) < 2:
        return None

    query_lower = query.lower()

    # Scatter plot indicators
    if any(word in query_lower for word in ['correlation', 'relationship', 'scatter', 'plot']):
        # Check if we have numeric columns
        df = pd.DataFrame(results)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_cols) >= 2:
            return 'scatter'

    # Bar chart indicators
    if any(word in query_lower for word in ['compare', 'top', 'ranking', 'distribution']):
        df = pd.DataFrame(results)
        if len(df) <= 20:  # Reasonable for bar chart
            return 'bar'

    return None


def export_to_csv(data: List[Dict], filename: str = "export.csv") -> str:
    """
    Export data to CSV

    Args:
        data: List of dictionaries
        filename: Output filename

    Returns:
        CSV string
    """
    if not data:
        return ""

    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def calculate_summary_stats(data: List[Dict], numeric_cols: List[str]) -> Dict:
    """
    Calculate summary statistics

    Args:
        data: List of dictionaries
        numeric_cols: Columns to summarize

    Returns:
        Dictionary of statistics
    """
    if not data:
        return {}

    df = pd.DataFrame(data)

    stats = {
        'count': len(df)
    }

    for col in numeric_cols:
        if col in df.columns:
            stats[f'{col}_mean'] = df[col].mean()
            stats[f'{col}_median'] = df[col].median()
            stats[f'{col}_min'] = df[col].min()
            stats[f'{col}_max'] = df[col].max()

    return stats


def generate_example_questions() -> List[str]:
    """Generate list of example questions for UI"""

    return [
        "Which district is Vega Collegiate Academy from?",
        "How many single site charter schools serve more than 50% FRL and are in the top third for ELA?",
        "List the top 5 networks by average ELA performance",
        "Show me schools with FRL > 80% performing in the top third",
        "What's the correlation between FRL and Math performance?",
        "Compare KIPP Colorado to DSST Public Schools",
        "Which elementary charter schools are outperforming their demographics?",
        "Find high-poverty schools (70%+ FRL) with strong Math results",
        "What percentage of charter schools are in the top third for ELA?",
        "Show me all schools in Denver County 1"
    ]


def sanitize_sql(sql: str) -> str:
    """
    Basic SQL sanitization (security check)

    Args:
        sql: SQL query string

    Returns:
        Sanitized SQL or raises ValueError
    """

    sql_upper = sql.upper().strip()

    # Only allow SELECT
    if not sql_upper.startswith('SELECT'):
        raise ValueError("Only SELECT queries are allowed")

    # Disallow dangerous keywords
    dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'EXEC', '--', ';--']

    for keyword in dangerous:
        if keyword in sql_upper:
            raise ValueError(f"Disallowed keyword: {keyword}")

    return sql


def format_conversation_for_export(messages: List[Dict]) -> str:
    """
    Format conversation history for export

    Args:
        messages: List of message dictionaries

    Returns:
        Formatted markdown string
    """

    lines = [
        "# Conversation Export",
        f"*Exported on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "---",
        ""
    ]

    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')

        if role == 'user':
            lines.append(f"## 👤 User")
            lines.append(content)
        else:
            lines.append(f"## 🤖 Assistant")
            lines.append(content)

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test utilities
    print("🧪 Testing utilities...")

    # Test format_school_list
    schools = [
        {'school_name': 'Test School 1', 'network': 'KIPP', 'frl_percent': 75, 'ela_performance': 45, 'math_performance': 40},
        {'school_name': 'Test School 2', 'network': 'DSST', 'frl_percent': 60, 'ela_performance': 55, 'math_performance': 50},
    ]

    formatted = format_school_list(schools)
    print("Formatted schools:")
    print(formatted)

    # Test example questions
    print("\n📝 Example questions:")
    for i, q in enumerate(generate_example_questions()[:3], 1):
        print(f"{i}. {q}")

    print("\n✅ Utilities working!")
