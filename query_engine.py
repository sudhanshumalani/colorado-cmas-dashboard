"""
Query Engine Module
Handles SQL query generation using Claude and query execution.
"""

import duckdb
from typing import Dict, List, Tuple, Optional
import re
import anthropic


class QueryEngine:
    """Generate and execute SQL queries using Claude"""

    def __init__(self, db_path: str = "school_data.duckdb", api_key: str = None):
        """Initialize query engine"""
        self.db_path = db_path
        self.conn = None
        self.api_key = api_key
        self.claude_client = anthropic.Anthropic(api_key=api_key) if api_key else None

    def connect(self):
        """Connect to database"""
        if not self.conn:
            self.conn = duckdb.connect(self.db_path)

    def classify_query(self, user_question: str) -> Dict:
        """
        Classify user query to determine best approach

        Returns:
            {
                'type': 'factual_lookup' | 'analytical' | 'comparison' | 'insight',
                'requires_sql': bool,
                'requires_reasoning': bool,
                'confidence': float
            }
        """

        # Simple keyword-based classification (fast, no API call)
        question_lower = user_question.lower()

        # Factual lookup indicators
        factual_keywords = ['what is', 'how many', 'list', 'show me', 'which district', 'what district']

        # Analytical indicators
        analytical_keywords = ['why', 'analyze', 'explain', 'correlation', 'relationship', 'trend']

        # Comparison indicators
        comparison_keywords = ['compare', 'versus', 'vs', 'difference between', 'better than']

        # Count keyword matches
        factual_score = sum(1 for kw in factual_keywords if kw in question_lower)
        analytical_score = sum(1 for kw in analytical_keywords if kw in question_lower)
        comparison_score = sum(1 for kw in comparison_keywords if kw in question_lower)

        # Determine query type
        if factual_score > analytical_score and factual_score > comparison_score:
            query_type = 'factual_lookup'
            requires_sql = True
            requires_reasoning = False
        elif analytical_score > 0:
            query_type = 'analytical'
            requires_sql = True
            requires_reasoning = True
        elif comparison_score > 0:
            query_type = 'comparison'
            requires_sql = True
            requires_reasoning = True
        else:
            # Default to insight for open-ended questions
            query_type = 'insight'
            requires_sql = False
            requires_reasoning = True

        return {
            'type': query_type,
            'requires_sql': requires_sql,
            'requires_reasoning': requires_reasoning,
            'confidence': 0.8
        }

    def generate_sql(self, user_question: str, schema: str) -> Optional[str]:
        """
        Generate SQL query using Claude

        Args:
            user_question: Natural language question
            schema: Database schema description

        Returns:
            SQL query string or None if failed
        """

        if not self.claude_client:
            print("⚠️ Claude API not configured")
            return None

        system_prompt = f"""You are an expert SQL query generator for educational data analysis.

{schema}

RULES:
1. Generate valid DuckDB SQL queries only
2. Use proper SQL syntax and formatting
3. Return ONLY the SQL query, no explanations
4. Use appropriate JOINs, WHERE clauses, and aggregations
5. Limit results to reasonable amounts (use LIMIT when appropriate)
6. For "single site" schools, use: network = 'Single Site Charter School'
7. For tercile queries, use the tercile columns directly (ela_tercile, math_tercile)
8. Be careful with NULL values - use COALESCE or IS NOT NULL when needed

EXAMPLES:

Question: "Which district is Vega Collegiate Academy from?"
SQL: SELECT school_name, district_name FROM schools WHERE school_name LIKE '%Vega Collegiate%'

Question: "How many charter schools serve more than 70% FRL?"
SQL: SELECT COUNT(*) as school_count FROM charter_schools WHERE frl_percent > 70

Question: "List top 5 networks by average ELA performance"
SQL: SELECT network, AVG(ela_performance) as avg_ela, COUNT(*) as school_count
     FROM schools
     WHERE network IS NOT NULL
     GROUP BY network
     ORDER BY avg_ela DESC
     LIMIT 5

Question: "Single site schools with FRL > 50% in top third ELA"
SQL: SELECT school_name, frl_percent, ela_performance, ela_tercile
     FROM schools
     WHERE network = 'Single Site Charter School'
       AND frl_percent > 50
       AND ela_tercile = 'Top Third'
     ORDER BY ela_performance DESC
"""

        try:
            # Use Claude to generate SQL
            message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Generate a SQL query for this question:\n\n{user_question}"
                }]
            )

            sql = message.content[0].text.strip()

            # Clean up the SQL (remove markdown code blocks if present)
            sql = re.sub(r'```sql\s*', '', sql)
            sql = re.sub(r'```\s*', '', sql)
            sql = sql.strip()

            # Validate it looks like SQL
            if not any(keyword in sql.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                print("⚠️ Generated text doesn't look like SQL")
                return None

            # Security: Only allow SELECT queries
            if not sql.upper().strip().startswith('SELECT'):
                print("⚠️ Only SELECT queries are allowed")
                return None

            return sql

        except Exception as e:
            print(f"❌ Error generating SQL: {e}")
            return None

    def execute_query(self, sql: str) -> Tuple[bool, Optional[List], Optional[str]]:
        """
        Execute SQL query safely

        Args:
            sql: SQL query string

        Returns:
            (success: bool, results: List or None, error: str or None)
        """

        self.connect()

        try:
            # Execute query
            result = self.conn.execute(sql).fetchall()

            # Get column names
            description = self.conn.description
            columns = [desc[0] for desc in description] if description else []

            # Format results
            formatted_results = []
            for row in result:
                formatted_results.append(dict(zip(columns, row)))

            return True, formatted_results, None

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Query execution error: {error_msg}")
            return False, None, error_msg

    def query_to_answer(self, user_question: str, schema: str) -> Tuple[bool, str, Optional[List]]:
        """
        Complete pipeline: question -> SQL -> results

        Args:
            user_question: Natural language question
            schema: Database schema

        Returns:
            (success: bool, answer: str, data: List or None)
        """

        # Generate SQL
        sql = self.generate_sql(user_question, schema)

        if not sql:
            return False, "Could not generate SQL query", None

        # Execute SQL
        success, results, error = self.execute_query(sql)

        if not success:
            return False, f"Query failed: {error}", None

        # Format answer
        if not results:
            answer = "No results found for your query."
        elif len(results) == 1 and len(results[0]) == 1:
            # Single value result
            key = list(results[0].keys())[0]
            value = results[0][key]
            answer = f"{value}"
        else:
            # Multiple results - format as text
            answer = self._format_results_as_text(results)

        return True, answer, results

    def _format_results_as_text(self, results: List[Dict]) -> str:
        """Format query results as readable text"""

        if not results:
            return "No results found."

        # If single row, format as key-value
        if len(results) == 1:
            parts = []
            for key, value in results[0].items():
                if value is not None:
                    # Format nicely
                    if isinstance(value, float):
                        parts.append(f"{key}: {value:.1f}")
                    else:
                        parts.append(f"{key}: {value}")
            return ", ".join(parts)

        # Multiple rows - format as list
        if len(results) <= 10:
            # Show all
            lines = []
            for i, row in enumerate(results, 1):
                row_str = ", ".join([f"{k}: {v:.1f if isinstance(v, float) else v}"
                                     for k, v in row.items() if v is not None])
                lines.append(f"{i}. {row_str}")
            return "\n".join(lines)
        else:
            # Show first 10 + count
            lines = []
            for i, row in enumerate(results[:10], 1):
                row_str = ", ".join([f"{k}: {v:.1f if isinstance(v, float) else v}"
                                     for k, v in row.items() if v is not None])
                lines.append(f"{i}. {row_str}")
            lines.append(f"\n... and {len(results) - 10} more results")
            return "\n".join(lines)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    # Test the query engine
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    engine = QueryEngine(api_key=api_key)

    # Test classification
    print("🔍 Testing query classification:")
    query = "Which district is Vega Collegiate Academy from?"
    classification = engine.classify_query(query)
    print(f"Query: {query}")
    print(f"Classification: {classification}")

    # Test SQL generation
    print("\n🔍 Testing SQL generation:")
    from data_processor import DataProcessor
    processor = DataProcessor()
    schema = processor.get_schema()

    sql = engine.generate_sql(query, schema)
    print(f"Generated SQL:\n{sql}")

    # Test execution
    if sql:
        print("\n🔍 Testing query execution:")
        success, results, error = engine.execute_query(sql)
        if success:
            print(f"Results: {results}")
        else:
            print(f"Error: {error}")

    engine.close()
