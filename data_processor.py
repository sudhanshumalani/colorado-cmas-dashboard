"""
Data Processor Module
Handles loading Excel files, creating DuckDB database, and generating statistical summaries.
"""

import pandas as pd
import duckdb
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from scipy import stats as scipy_stats


class DataProcessor:
    """Process school data and create optimized database"""

    def __init__(self, db_path: str = "school_data.duckdb"):
        """Initialize data processor with database path"""
        self.db_path = db_path
        self.conn = None

    def load_and_process(self, ela_file: str, math_file: str) -> bool:
        """
        Load Excel files and create optimized DuckDB database

        Args:
            ela_file: Path to ELA Excel file
            math_file: Path to Math Excel file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load Excel files
            print("Loading Excel files...")
            df_ela = pd.read_excel(ela_file)
            df_math = pd.read_excel(math_file)

            # Process and merge data
            print("Processing data...")
            df_merged = self._merge_and_clean(df_ela, df_math)

            # Create database
            print("Creating DuckDB database...")
            self._create_database(df_merged)

            # Generate statistical summaries
            print("Generating statistical summaries...")
            self._generate_summaries()

            print("✅ Data processing complete!")
            return True

        except Exception as e:
            print(f"❌ Error processing data: {str(e)}")
            return False

    def _merge_and_clean(self, df_ela: pd.DataFrame, df_math: pd.DataFrame) -> pd.DataFrame:
        """Merge ELA and Math data, clean and standardize"""

        # Extract relevant columns from ELA
        ela_cols = {
            'School Name': 'school_name',
            'Network': 'network',
            'District Name': 'district_name',
            'School type': 'school_type',
            'Percent FRL': 'frl_percent',
            'School Performance Value': 'ela_performance',
            'Gradespan Level': 'gradespan',
            'CSF Portfolio': 'csf_portfolio',
            'Charter Y/N': 'is_charter'
        }

        # Handle alternate column names
        if 'School Type' in df_ela.columns:
            ela_cols['School Type'] = 'school_type'
        if 'School type' in df_ela.columns:
            ela_cols['School type'] = 'school_type'

        df_ela_processed = df_ela.rename(columns=ela_cols)

        # Extract relevant columns from Math
        df_math_processed = df_math[['School Name', 'School Performance Value']].rename(
            columns={'School Name': 'school_name', 'School Performance Value': 'math_performance'}
        )

        # Merge
        df = df_ela_processed[[c for c in ela_cols.values() if c in df_ela_processed.columns]].merge(
            df_math_processed,
            on='school_name',
            how='outer'
        )

        # Clean data
        df['frl_percent'] = pd.to_numeric(df['frl_percent'], errors='coerce')
        df['ela_performance'] = pd.to_numeric(df['ela_performance'], errors='coerce')
        df['math_performance'] = pd.to_numeric(df['math_performance'], errors='coerce')

        # Convert FRL to percentage if needed
        if df['frl_percent'].max() <= 1.0:
            df['frl_percent'] = df['frl_percent'] * 100

        # Clip values to valid ranges
        df['frl_percent'] = df['frl_percent'].clip(0, 100)
        df['ela_performance'] = df['ela_performance'].clip(0, 100)
        df['math_performance'] = df['math_performance'].clip(0, 100)

        # Categorize gradespan
        df['gradespan_category'] = df['gradespan'].apply(self._categorize_gradespan)

        # Standardize charter designation
        if 'is_charter' in df.columns:
            df['is_charter'] = df['is_charter'].apply(lambda x: str(x).upper() == 'Y')
        elif 'school_type' in df.columns:
            df['is_charter'] = df['school_type'].str.upper().str.contains('CHARTER', na=False)
        else:
            df['is_charter'] = False

        # Calculate terciles
        df = self._calculate_terciles(df)

        # Drop rows with missing school names
        df = df.dropna(subset=['school_name'])

        # Add unique ID
        df['school_id'] = range(1, len(df) + 1)

        return df

    def _categorize_gradespan(self, gradespan) -> str:
        """Categorize gradespan into Elementary/Middle/High/Multiple"""
        if pd.isna(gradespan):
            return 'Unknown'

        gradespan_str = str(gradespan).upper()

        if 'ELEMENTARY' in gradespan_str:
            return 'Elementary'
        elif 'MIDDLE' in gradespan_str:
            return 'Middle'
        elif 'HIGH' in gradespan_str:
            return 'High'
        elif 'K-12' in gradespan_str or 'K-8' in gradespan_str or 'K-6' in gradespan_str:
            return 'Multiple'
        else:
            return 'Unknown'

    def _calculate_terciles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance terciles relative to trendline"""

        # Remove rows with missing data for regression
        valid_data = df.dropna(subset=['frl_percent', 'ela_performance', 'math_performance'])

        if len(valid_data) < 10:
            df['ela_tercile'] = 'Unknown'
            df['math_tercile'] = 'Unknown'
            return df

        x = valid_data['frl_percent'].values
        y_ela = valid_data['ela_performance'].values
        y_math = valid_data['math_performance'].values

        # Calculate ELA terciles
        try:
            slope_ela, intercept_ela, _, _, _ = scipy_stats.linregress(x, y_ela)
            df['ela_residual'] = df['ela_performance'] - (slope_ela * df['frl_percent'] + intercept_ela)

            ela_67 = df['ela_residual'].quantile(0.67)
            ela_33 = df['ela_residual'].quantile(0.33)

            df['ela_tercile'] = df['ela_residual'].apply(
                lambda r: 'Top Third' if pd.notna(r) and r >= ela_67
                else ('Middle Third' if pd.notna(r) and r >= ela_33 else 'Bottom Third')
            )
        except:
            df['ela_tercile'] = 'Unknown'

        # Calculate Math terciles
        try:
            slope_math, intercept_math, _, _, _ = scipy_stats.linregress(x, y_math)
            df['math_residual'] = df['math_performance'] - (slope_math * df['frl_percent'] + intercept_math)

            math_67 = df['math_residual'].quantile(0.67)
            math_33 = df['math_residual'].quantile(0.33)

            df['math_tercile'] = df['math_residual'].apply(
                lambda r: 'Top Third' if pd.notna(r) and r >= math_67
                else ('Middle Third' if pd.notna(r) and r >= math_33 else 'Bottom Third')
            )
        except:
            df['math_tercile'] = 'Unknown'

        return df

    def _create_database(self, df: pd.DataFrame):
        """Create DuckDB database with optimized schema and indexes"""

        # Create connection
        self.conn = duckdb.connect(self.db_path)

        # Create main schools table
        self.conn.execute("""
            CREATE OR REPLACE TABLE schools AS
            SELECT * FROM df
        """)

        # Create indexes for fast lookups
        self.conn.execute("CREATE INDEX idx_school_name ON schools(school_name)")
        self.conn.execute("CREATE INDEX idx_network ON schools(network)")
        self.conn.execute("CREATE INDEX idx_district ON schools(district_name)")
        self.conn.execute("CREATE INDEX idx_frl ON schools(frl_percent)")
        self.conn.execute("CREATE INDEX idx_charter ON schools(is_charter)")

        # Create charter schools view
        self.conn.execute("""
            CREATE OR REPLACE VIEW charter_schools AS
            SELECT * FROM schools WHERE is_charter = true
        """)

        # Create network summary view
        self.conn.execute("""
            CREATE OR REPLACE VIEW network_summary AS
            SELECT
                network,
                COUNT(*) as school_count,
                AVG(frl_percent) as avg_frl,
                AVG(ela_performance) as avg_ela,
                AVG(math_performance) as avg_math,
                AVG(CASE WHEN ela_tercile = 'Top Third' THEN 1.0 ELSE 0.0 END) as pct_top_third_ela,
                AVG(CASE WHEN math_tercile = 'Top Third' THEN 1.0 ELSE 0.0 END) as pct_top_third_math
            FROM schools
            WHERE network IS NOT NULL
            GROUP BY network
            ORDER BY school_count DESC
        """)

        # Create FRL bands view
        self.conn.execute("""
            CREATE OR REPLACE VIEW frl_bands AS
            SELECT
                CASE
                    WHEN frl_percent >= 90 THEN '90-100%'
                    WHEN frl_percent >= 80 THEN '80-90%'
                    WHEN frl_percent >= 70 THEN '70-80%'
                    WHEN frl_percent >= 60 THEN '60-70%'
                    WHEN frl_percent >= 50 THEN '50-60%'
                    WHEN frl_percent >= 40 THEN '40-50%'
                    WHEN frl_percent >= 30 THEN '30-40%'
                    WHEN frl_percent >= 20 THEN '20-30%'
                    WHEN frl_percent >= 10 THEN '10-20%'
                    ELSE '0-10%'
                END as frl_band,
                COUNT(*) as school_count,
                AVG(ela_performance) as avg_ela,
                AVG(math_performance) as avg_math
            FROM schools
            GROUP BY frl_band
            ORDER BY frl_band DESC
        """)

        print(f"✅ Created database with {len(df)} schools")

    def _generate_summaries(self):
        """Generate and cache statistical summaries"""

        summaries = {}

        # Overall statistics
        result = self.conn.execute("""
            SELECT
                COUNT(*) as total_schools,
                COUNT(CASE WHEN is_charter THEN 1 END) as charter_schools,
                AVG(frl_percent) as avg_frl,
                AVG(ela_performance) as avg_ela,
                AVG(math_performance) as avg_math,
                CORR(frl_percent, ela_performance) as frl_ela_correlation,
                CORR(frl_percent, math_performance) as frl_math_correlation
            FROM schools
        """).fetchone()

        summaries['overall'] = {
            'total_schools': result[0],
            'charter_schools': result[1],
            'district_schools': result[0] - result[1],
            'avg_frl': round(result[2], 1),
            'avg_ela': round(result[3], 1),
            'avg_math': round(result[4], 1),
            'frl_ela_correlation': round(result[5], 3),
            'frl_math_correlation': round(result[6], 3)
        }

        # Network rankings
        network_data = self.conn.execute("""
            SELECT * FROM network_summary
            ORDER BY avg_ela DESC
            LIMIT 10
        """).fetchall()

        summaries['top_networks'] = [
            {
                'network': row[0],
                'school_count': row[1],
                'avg_frl': round(row[2], 1),
                'avg_ela': round(row[3], 1),
                'avg_math': round(row[4], 1)
            }
            for row in network_data
        ]

        # Save to JSON
        with open('data_summaries.json', 'w') as f:
            json.dump(summaries, f, indent=2)

        print("✅ Generated statistical summaries")

    def get_schema(self) -> str:
        """Get database schema as string for Claude"""

        if not self.conn:
            self.conn = duckdb.connect(self.db_path)

        schema = """
DATABASE SCHEMA:

Table: schools
Columns:
- school_id (INTEGER): Unique identifier
- school_name (VARCHAR): Name of the school
- network (VARCHAR): Network/CMO name (e.g., "KIPP Colorado", "Single Site Charter School")
- district_name (VARCHAR): School district name
- school_type (VARCHAR): Type of school
- is_charter (BOOLEAN): True if charter school
- frl_percent (DOUBLE): Free/Reduced Lunch percentage (0-100)
- ela_performance (DOUBLE): ELA test performance percentage (0-100)
- math_performance (DOUBLE): Math test performance percentage (0-100)
- gradespan (VARCHAR): Grade levels served
- gradespan_category (VARCHAR): Elementary/Middle/High/Multiple
- csf_portfolio (VARCHAR): CSF portfolio status
- ela_tercile (VARCHAR): Top Third/Middle Third/Bottom Third (relative to trendline)
- math_tercile (VARCHAR): Top Third/Middle Third/Bottom Third (relative to trendline)
- ela_residual (DOUBLE): Distance from ELA trendline
- math_residual (DOUBLE): Distance from Math trendline

Views Available:
- charter_schools: Filtered to charter schools only
- network_summary: Aggregated stats by network
- frl_bands: Performance grouped by FRL ranges

Example Queries:
- SELECT * FROM schools WHERE network = 'KIPP Colorado'
- SELECT * FROM charter_schools WHERE frl_percent > 70 AND ela_tercile = 'Top Third'
- SELECT * FROM network_summary ORDER BY avg_ela DESC
"""
        return schema

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    # Test the processor
    processor = DataProcessor()
    success = processor.load_and_process(
        "2025 CMAS Performance_ELA.xlsx",
        "2025 CMAS Performance_Math.xlsx"
    )

    if success:
        print("\n" + processor.get_schema())
        processor.close()
