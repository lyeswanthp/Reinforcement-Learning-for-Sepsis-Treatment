"""
Database Connection Utility
Handles PostgreSQL connection to MIMIC-IV v3.1
"""

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import Optional, Dict, Any
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MIMICDatabase:
    """Database connection handler for MIMIC-IV"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database connection

        Args:
            config: Database configuration dictionary
        """
        self.config = config
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                dbname=self.config['name'],
                user=self.config['user'],
                password=self.config['password'],
                host=self.config['host'],
                port=self.config['port']
            )
            logger.info(f"Connected to database: {self.config['name']}")
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    @contextmanager
    def cursor(self, cursor_factory=None):
        """
        Context manager for database cursor

        Args:
            cursor_factory: Optional cursor factory (e.g., RealDictCursor)

        Yields:
            Database cursor
        """
        cur = self.conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cur
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cur.close()

    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            Query results as pandas DataFrame
        """
        try:
            df = pd.read_sql_query(query, self.conn, params=params)
            logger.info(f"Query executed successfully. Returned {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def get_table_row_count(self, schema: str, table: str) -> int:
        """
        Get row count for a table

        Args:
            schema: Schema name (e.g., 'mimic_hosp')
            table: Table name (e.g., 'patients')

        Returns:
            Number of rows
        """
        query = sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
            sql.Identifier(schema),
            sql.Identifier(table)
        )

        with self.cursor() as cur:
            cur.execute(query)
            count = cur.fetchone()[0]

        logger.info(f"{schema}.{table}: {count:,} rows")
        return count

    def check_tables_exist(self, required_tables: Dict[str, list]) -> bool:
        """
        Check if required tables exist in the database

        Args:
            required_tables: Dict mapping schema -> list of table names

        Returns:
            True if all tables exist, False otherwise
        """
        for schema, tables in required_tables.items():
            for table in tables:
                query = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = %s
                        AND table_name = %s
                    );
                """
                with self.cursor() as cur:
                    cur.execute(query, (schema, table))
                    exists = cur.fetchone()[0]

                    if not exists:
                        logger.error(f"Table {schema}.{table} does not exist!")
                        return False

        logger.info("All required tables exist")
        return True

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __repr__(self):
        return f"MIMICDatabase(host={self.config['host']}, db={self.config['name']})"


if __name__ == "__main__":
    # Test database connection
    logging.basicConfig(level=logging.INFO)

    test_config = {
        'name': 'mimic4',
        'user': 'your_username',
        'password': 'your_password',
        'host': 'localhost',
        'port': 5432
    }

    # Test with context manager
    with MIMICDatabase(test_config) as db:
        # Check required tables
        required = {
            'mimic_hosp': ['patients', 'admissions', 'labevents'],
            'mimic_icu': ['icustays', 'chartevents', 'inputevents']
        }
        db.check_tables_exist(required)

        # Get row counts
        db.get_table_row_count('mimic_hosp', 'patients')
        db.get_table_row_count('mimic_icu', 'icustays')
