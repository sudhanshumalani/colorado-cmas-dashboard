"""
Cache Manager Module
Handles response caching to reduce API costs and improve response time.
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import sqlite3


class CacheManager:
    """Manage query and response caching"""

    def __init__(self, cache_db: str = "query_cache.db", default_ttl: int = 3600):
        """
        Initialize cache manager

        Args:
            cache_db: Path to cache database
            default_ttl: Default time-to-live in seconds (1 hour)
        """
        self.cache_db = cache_db
        self.default_ttl = default_ttl
        self._init_cache_db()

    def _init_cache_db(self):
        """Initialize cache database"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        # Create cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT,
                response TEXT,
                metadata TEXT,
                created_at INTEGER,
                expires_at INTEGER,
                hit_count INTEGER DEFAULT 0
            )
        """)

        # Create index on expiration
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires
            ON query_cache(expires_at)
        """)

        conn.commit()
        conn.close()

    def _hash_query(self, query: str) -> str:
        """Generate hash for query"""
        # Normalize query (lowercase, strip whitespace)
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available and not expired

        Args:
            query: User query

        Returns:
            Cached response dict or None
        """
        query_hash = self._hash_query(query)
        current_time = int(time.time())

        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        # Get cached response
        cursor.execute("""
            SELECT response, metadata, hit_count
            FROM query_cache
            WHERE query_hash = ? AND expires_at > ?
        """, (query_hash, current_time))

        result = cursor.fetchone()

        if result:
            response, metadata_json, hit_count = result

            # Update hit count
            cursor.execute("""
                UPDATE query_cache
                SET hit_count = hit_count + 1
                WHERE query_hash = ?
            """, (query_hash,))
            conn.commit()

            conn.close()

            return {
                'response': response,
                'metadata': json.loads(metadata_json) if metadata_json else {},
                'hit_count': hit_count + 1,
                'cached': True
            }

        conn.close()
        return None

    def set(
        self,
        query: str,
        response: str,
        metadata: Dict = None,
        ttl: int = None
    ):
        """
        Cache a response

        Args:
            query: User query
            response: Response to cache
            metadata: Additional metadata
            ttl: Time-to-live in seconds (uses default if None)
        """
        query_hash = self._hash_query(query)
        current_time = int(time.time())
        expires_at = current_time + (ttl if ttl else self.default_ttl)

        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        # Insert or replace
        cursor.execute("""
            INSERT OR REPLACE INTO query_cache
            (query_hash, query_text, response, metadata, created_at, expires_at, hit_count)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (
            query_hash,
            query,
            response,
            json.dumps(metadata) if metadata else None,
            current_time,
            expires_at
        ))

        conn.commit()
        conn.close()

    def clear_expired(self):
        """Remove expired cache entries"""
        current_time = int(time.time())

        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM query_cache
            WHERE expires_at <= ?
        """, (current_time,))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count

    def clear_all(self):
        """Clear all cache entries"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM query_cache")
        deleted_count = cursor.rowcount

        conn.commit()
        conn.close()

        return deleted_count

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        # Total entries
        cursor.execute("SELECT COUNT(*) FROM query_cache")
        total = cursor.fetchone()[0]

        # Valid entries
        current_time = int(time.time())
        cursor.execute("""
            SELECT COUNT(*) FROM query_cache
            WHERE expires_at > ?
        """, (current_time,))
        valid = cursor.fetchone()[0]

        # Total hits
        cursor.execute("SELECT SUM(hit_count) FROM query_cache")
        total_hits = cursor.fetchone()[0] or 0

        # Average hits
        cursor.execute("""
            SELECT AVG(hit_count) FROM query_cache
            WHERE hit_count > 0
        """)
        avg_hits = cursor.fetchone()[0] or 0

        # Most popular queries
        cursor.execute("""
            SELECT query_text, hit_count
            FROM query_cache
            WHERE hit_count > 0
            ORDER BY hit_count DESC
            LIMIT 5
        """)
        popular = cursor.fetchall()

        conn.close()

        return {
            'total_entries': total,
            'valid_entries': valid,
            'expired_entries': total - valid,
            'total_hits': total_hits,
            'avg_hits_per_query': round(avg_hits, 2),
            'most_popular': [
                {'query': q, 'hits': h} for q, h in popular
            ]
        }

    def warm_cache(self, common_queries: List[str], query_func):
        """
        Pre-populate cache with common queries

        Args:
            common_queries: List of common questions
            query_func: Function to get responses (takes query, returns response)
        """
        warmed = 0

        for query in common_queries:
            # Check if already cached
            if self.get(query):
                continue

            # Get response
            try:
                response = query_func(query)
                if response:
                    self.set(
                        query=query,
                        response=response,
                        metadata={'warmed': True},
                        ttl=86400  # 24 hours for pre-warmed cache
                    )
                    warmed += 1
            except Exception as e:
                print(f"⚠️ Error warming cache for '{query}': {e}")

        return warmed


if __name__ == "__main__":
    # Test cache manager
    cache = CacheManager()

    # Test set and get
    print("🔧 Testing cache...")

    query1 = "Which district is Vega Collegiate Academy from?"
    response1 = "Vega Collegiate Academy is from Adams-Arapahoe 28J district."

    # Cache response
    cache.set(query1, response1, metadata={'source': 'sql_query'})
    print("✅ Cached response")

    # Retrieve from cache
    cached = cache.get(query1)
    if cached:
        print(f"✅ Retrieved from cache: {cached['response'][:50]}...")
        print(f"   Hit count: {cached['hit_count']}")

    # Test with slight variation (should match due to normalization)
    query2 = "  which DISTRICT is vega collegiate academy from?  "
    cached2 = cache.get(query2)
    if cached2:
        print("✅ Normalized query matched!")

    # Get stats
    print("\n📊 Cache statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        if key != 'most_popular':
            print(f"   {key}: {value}")

    # Clear expired
    expired = cache.clear_expired()
    print(f"\n🗑️  Cleared {expired} expired entries")
