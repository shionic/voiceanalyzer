#!/usr/bin/env python3
"""
Database Component for Voice Analysis Data Storage
Handles PostgreSQL operations with pgvector extension support for x-vectors.
"""

import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import Json, execute_values
from psycopg2.extensions import register_adapter
import numpy as np


class VoiceDatabase:
    """
    PostgreSQL database interface for voice analysis data.
    
    Features:
    - Stores voice analysis JSON output
    - Tracks metadata (author, source, tags, file hash)
    - Stores quality ratings (reliable and unreliable)
    - Supports x-vector embeddings using pgvector extension
    - Automatic timestamp tracking
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "voice_analysis",
                 user: str = "postgres",
                 password: str = ""):
        """
        Initialize database connection parameters.
        
        Args:
            host: Database host address
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self._connection = None
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            psycopg2 connection object
        """
        conn = psycopg2.connect(**self.connection_params)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def initialize_database(self):
        """
        Initialize database schema with all required tables and extensions.
        Creates the pgvector extension and voice_recordings table.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create main table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_recordings (
                    id SERIAL PRIMARY KEY,
                    
                    -- Voice analysis data
                    analysis_data JSONB NOT NULL,
                    
                    -- Metadata
                    author VARCHAR(255),
                    author_source VARCHAR(255),
                    tags TEXT[],
                    
                    -- File information
                    file_hash VARCHAR(64) NOT NULL UNIQUE,
                    duration FLOAT NOT NULL,
                    
                    -- Quality ratings
                    reliable_quality_rating FLOAT,
                    unreliable_quality_rating FLOAT,
                    
                    -- Embedding vector (configurable dimension, typically 192 for x-vectors)
                    x_vector vector(192),
                    
                    -- Timestamps
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_voice_recordings_file_hash 
                ON voice_recordings(file_hash);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_voice_recordings_author 
                ON voice_recordings(author);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_voice_recordings_tags 
                ON voice_recordings USING GIN(tags);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_voice_recordings_created_at 
                ON voice_recordings(created_at);
            """)
            
            # Create index for similarity search using pgvector
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_voice_recordings_x_vector 
                ON voice_recordings USING ivfflat (x_vector vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # Create GIN index for JSONB queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_voice_recordings_analysis_data 
                ON voice_recordings USING GIN(analysis_data);
            """)
            
            # Create trigger for automatic updated_at timestamp
            cursor.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """)
            
            cursor.execute("""
                DROP TRIGGER IF EXISTS update_voice_recordings_updated_at 
                ON voice_recordings;
            """)
            
            cursor.execute("""
                CREATE TRIGGER update_voice_recordings_updated_at 
                BEFORE UPDATE ON voice_recordings
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
            """)
            
            conn.commit()
            print("Database initialized successfully!")
    
    def calculate_file_hash(self, filepath: str) -> str:
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def insert_voice_recording(self,
                              analysis_data: Dict[str, Any],
                              file_hash: str,
                              duration: float,
                              author: Optional[str] = None,
                              author_source: Optional[str] = None,
                              tags: Optional[List[str]] = None,
                              reliable_quality_rating: Optional[float] = None,
                              unreliable_quality_rating: Optional[float] = None,
                              x_vector: Optional[np.ndarray] = None) -> int:
        """
        Insert a new voice recording into the database.
        
        Args:
            analysis_data: Voice analysis JSON data (from voice_analyzer.py)
            file_hash: SHA-256 hash of the audio file
            duration: Duration of the audio in seconds
            author: Name or identifier of the speaker
            author_source: Source or origin of the recording
            tags: List of tags for categorization
            reliable_quality_rating: Quality rating from reliable source
            unreliable_quality_rating: Quality rating from unreliable source
            x_vector: Speaker embedding vector (numpy array)
            
        Returns:
            ID of the inserted record
            
        Raises:
            psycopg2.IntegrityError: If file_hash already exists
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Convert x_vector to PostgreSQL array format if provided
            x_vector_str = None
            if x_vector is not None:
                if isinstance(x_vector, np.ndarray):
                    x_vector_list = x_vector.tolist()
                else:
                    x_vector_list = list(x_vector)
                x_vector_str = str(x_vector_list)
            
            cursor.execute("""
                INSERT INTO voice_recordings (
                    analysis_data,
                    author,
                    author_source,
                    tags,
                    file_hash,
                    duration,
                    reliable_quality_rating,
                    unreliable_quality_rating,
                    x_vector
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id;
            """, (
                Json(analysis_data),
                author,
                author_source,
                tags,
                file_hash,
                duration,
                reliable_quality_rating,
                unreliable_quality_rating,
                x_vector_str
            ))
            
            record_id = cursor.fetchone()[0]
            conn.commit()
            return record_id
    
    def get_recording_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a voice recording by its ID.
        
        Args:
            record_id: Database record ID
            
        Returns:
            Dictionary containing all record fields, or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    id,
                    analysis_data,
                    author,
                    author_source,
                    tags,
                    file_hash,
                    duration,
                    reliable_quality_rating,
                    unreliable_quality_rating,
                    x_vector,
                    created_at,
                    updated_at
                FROM voice_recordings
                WHERE id = %s;
            """, (record_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_dict(row)
    
    def get_recording_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a voice recording by its file hash.
        
        Args:
            file_hash: SHA-256 hash of the audio file
            
        Returns:
            Dictionary containing all record fields, or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    id,
                    analysis_data,
                    author,
                    author_source,
                    tags,
                    file_hash,
                    duration,
                    reliable_quality_rating,
                    unreliable_quality_rating,
                    x_vector,
                    created_at,
                    updated_at
                FROM voice_recordings
                WHERE file_hash = %s;
            """, (file_hash,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_dict(row)
    
    def search_by_author(self, author: str) -> List[Dict[str, Any]]:
        """
        Search recordings by author.
        
        Args:
            author: Author name or identifier
            
        Returns:
            List of matching records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    id,
                    analysis_data,
                    author,
                    author_source,
                    tags,
                    file_hash,
                    duration,
                    reliable_quality_rating,
                    unreliable_quality_rating,
                    x_vector,
                    created_at,
                    updated_at
                FROM voice_recordings
                WHERE author = %s
                ORDER BY created_at DESC;
            """, (author,))
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[Dict[str, Any]]:
        """
        Search recordings by tags.
        
        Args:
            tags: List of tags to search for
            match_all: If True, require all tags to match; if False, match any tag
            
        Returns:
            List of matching records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if match_all:
                # Require all tags to be present
                cursor.execute("""
                    SELECT 
                        id,
                        analysis_data,
                        author,
                        author_source,
                        tags,
                        file_hash,
                        duration,
                        reliable_quality_rating,
                        unreliable_quality_rating,
                        x_vector,
                        created_at,
                        updated_at
                    FROM voice_recordings
                    WHERE tags @> %s
                    ORDER BY created_at DESC;
                """, (tags,))
            else:
                # Match any tag
                cursor.execute("""
                    SELECT 
                        id,
                        analysis_data,
                        author,
                        author_source,
                        tags,
                        file_hash,
                        duration,
                        reliable_quality_rating,
                        unreliable_quality_rating,
                        x_vector,
                        created_at,
                        updated_at
                    FROM voice_recordings
                    WHERE tags && %s
                    ORDER BY created_at DESC;
                """, (tags,))
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def find_similar_speakers(self, 
                             x_vector: np.ndarray, 
                             limit: int = 10,
                             min_similarity: Optional[float] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find recordings with similar x-vectors (speaker embeddings).
        
        Args:
            x_vector: Query x-vector embedding
            limit: Maximum number of results to return
            min_similarity: Minimum cosine similarity threshold (0-1)
            
        Returns:
            List of tuples (record_dict, similarity_score) sorted by similarity
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Convert x_vector to PostgreSQL array format
            if isinstance(x_vector, np.ndarray):
                x_vector_list = x_vector.tolist()
            else:
                x_vector_list = list(x_vector)
            x_vector_str = str(x_vector_list)
            
            query = """
                SELECT 
                    id,
                    analysis_data,
                    author,
                    author_source,
                    tags,
                    file_hash,
                    duration,
                    reliable_quality_rating,
                    unreliable_quality_rating,
                    x_vector,
                    created_at,
                    updated_at,
                    1 - (x_vector <=> %s::vector) as similarity
                FROM voice_recordings
                WHERE x_vector IS NOT NULL
            """
            
            params = [x_vector_str]
            
            if min_similarity is not None:
                query += " AND (1 - (x_vector <=> %s::vector)) >= %s"
                params.append(x_vector_str)
                params.append(min_similarity)
            
            query += " ORDER BY x_vector <=> %s::vector LIMIT %s;"
            params.append(x_vector_str)
            params.append(limit)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                record = self._row_to_dict(row[:-1])  # Exclude similarity from dict
                similarity = float(row[-1])
                results.append((record, similarity))
            
            return results
    
    def update_quality_ratings(self,
                              record_id: int,
                              reliable_quality_rating: Optional[float] = None,
                              unreliable_quality_rating: Optional[float] = None) -> bool:
        """
        Update quality ratings for a recording.
        
        Args:
            record_id: Database record ID
            reliable_quality_rating: New reliable quality rating
            unreliable_quality_rating: New unreliable quality rating
            
        Returns:
            True if update successful, False if record not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if reliable_quality_rating is not None:
                updates.append("reliable_quality_rating = %s")
                params.append(reliable_quality_rating)
            
            if unreliable_quality_rating is not None:
                updates.append("unreliable_quality_rating = %s")
                params.append(unreliable_quality_rating)
            
            if not updates:
                return True
            
            params.append(record_id)
            
            cursor.execute(f"""
                UPDATE voice_recordings
                SET {', '.join(updates)}
                WHERE id = %s;
            """, params)
            
            conn.commit()
            return cursor.rowcount > 0
    
    def update_tags(self, record_id: int, tags: List[str]) -> bool:
        """
        Update tags for a recording.
        
        Args:
            record_id: Database record ID
            tags: New list of tags
            
        Returns:
            True if update successful, False if record not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE voice_recordings
                SET tags = %s
                WHERE id = %s;
            """, (tags, record_id))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_recording(self, record_id: int) -> bool:
        """
        Delete a voice recording.
        
        Args:
            record_id: Database record ID
            
        Returns:
            True if deletion successful, False if record not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM voice_recordings
                WHERE id = %s;
            """, (record_id,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total recordings
            cursor.execute("SELECT COUNT(*) FROM voice_recordings;")
            stats['total_recordings'] = cursor.fetchone()[0]
            
            # Unique authors
            cursor.execute("SELECT COUNT(DISTINCT author) FROM voice_recordings WHERE author IS NOT NULL;")
            stats['unique_authors'] = cursor.fetchone()[0]
            
            # Average duration
            cursor.execute("SELECT AVG(duration) FROM voice_recordings;")
            stats['average_duration'] = float(cursor.fetchone()[0] or 0)
            
            # Recordings with x-vectors
            cursor.execute("SELECT COUNT(*) FROM voice_recordings WHERE x_vector IS NOT NULL;")
            stats['recordings_with_xvector'] = cursor.fetchone()[0]
            
            # Average quality ratings
            cursor.execute("SELECT AVG(reliable_quality_rating) FROM voice_recordings WHERE reliable_quality_rating IS NOT NULL;")
            stats['average_reliable_quality'] = float(cursor.fetchone()[0] or 0)
            
            cursor.execute("SELECT AVG(unreliable_quality_rating) FROM voice_recordings WHERE unreliable_quality_rating IS NOT NULL;")
            stats['average_unreliable_quality'] = float(cursor.fetchone()[0] or 0)
            
            return stats
    
    def _row_to_dict(self, row: tuple) -> Dict[str, Any]:
        """
        Convert a database row to a dictionary.
        
        Args:
            row: Database row tuple
            
        Returns:
            Dictionary with named fields
        """
        return {
            'id': row[0],
            'analysis_data': row[1],
            'author': row[2],
            'author_source': row[3],
            'tags': row[4],
            'file_hash': row[5],
            'duration': row[6],
            'reliable_quality_rating': row[7],
            'unreliable_quality_rating': row[8],
            'x_vector': self._parse_vector(row[9]) if row[9] else None,
            'created_at': row[10],
            'updated_at': row[11]
        }
    
    def _parse_vector(self, vector_str: str) -> Optional[np.ndarray]:
        """
        Parse vector string from database to numpy array.
        
        Args:
            vector_str: Vector string from database
            
        Returns:
            Numpy array or None
        """
        if vector_str is None:
            return None
        try:
            # Remove brackets and split
            vector_list = [float(x) for x in vector_str.strip('[]').split(',')]
            return np.array(vector_list)
        except:
            return None


# Example usage and utility functions
def example_usage():
    """
    Example demonstrating how to use the VoiceDatabase class.
    """
    # Initialize database connection
    db = VoiceDatabase(
        host="localhost",
        port=5432,
        database="voice_analysis",
        user="postgres",
        password="your_password"
    )
    
    # Initialize database schema (run once)
    db.initialize_database()
    
    # Example: Insert a voice recording
    analysis_data = {
        "filename": "example.wav",
        "duration": 3.5,
        "sample_rate": 22050,
        "pitch_statistics": {"mean": 180.5, "std": 25.3},
        # ... more analysis data from voice_analyzer.py
    }
    
    file_hash = db.calculate_file_hash("path/to/audio.wav")
    
    # Create a sample x-vector (normally from speaker embedding model)
    x_vector = np.random.randn(512)
    
    record_id = db.insert_voice_recording(
        analysis_data=analysis_data,
        file_hash=file_hash,
        duration=3.5,
        author="John Doe",
        author_source="interview_2024",
        tags=["interview", "english", "male"],
        reliable_quality_rating=0.85,
        unreliable_quality_rating=0.72,
        x_vector=x_vector
    )
    
    print(f"Inserted record with ID: {record_id}")
    
    # Retrieve the record
    record = db.get_recording_by_id(record_id)
    print(f"Retrieved record: {record['author']}")
    
    # Search by tags
    results = db.search_by_tags(["interview", "english"])
    print(f"Found {len(results)} recordings with matching tags")
    
    # Find similar speakers
    similar_speakers = db.find_similar_speakers(x_vector, limit=5)
    for record, similarity in similar_speakers:
        print(f"Similar speaker: {record['author']} (similarity: {similarity:.3f})")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"Database statistics: {stats}")


if __name__ == "__main__":
    # Run example usage
    print("VoiceDatabase component loaded.")
    print("Use example_usage() to see how to use this module.")
