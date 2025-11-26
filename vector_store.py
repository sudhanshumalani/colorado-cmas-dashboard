"""
Vector Store Module
Handles embeddings generation and semantic search using ChromaDB and sentence-transformers.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import duckdb
from typing import List, Dict, Tuple
import json


class VectorStore:
    """Manage embeddings and semantic search for school data"""

    def __init__(self, db_path: str = "school_data.duckdb", collection_name: str = "schools"):
        """Initialize vector store"""
        self.db_path = db_path
        self.collection_name = collection_name

        # Initialize embedding model (lightweight, runs locally)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality

        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))

        self.collection = None

    def build_index(self):
        """Build vector index from school data"""

        print("Building vector index...")

        # Connect to DuckDB
        conn = duckdb.connect(self.db_path)

        # Get all charter schools (focus on charter for better results)
        schools = conn.execute("""
            SELECT
                school_id,
                school_name,
                network,
                district_name,
                frl_percent,
                ela_performance,
                math_performance,
                gradespan_category,
                ela_tercile,
                math_tercile
            FROM schools
            WHERE is_charter = true
        """).fetchall()

        conn.close()

        if not schools:
            print("❌ No schools found in database")
            return False

        # Create or get collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            # Delete existing collection to rebuild
            self.client.delete_collection(self.collection_name)
        except:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Generate embeddings
        documents = []
        metadatas = []
        ids = []

        for school in schools:
            (school_id, name, network, district, frl, ela, math,
             gradespan, ela_tercile, math_tercile) = school

            # Create rich text description for embedding
            desc = self._create_school_description(
                name, network, district, frl, ela, math, gradespan, ela_tercile, math_tercile
            )

            documents.append(desc)
            metadatas.append({
                'school_id': str(school_id),
                'school_name': str(name),
                'network': str(network) if network else 'Unknown',
                'district': str(district) if district else 'Unknown',
                'frl_percent': float(frl) if frl else 0.0,
                'ela_performance': float(ela) if ela else 0.0,
                'math_performance': float(math) if math else 0.0,
                'gradespan': str(gradespan) if gradespan else 'Unknown',
                'ela_tercile': str(ela_tercile) if ela_tercile else 'Unknown',
                'math_tercile': str(math_tercile) if math_tercile else 'Unknown'
            })
            ids.append(f"school_{school_id}")

        # Generate embeddings in batches
        print(f"Generating embeddings for {len(documents)} schools...")
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            batch_size=32
        ).tolist()

        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Vector index built with {len(documents)} schools")
        return True

    def _create_school_description(self, name, network, district, frl, ela, math,
                                   gradespan, ela_tercile, math_tercile) -> str:
        """Create rich text description for embedding"""

        desc_parts = [
            f"School: {name}",
            f"Network: {network}" if network else "",
            f"District: {district}" if district else "",
            f"Gradespan: {gradespan}" if gradespan else "",
        ]

        if frl:
            desc_parts.append(f"FRL: {frl:.0f}% (poverty level: {'high' if frl > 70 else 'moderate' if frl > 40 else 'low'})")

        if ela:
            desc_parts.append(f"ELA Performance: {ela:.0f}% ({ela_tercile} relative to trend)")

        if math:
            desc_parts.append(f"Math Performance: {math:.0f}% ({math_tercile} relative to trend)")

        # Add performance characterization
        if ela and math:
            avg_perf = (ela + math) / 2
            if avg_perf > 60:
                desc_parts.append("High performing school")
            elif avg_perf > 40:
                desc_parts.append("Moderate performing school")

        return ". ".join([p for p in desc_parts if p])

    def search(self, query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """
        Semantic search for schools

        Args:
            query: Natural language query
            top_k: Number of results to return
            filter_dict: Optional filters (e.g., {'network': 'KIPP Colorado'})

        Returns:
            List of school metadata with similarity scores
        """

        if not self.collection:
            try:
                self.collection = self.client.get_collection(self.collection_name)
            except:
                print("❌ Vector index not found. Run build_index() first.")
                return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search
        where_clause = filter_dict if filter_dict else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )

        # Format results
        formatted_results = []
        if results and results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                formatted_results.append({
                    **metadata,
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'description': results['documents'][0][i]
                })

        return formatted_results

    def get_similar_schools(self, school_name: str, top_k: int = 5) -> List[Dict]:
        """Find schools similar to a given school"""

        # Get the school's description
        conn = duckdb.connect(self.db_path)
        school_data = conn.execute("""
            SELECT school_id FROM schools WHERE school_name = ?
        """, [school_name]).fetchone()
        conn.close()

        if not school_data:
            return []

        school_id = school_data[0]

        # Get the school's embedding
        try:
            if not self.collection:
                self.collection = self.client.get_collection(self.collection_name)

            result = self.collection.get(
                ids=[f"school_{school_id}"],
                include=['embeddings', 'documents']
            )

            if not result or not result['embeddings']:
                return []

            # Search for similar schools
            similar = self.collection.query(
                query_embeddings=result['embeddings'],
                n_results=top_k + 1  # +1 because it will include itself
            )

            # Filter out the school itself and format results
            formatted_results = []
            for i, metadata in enumerate(similar['metadatas'][0]):
                if metadata['school_id'] != str(school_id):
                    formatted_results.append({
                        **metadata,
                        'similarity_score': 1 - similar['distances'][0][i]
                    })

            return formatted_results[:top_k]

        except Exception as e:
            print(f"Error finding similar schools: {e}")
            return []


if __name__ == "__main__":
    # Test the vector store
    store = VectorStore()
    store.build_index()

    # Test search
    print("\n🔍 Testing semantic search:")
    results = store.search("high poverty schools performing well in math", top_k=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['school_name']}")
        print(f"   Network: {result['network']}")
        print(f"   FRL: {result['frl_percent']:.0f}%, Math: {result['math_performance']:.0f}%")
        print(f"   Similarity: {result['similarity_score']:.3f}")
