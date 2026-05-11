import psycopg2
import numpy as np

try:
    from .feature_extractor import extract_features
except ImportError:
    from feature_extractor import extract_features

# ================= DATABASE CONFIGURATION =================
DB_HOST = "localhost"
DB_PORT = "5432" 
DB_NAME = "fruit_mmdb"  
DB_USER = "postgres"     
DB_PASS = "1" 
# ==========================================================

class Searcher:
    def __init__(self):
        """Initialize connection and load data into RAM for extremely fast search."""
        print("[Searcher] Connecting to PostgreSQL...")
        self.conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS
        )
        self.cursor = self.conn.cursor()
        
        # Load the entire image_features table into RAM
        print("[Searcher] Loading all vectors from database into memory (RAM)...")
        self.cursor.execute("SELECT image_path, label, features FROM image_features;")
        self.db_data = self.cursor.fetchall()
        print(f"[Searcher] Successfully loaded {len(self.db_data)} images ready for search!")

    def search(self, query_features, top_k=5):
        """Function to compare and find the most similar images."""
        results = []
        query_features = np.asarray(query_features, dtype=np.float32)
        query_dimension = query_features.shape[0]
        
        # Loop through each record in the database
        for row in self.db_data:
            img_path = row[0]
            label = row[1]
            
            # PostgreSQL returns a standard list, convert it back to Numpy Array for calculation
            db_features = np.asarray(row[2], dtype=np.float32)
            if db_features.shape[0] != query_dimension:
                raise ValueError(
                    "Feature dimension mismatch: "
                    f"query has {query_dimension}, database has {db_features.shape[0]}. "
                    "Please re-run src/indexer_postgres.py to rebuild the feature database."
                )
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(query_features - db_features)
            
            # Save the calculated result into the array
            results.append((distance, img_path, label))
        
        # Sort the results array by distance (smallest = most similar at the top)
        results.sort(key=lambda x: x[0])
        
        # Only return top_k results (default is 5)
        return results[:top_k]

    def close(self):
        """Close database connection when no longer needed."""
        self.cursor.close()
        self.conn.close()

# =====================================================================
# TEST RUN CODE
# =====================================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
   
    test_query_image = "dataset/query_images/Apple 13/r0_19_100.jpg"
    
    print(f"Starting to extract features from the query image: {test_query_image}")
    query_vector = extract_features(test_query_image)
    
    if query_vector is not None:
        # Initialize the search engine
        searcher = Searcher()
        
        print("\nCalculating and comparing with the entire database...")
        # Proceed to find top 5
        results = searcher.search(query_vector, top_k=5)
        
        print("\n=== TOP 5 MOST SIMILAR IMAGES RESULTS ===")
        for rank, (distance, path, label) in enumerate(results, 1):
            print(f"Top {rank} | Distance: {distance:.4f} | Fruit: {label} | Path: {path}")
        
        searcher.close()
    else:
        print("Error: Cannot read query image. Please check the path again!")