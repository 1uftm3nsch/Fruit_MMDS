import os
import psycopg2

try:
    from .feature_extractor import extract_features
except ImportError:
    from feature_extractor import extract_features

# ================= DATABASE CONFIGURATION =================
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "fruit_mmdb"  
DB_USER = "postgres"     
DB_PASS = "1" 
# ==========================================================

dataset_dir = "dataset/database_images"


def main():
    print("Starting database connection and scanning images...")

    try:
        # 1. Connect to PostgreSQL
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS
        )
        cursor = conn.cursor()

        # (Optional) Clear old data if running from scratch
        cursor.execute("TRUNCATE TABLE image_features RESTART IDENTITY;")

        count = 0
        # 2. Iterate through directories
        for label in os.listdir(dataset_dir):
            label_path = os.path.join(dataset_dir, label)

            if os.path.isdir(label_path):
                print(f"Processing: {label}...")

                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)

                    # Extract 519 features
                    features = extract_features(img_path)

                    if features is not None:
                        # Convert numpy array to standard Python list
                        feature_list = features.tolist()

                        # 3. SQL query to insert data
                        insert_query = """
                            INSERT INTO image_features (image_path, label, features)
                            VALUES (%s, %s, %s)
                        """
                        # Execute insert command
                        cursor.execute(insert_query, (img_path, label, feature_list))
                        count += 1

        # 4. Commit changes and close connection
        conn.commit()
        cursor.close()
        conn.close()

        print(f"\nComplete! Successfully saved {count} feature vectors to PostgreSQL.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
