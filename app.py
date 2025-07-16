# chroma_setup.py

import chromadb
import pandas as pd
import os
import math
import sys
from dotenv import load_dotenv
from data_loader import load_data
from gemini_embedding import GeminiEmbeddingFunction  # Import the custom Gemini embedding function

# Load environment variables
load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DATA_FILEPATH = '../cleaned_dataset.csv'  # Updated path to the cleaned dataset

def row_to_text(row):
    """
    Converts a DataFrame row to a text representation for embedding.

    Args:
        row (pd.Series): A row from the DataFrame.

    Returns:
        str: Text representation of the row.
    """
    return (
        f"Property Number: {row['prop_num']}, "
        f"City: {row['city']}, "
        f"Development: {row['development']}, "
        f"Sale Date: {row['sale_date'].strftime('%Y-%m-%d')}, "
        f"Price AED: {row['price_aed']}, "
        f"Percentage Change: {row['percentage_change']}%, "
        f"Size: {row['size_sqft']} sqft, "
        f"Price per sqft AED: {row['price_per_sqft_aed']}, "
        f"Beds: {row['beds']}, "
        f"Halls: {row['hall']}, "
        f"Studio: {row['studio']}, "
        f"Office: {row['office']}, "
        f"First Sale: {row['first_sale']}, "
        f"Resale: {row['resale']}"
    )

def initialize_chroma_db():
    """
    Initializes Chroma DB and populates it with the dataset using batching.
    """
    try:
        print("Initializing Chroma DB client...")
        client = chromadb.Client()
        print("Chroma DB client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Chroma DB client: {e}")
        sys.exit(1)

    try:
        print("Creating or retrieving the 'real_estate_batches' collection...")
        collection = client.create_collection("real_estate_batches")
        print("Collection 'real_estate_batches' is ready.")
    except Exception as e:
        print(f"Error creating or retrieving collection: {e}")
        sys.exit(1)

    try:
        print(f"Loading data from '{DATA_FILEPATH}'...")
        df = load_data(DATA_FILEPATH)
        print(f"Data loaded successfully. Total records: {len(df)}")
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    if df.empty:
        print("The DataFrame is empty. No data to process.")
        sys.exit(1)

    try:
        texts = df.apply(row_to_text, axis=1).tolist()
        ids = df['prop_num'].astype(str).tolist()
    except Exception as e:
        print(f"Error converting DataFrame rows to text: {e}")
        sys.exit(1)

    # Define Batch Size
    BATCH_SIZE = 100  # Adjust based on your system's capabilities
    num_batches = math.ceil(len(texts) / BATCH_SIZE)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches to process: {num_batches}")

    # Initialize Gemini Embedding Function
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_EMBEDDING_URL = os.getenv("GEMINI_EMBEDDING_URL")
    GEMINI_MODEL_NAME = "models/text-embedding-004"  # Replace with the actual Gemini model name

    if not GEMINI_API_KEY or not GEMINI_EMBEDDING_URL:
        print("GEMINI_API_KEY or GEMINI_EMBEDDING_URL not set in environment variables.")
        sys.exit(1)

    try:
        print("Initializing Gemini Embedding Function...")
        embed_fn = GeminiEmbeddingFunction(
            api_key=GEMINI_API_KEY,
            api_url=GEMINI_EMBEDDING_URL,
            model_name=GEMINI_MODEL_NAME
        )
        print("Gemini Embedding Function initialized.")
    except Exception as e:
        print(f"Error initializing Gemini Embedding Function: {e}")
        sys.exit(1)

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        batch_texts = texts[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]

        print(f"Processing batch {i+1}/{num_batches}...")
        try:
            # Generate embeddings using Gemini
            embeddings = embed_fn(batch_texts)
            if not embeddings:
                print(f"Warning: No embeddings returned for batch {i+1}. Skipping.")
                continue
            print(f"Embeddings generated for batch {i+1}.")
        except Exception as e:
            print(f"Error generating embeddings for batch {i+1}: {e}")
            continue  # Skip this batch and proceed with the next

        try:
            # Add to Chroma Collection
            collection.add(
                documents=batch_texts,
                ids=batch_ids,
                embeddings=embeddings
            )
            print(f"Batch {i+1} added to Chroma DB.")
        except Exception as e:
            print(f"Error adding batch {i+1} to Chroma DB: {e}")
            continue  # Skip this batch and proceed with the next

    print("Chroma DB initialization and population complete.")

if __name__ == "__main__":
    initialize_chroma_db()
