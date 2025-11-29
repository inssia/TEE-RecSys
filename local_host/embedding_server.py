"""
Local Host - Embedding Server & TEE Client
Runs OUTSIDE the TEE (simulates the platform's infrastructure)

The client:
1. Holds the large embedding tables (would be 100s of GB in production)
2. Performs candidate generation
3. Fetches vectors by ID (locally)
4. Sends ONLY opaque vectors to TEE (no IDs cross the boundary)
5. Receives scores from TEE
"""

import os
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import pickle
import random
import sys
import argparse
import zipfile
import io


# ============================================================================
# EMBEDDING TABLES
# ============================================================================


class EmbeddingServer:
    """
    Simulates the platform's embedding infrastructure.
    In production, this would be 100s of GB of embedding tables.

    Knows user IDs and item IDs, but NEVER sends them to TEE.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # embedding tables (really huge for production-scale systems)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # init with random weights, overwritten after training
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

    def generate_candidates(
        self, user_vector: np.ndarray, top_k: int = 100
    ) -> List[int]:
        """
        Simple Candidate Generation:
        Compare user against ALL items to find the approximate best matches.
        Would use Faiss or similar in production for scalability.
        """
        user_tensor = torch.tensor(user_vector)

        # user_embeddings.weight is shape [Num_Items, 64]
        # user_tensor is shape [64]
        # result is [Num_Items] score vector
        all_scores = torch.matmul(self.item_embeddings.weight, user_tensor)

        # get the top K indices
        # this returns (values, indices) - we only want indices
        _, top_indices = torch.topk(all_scores, top_k)

        return top_indices.tolist()

    def fetch_user_vector(self, user_id: int) -> np.ndarray:
        """
        Fetch user embedding by ID - returns OPAQUE vector
        """
        with torch.no_grad():
            vec = self.user_embeddings(torch.tensor([user_id]))
        return vec.squeeze().numpy()

    def fetch_item_vector(self, item_id: int) -> np.ndarray:
        """
        Fetch item embedding by ID - returns OPAQUE vector
        """
        with torch.no_grad():
            vec = self.item_embeddings(torch.tensor([item_id]))
        return vec.squeeze().numpy()

    def fetch_item_vectors(self, item_ids: List[int]) -> np.ndarray:
        """
        Fetch multiple item embeddings - returns OPAQUE vectors
        """
        with torch.no_grad():
            vecs = self.item_embeddings(torch.tensor(item_ids))
        return vecs.numpy()

    def get_memory_size_mb(self) -> float:
        """
        Calculate memory size of embedding tables
        """
        # Each embedding is embedding_dim * 4 bytes (float32)
        user_size = self.num_users * self.embedding_dim * 4
        item_size = self.num_items * self.embedding_dim * 4
        return (user_size + item_size) / (1024 * 1024)

    def load_weights(self, user_weights: dict, item_weights: dict):
        """
        Load trained embedding weights
        """
        self.user_embeddings.load_state_dict(user_weights)
        self.item_embeddings.load_state_dict(item_weights)

    def save(self, path: str):
        """
        Save embedding tables to disk
        """
        torch.save(
            {
                "user_embeddings": self.user_embeddings.state_dict(),
                "item_embeddings": self.item_embeddings.state_dict(),
                "num_users": self.num_users,
                "num_items": self.num_items,
                "embedding_dim": self.embedding_dim,
            },
            path,
        )


# ============================================================================
# TEE CLIENT (Communication with Phala Cloud or local TEE server)
# ============================================================================


class TEEClient:
    """
    Client for communicating with the TEE-hosted relevance estimator.

    SECURITY PROPERTY: This client ONLY sends numeric vectors to TEE.
    No user IDs, item IDs, or any identifying information crosses the boundary.
    """

    def __init__(self, tee_url: str):
        self.tee_url = tee_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> dict:
        """
        Check if TEE server is healthy
        """
        response = self.session.get(f"{self.tee_url}/health")
        response.raise_for_status()
        return response.json()

    def score(self, user_vector: np.ndarray, item_vector: np.ndarray) -> float:
        """
        Score a single user-item pair via TEE.

        ONLY vectors are sent - no IDs, no identifying information.
        """
        payload = {
            "user_vector": user_vector.tolist(),
            "item_vector": item_vector.tolist(),
        }
        response = self.session.post(f"{self.tee_url}/score", json=payload)
        response.raise_for_status()
        return response.json()["score"]

    def score_batch(
        self, user_vector: np.ndarray, item_vectors: np.ndarray
    ) -> List[float]:
        """
        Score multiple items for a single user via TEE.

        This is the typical recommendation flow:
        1. Host generates candidates
        2. Host fetches all vectors
        3. Host sends vectors to TEE
        4. TEE returns scores
        5. Host ranks by score
        """
        payload = {
            "user_vector": user_vector.tolist(),
            "item_vectors": item_vectors.tolist(),
        }
        response = self.session.post(f"{self.tee_url}/score_batch", json=payload)
        response.raise_for_status()
        return response.json()["scores"]

    def get_model_info(self) -> dict:
        """
        Get model architecture info for auditing
        """
        response = self.session.get(f"{self.tee_url}/model_info")
        response.raise_for_status()
        return response.json()


# ============================================================================
# SECURE RECOMMENDATION PIPELINE
# ============================================================================


class SecureRecommendationPipeline:
    """
    Complete recommendation pipeline with TEE-based scoring.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     LOCAL HOST (Untrusted)                  │
    │  ┌──────────────────┐    ┌──────────────────────────────┐   │
    │  │ EmbeddingServer  │    │ Candidate Generation         │   │
    │  │ - user_emb (NxD) │───▶│ 1. Select candidate items    │   │
    │  │ - item_emb (MxD) │    │ 2. Fetch vectors by ID       │   │
    │  └──────────────────┘    │ 3. Send vectors (no IDs)     │   │
    │                          └──────────────────────────────┘   │
    └───────────────────────────────────│─────────────────────────┘
                                        │ vectors only
                                        ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     TEE (Trusted)                           │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │ RelevanceEstimator                                     │ │
    │  │ score = MLP(concat(user_vec, item_vec))                │ │
    │  └────────────────────────────────────────────────────────┘ │
    └───────────────────────────────────│─────────────────────────┘
                                        │ scores only
                                        ▼
                              Rankings returned to host
    """

    def __init__(self, embedding_server: EmbeddingServer, tee_client: TEEClient):
        self.embedding_server = embedding_server
        self.tee_client = tee_client

    def recommend(
        self, user_id: int, candidate_item_ids: List[int], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: The user to generate recommendations for
            candidate_item_ids: Pre-filtered candidate items
            top_k: Number of recommendations to return

        Returns:
            List of (item_id, score) tuples, sorted by score descending

        SECURITY FLOW:
        1. Host resolves user_id → user_vector (locally)
        2. Host resolves item_ids → item_vectors (locally)
        3. Host sends ONLY vectors to TEE (no IDs)
        4. TEE returns scores
        5. Host maps scores back to item_ids (locally)
        """

        # step 1-2: fetch vectors locally (IDs stay on host)
        user_vector = self.embedding_server.fetch_user_vector(user_id)
        item_vectors = self.embedding_server.fetch_item_vectors(candidate_item_ids)

        # step 3-4: send to TEE for scoring (ONLY vectors cross boundary)
        scores = self.tee_client.score_batch(user_vector, item_vectors)

        # step 5: map scores back to IDs locally
        scored_items = list(zip(candidate_item_ids, scores))
        scored_items.sort(key=lambda x: x[1], reverse=True)

        return scored_items[:top_k]

    def get_all_candidates(self) -> List[int]:
        """
        Simple candidate generation - return all items.
        In production, this would use more advanced filtering.
        """
        return list(range(self.embedding_server.num_items))


# ============================================================================
# TRAINING (For proof-of-concept with MovieLens)
# ============================================================================


class MovieLensDataset(Dataset):
    """
    PyTorch Dataset for MovieLens ratings
    """

    def __init__(self, ratings_df: pd.DataFrame, user_map: dict, item_map: dict):
        self.users = ratings_df["userId"].map(user_map).values
        self.items = ratings_df["movieId"].map(item_map).values
        # Normalize ratings to [0, 1]; original ratings are 0.5-5
        self.ratings = (ratings_df["rating"].values - 0.5) / 4.5

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32),
        )


class FullRecommender(nn.Module):
    """
    Full recommender model for training.
    After training, we split this into:
    - Embedding tables → EmbeddingServer (outside TEE)
    - Scorer → RelevanceEstimator (inside TEE)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        in_features = embedding_dim * 2

        layers = []

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())

        self.scorer = nn.Sequential(*layers)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_vecs = self.user_emb(user_ids)
        item_vecs = self.item_emb(item_ids)
        combined = torch.cat([user_vecs, item_vecs], dim=-1)
        return self.scorer(combined).squeeze()


def train_model(
    ratings_path: str,
    embedding_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 0.001,
) -> Tuple[FullRecommender, dict, dict]:
    """
    Train a recommender model on MovieLens data.

    Returns:
        model: Trained FullRecommender
        user_map: dict mapping original user IDs to indices
        item_map: dict mapping original item IDs to indices
    """
    print(f"Loading data from {ratings_path}...")
    ratings = pd.read_csv(
        ratings_path, sep="\t", names=["userId", "movieId", "rating", "timestamp"]
    )

    print(f"Original size: {len(ratings)} ratings")

    # filter out movies with fewer than 50 ratings
    # to prevent cold-start issues
    movie_counts = ratings.groupby("movieId")["rating"].count()
    valid_movie_ids = movie_counts[movie_counts >= 50].index
    ratings = ratings[ratings["movieId"].isin(valid_movie_ids)]

    print(f"Filtered size: {len(ratings)} ratings (removed items < 50 ratings)")

    # create ID mappings (MovieLens IDs aren't contiguous)
    user_ids = ratings["userId"].unique()
    item_ids = ratings["movieId"].unique()
    user_map = {id: idx for idx, id in enumerate(user_ids)}
    item_map = {id: idx for idx, id in enumerate(item_ids)}

    num_users = len(user_map)
    num_items = len(item_map)

    print(f"Dataset: {len(ratings)} ratings, {num_users} users, {num_items} items")

    # create dataset and dataloader
    dataset = MovieLensDataset(ratings, user_map, item_map)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize model
    model = FullRecommender(num_users, num_items, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (users, items, ratings_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, ratings_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} complete, Avg Loss: {avg_loss:.4f}")

    return model, user_map, item_map


def partition_model(model: FullRecommender, output_dir: str):
    """
    Split trained model into host and TEE components.

    Creates:
    - embedding_server.pt: For local host (large)
    - relevance_estimator.pt: For TEE (small)
    """
    os.makedirs(output_dir, exist_ok=True)

    # host component: embedding tables
    host_state = {
        "user_embeddings": model.user_emb.state_dict(),
        "item_embeddings": model.item_emb.state_dict(),
        "embedding_dim": model.embedding_dim,
    }
    host_path = os.path.join(output_dir, "embedding_server.pt")
    torch.save(host_state, host_path)

    # calculate host component size
    host_size = os.path.getsize(host_path) / (1024 * 1024)
    print(f"Host component (embeddings): {host_size:.2f} MB")

    # TEE component: Scorer MLP
    # map from sequential indices to RelevanceEstimator structure
    tee_state = {"mlp." + k: v for k, v in model.scorer.state_dict().items()}
    tee_path = os.path.join(output_dir, "relevance_estimator.pt")
    torch.save(tee_state, tee_path)

    # calculate TEE component size
    tee_size = os.path.getsize(tee_path) / (1024 * 1024)
    print(f"TEE component (scorer): {tee_size:.4f} MB")

    print(f"\nPartitioning complete!")
    print(f"Host: {host_path}")
    print(f"TEE:  {tee_path}")

    return host_path, tee_path


# ============================================================================
# UTILITIES FOR DATASET
# ============================================================================


def download_movielens_data():
    """
    Downloads and extracts the MovieLens 100k dataset if not present.
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    folder_name = "ml-100k"

    if os.path.exists(folder_name):
        return

    print(f"Downloading MovieLens 100k dataset from {url}...")
    try:
        r = requests.get(url)
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(".")

        print("Download and extraction complete.")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        sys.exit(1)


def print_recommendation_analysis(
    user_id: int, recommended_item_ids: List[int], data_dir: str = "ml-100k"
):
    """
    Prints a human-readable analysis of the user's history vs the model's recommendations.
    """
    items_path = os.path.join(data_dir, "u.item")
    data_path = os.path.join(data_dir, "u.data")

    # check if data exists
    if not os.path.exists(items_path) or not os.path.exists(data_path):
        print("\n[Analysis Skipped] MovieLens data not found for verification.")
        return

    try:
        # load movie titles
        # (Latin-1 encoding is required for this specific dataset)
        cols = [
            "movieId",
            "title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]
        movies = pd.read_csv(items_path, sep="|", names=cols, encoding="latin-1")
        id_to_title = dict(zip(movies["movieId"], movies["title"]))

        # load user history
        ratings = pd.read_csv(
            data_path, sep="\t", names=["userId", "movieId", "rating", "timestamp"]
        )

        # filter for high ratings
        user_history = ratings[ratings["userId"] == user_id]
        liked_movies = user_history[user_history["rating"] >= 4]

        print(f"\n{'='*20} ANALYSIS REPORT {'='*20}")
        print(f"User {user_id} History (Rated 4+ stars):")
        if liked_movies.empty:
            print("  (No high ratings found for this user)")
        else:
            # show up to 10 liked movies
            recent_likes = liked_movies["movieId"].values[-10:]
            for mid in recent_likes:
                title = id_to_title.get(mid, "Unknown Title")
                print(f"  [{mid}] {title}")
            if len(liked_movies) > 10:
                print(f"  ... and {len(liked_movies) - 10} more.")

        print(f"\nModel Recommendations:")
        for mid in recommended_item_ids:
            title = id_to_title.get(mid, "Unknown Title")
            print(f"  [{mid}] {title}")
        print(f"{'='*57}\n")

    except Exception as e:
        print(f"Error during analysis: {e}")


# ============================================================================
# CLI & ENTRY POINT LOGIC
# ============================================================================


def parse_args():
    """
    Defines and parses command line arguments.
    Separating this allows us to auto-generate docs or test CLI logic easily.
    """
    parser = argparse.ArgumentParser(description="Local Host for TEE Recommender")

    parser.add_argument(
        "--mode",
        choices=["train", "demo", "test_local"],
        required=True,
        help="Operation mode: 'train' new model, 'demo' with existing model, or 'test_local' components.",
    )

    # data arguments
    parser.add_argument(
        "--ratings",
        default="ml-100k/u.data",
        help="Path to MovieLens ratings file (default: ml-100k/u.data)",
    )
    parser.add_argument(
        "--output-dir",
        default="./model_weights",
        help="Directory to save/load model weights (default: ./model_weights)",
    )

    # network arguments
    parser.add_argument(
        "--tee-url",
        default="http://localhost:4768",  # Updated to match your Dockerfile port
        help="URL of the running TEE server",
    )

    # training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Vector size")

    return parser.parse_args()


def run_training(args):
    """
    Handler for --mode train
    """
    download_movielens_data()

    print(f"Starting training pipeline with {args.epochs} epochs...")

    # 1. train
    model, user_map, item_map = train_model(
        args.ratings, embedding_dim=args.embedding_dim, epochs=args.epochs
    )

    # 2. save metadata (mappings)
    os.makedirs(args.output_dir, exist_ok=True)
    mapping_path = os.path.join(args.output_dir, "mappings.pkl")

    with open(mapping_path, "wb") as f:
        pickle.dump({"user_map": user_map, "item_map": item_map}, f)
    print(f"Mappings saved to {mapping_path}")

    # 3. partition model
    partition_model(model, args.output_dir)


def run_local_test(args):
    """
    Handler for --mode test_local
    Validates the EmbeddingServer class without needing network calls.
    """
    print("Testing local embedding server logic...")

    # create a dummy server
    server = EmbeddingServer(num_users=1000, num_items=2000, embedding_dim=64)
    print(f"Simulated Memory Footprint: {server.get_memory_size_mb():.2f} MB")

    # verify vector shapes
    user_vec = server.fetch_user_vector(42)
    item_vec = server.fetch_item_vector(100)

    print(f"sanity check - User vector shape: {user_vec.shape}")
    print(f"sanity check - Item vector shape: {item_vec.shape}")
    print("Local test passed.")


def run_demo(args):
    """
    Handler for --mode demo
    Connects the local EmbeddingServer to the remote TEEClient.
    """
    print(f"Initializing Demo Client (Connecting to {args.tee_url})...")

    # 1. load embedding server
    host_weights_path = os.path.join(args.output_dir, "embedding_server.pt")
    mapping_path = os.path.join(args.output_dir, "mappings.pkl")

    if os.path.exists(host_weights_path) and os.path.exists(mapping_path):
        print(f"Loading trained model from {args.output_dir}...")

        # load weights and mappings
        host_data = torch.load(host_weights_path)
        with open(mapping_path, "rb") as f:
            maps = pickle.load(f)
            index_to_item_id = {v: k for k, v in maps["item_map"].items()}

        num_users = len(maps["user_map"])
        num_items = len(maps["item_map"])

        # intialize server
        embedding_server = EmbeddingServer(
            num_users, num_items, host_data["embedding_dim"]
        )
        embedding_server.load_weights(
            host_data["user_embeddings"], host_data["item_embeddings"]
        )
    else:
        print("WARNING: No trained model found. Train model before running demo.")
        sys.exit(1)

    # 2. initialize the TEE Client
    tee_client = TEEClient(args.tee_url)

    # 3. run pipeline
    try:
        health = tee_client.health_check()
        print(f"TEE Connection Established. Status: {health}")

        pipeline = SecureRecommendationPipeline(embedding_server, tee_client)

        # pick random valid index between 0 and (total users - 1)
        user_id = random.randint(0, num_users - 1)
        print(f"Randomly selected User Index: {user_id}")

        index_to_user_id = {v: k for k, v in maps["user_map"].items()}
        real_user_id = index_to_user_id.get(user_id)

        # fetch user vector
        user_vector = embedding_server.fetch_user_vector(user_id)

        # ask the server to find the 100 closest items (candidate generation)
        print(f"Generating top 100 candidates for User...")
        candidates = embedding_server.generate_candidates(user_vector, top_k=100)

        print(f"Generating recommendations for User...")
        recommendations = pipeline.recommend(user_id, candidates, top_k=10)

        real_recommended_ids = []
        print(f"\nTop 10 Recommendations:")
        print("-" * 30)
        for item_idx, score in recommendations:
            real_id = index_to_item_id.get(item_idx, "Unknown")
            real_recommended_ids.append(real_id)
            print(f"Movie ID {real_id:<6} (Index {item_idx}) | Score: {score:.4f}")

        print_recommendation_analysis(real_user_id, real_recommended_ids)

    except requests.exceptions.ConnectionError:
        print(f"\nCRITICAL ERROR: Could not connect to TEE at {args.tee_url}")
        print("1. Is the Docker container running?")
        print("2. Did you map the ports correctly?")
        print("3. Is the TEE server healthy?")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during the demo: {e}")
        sys.exit(1)


def main():
    """
    Main entry point. Dispatches execution to the appropriate handler.
    """
    args = parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "test_local":
        run_local_test(args)
    elif args.mode == "demo":
        run_demo(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
