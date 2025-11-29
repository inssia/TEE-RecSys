"""
TEE Server - Relevance Estimator
Runs inside Phala Cloud TEE (Trusted Execution Environment)

This is a STATELESS scorer that receives opaque vectors and returns scores.
It never sees user IDs or item IDs - only numeric vectors.
"""

import torch
import torch.nn as nn
import os, sys
from typing import List
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

# global model instance
model = None

# ============================================================================
# RELEVANCE ESTIMATOR MODEL
# ============================================================================


class RelevanceEstimator(nn.Module):
    """
    Stateless scoring function -- the core auditable unit.
    Takes pre-computed user and item vectors, outputs relevance score.

    This model runs entirely inside the TEE -- an auditor can:
    1. Inspect the model
    2. Verify the scoring logic
    3. Run fairness tests on the ranking behavior
    """

    def __init__(self, input_dim: int = 64, hidden_dims: List[int] = [128, 64]):
        super().__init__()

        self.embedding_dim = input_dim

        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")

        layers = []

        # input layer: combines User (input_dim) + Item (input_dim) -> First Hidden
        in_features = input_dim * 2

        # dynamically build the layers based on the list provided
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim

        # final output layer
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        # ensure inputs are at least 2D (batch_size, dim)
        if user_vec.dim() == 1:
            user_vec = user_vec.unsqueeze(0)
        if item_vec.dim() == 1:
            item_vec = item_vec.unsqueeze(0)

        # broadcasting: If 1 user and N items, expand user to N
        if user_vec.size(0) == 1 and item_vec.size(0) > 1:
            user_vec = user_vec.expand(item_vec.size(0), -1)

        # check for shape mismatch before crashing
        if user_vec.size(0) != item_vec.size(0):
            raise ValueError(
                f"Batch size mismatch: User {user_vec.size(0)} vs Item {item_vec.size(0)}"
            )

        combined = torch.cat([user_vec, item_vec], dim=-1)
        return self.mlp(combined)


# Load configuration from environment
try:
    EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 64))
    PORT = int(os.environ.get("PORT", 4768))
    MODEL_PATH = os.environ.get("MODEL_PATH", "app/model/relevance_estimator.pt")
except ValueError as e:
    sys.exit(f"Configuration Error: {e}")


def load_model() -> RelevanceEstimator:
    """
    Loads the model from disk safely.
    Returns the model object or raises an error.
    """
    global model
    device = torch.device("cpu")  # force CPU for safety

    model = RelevanceEstimator(input_dim=EMBEDDING_DIM)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at: {MODEL_PATH}")

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load model state: {e}")

    model.eval()  # set to inference mode
    return model


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint - for sanity!
    """
    global model
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "tee_environment": os.environ.get("TEE_ENV", "unknown"),
        }
    )


@app.route("/score", methods=["POST"])
def score():
    """
    Score a single user-item pair.

    Request body:
    {
        "user_vector": [float, float, ...],  # embedding_dim floats
        "item_vector": [float, float, ...]   # embedding_dim floats
    }

    Response:
    {
        "score": float  # relevance score between 0 and 1
    }

    SECURITY: This endpoint receives ONLY opaque vectors.
    No user IDs, no item IDs, no identifying information.
    """
    global model
    data = request.get_json()

    # lazy load model for gunicorn deployments
    if model is None:
        try:
            load_model()
        except:
            return jsonify({"error": "Model failed to load"}), 503

    if "user_vector" not in data or "item_vector" not in data:
        return jsonify({"error": "Missing user_vector or item_vector"}), 400

    try:
        user_vec = torch.tensor(data["user_vector"], dtype=torch.float32)
        item_vec = torch.tensor(data["item_vector"], dtype=torch.float32)

        with torch.no_grad():
            score = model(user_vec, item_vec)

        return jsonify({"score": float(score.squeeze())})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/score_batch", methods=["POST"])
def score_batch():
    """
    Score multiple items for a single user (typical recommendation scenario).

    Request body:
    {
        "user_vector": [float, ...],           # single user embedding
        "item_vectors": [[float, ...], ...]    # list of item embeddings
    }

    Response:
    {
        "scores": [float, float, ...]  # one score per item
        "compute_time_ms": float       # pure computation time
        "total_server_time_ms": float  # total time inside this function
    }
    """
    global model
    data = request.get_json()

    # lazy load model for gunicorn deployments
    if model is None:
        try:
            load_model()
        except:
            return jsonify({"error": "Model failed to load"}), 503

    if "user_vector" not in data or "item_vectors" not in data:
        return jsonify({"error": "Missing user_vector or item_vectors"}), 400

    try:
        # time only the tensor conversion
        t0 = time.perf_counter()
        user_vec = torch.tensor(data["user_vector"], dtype=torch.float32)
        item_vecs = torch.tensor(data["item_vectors"], dtype=torch.float32)
        t1 = time.perf_counter()
        tensor_time = (t1 - t0) * 1000

        # time only the forward pass (pure computation)
        t2 = time.perf_counter()
        with torch.no_grad():
            scores = model(user_vec, item_vecs)
        t3 = time.perf_counter()
        inference_time = (t3 - t2) * 1000

        # time the conversion back to list
        t4 = time.perf_counter()
        scores_list = scores.squeeze().tolist()
        t5 = time.perf_counter()
        serialization_time = (t5 - t4) * 1000

        total_compute = tensor_time + inference_time + serialization_time

        return jsonify(
            {
                "scores": scores_list,
                "compute_time_ms": inference_time,  # Just the forward pass
                "total_server_time_ms": total_compute,  # Everything inside this function
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    """
    Return model architecture info for auditing purposes.
    An auditor can verify the model structure without seeing the weights.
    """
    global model

    # lazy load model for gunicorn deployments
    if model is None:
        try:
            load_model()
        except:
            return jsonify({"error": "Model failed to load"}), 503

    info = {
        "architecture": "RelevanceEstimator",
        "embedding_dim": model.embedding_dim,
        "layers": [],
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }

    for name, layer in model.mlp.named_children():
        if isinstance(layer, nn.Linear):
            info["layers"].append(
                {
                    "name": name,
                    "type": "Linear",
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                }
            )
        elif isinstance(layer, nn.ReLU):
            info["layers"].append({"name": name, "type": "ReLU"})
        elif isinstance(layer, nn.Sigmoid):
            info["layers"].append({"name": name, "type": "Sigmoid"})

    return jsonify(info)


# for local testing of scoring without cloud deployment
# dockerfile is set up with gunicorn
if __name__ == "__main__":
    # get port from environment or default to 4768
    port = int(os.environ.get("PORT", 4768))

    print(f"Attempting to load model with dim {EMBEDDING_DIM}...")
    load_model()  # hard crash if loading fails
    print("Model loaded successfully.")

    print(f"Starting TEE Server on port {port}...")
    app.run(host="0.0.0.0", port=port)


# ============================================================================
# MEMORY PROFILING ENDPOINTS (Optional)
# ============================================================================

import psutil

"""
Monolithic Model (for memory testing purposes only)
basically combines huge user and item embeddings with tiny MLP
"""


class MonolithicRecSys(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embeddings(user_ids)
        item_vecs = self.item_embeddings(item_ids)
        combined = torch.cat([user_vecs, item_vecs], dim=-1)
        return self.mlp(combined)


@app.route("/test_memory", methods=["POST"])
def test_memory():
    """
    Test if a model of given size can be loaded

    Request:
    {
        "size_mb": 100,
        "embedding_dim": 64  (optional, default 64)
    }

    Response:
    {
        "success": true/false,
        "size_mb": 100,
        "stage_reached": "allocation" | "eval" | "inference",
        "allocation_time_ms": 123.45,
        "inference_time_ms": 10.23,
        "memory_used_mb": 105.67,
        "error": "error message if failed"
    }
    """
    data = request.get_json()

    if not data or "size_mb" not in data:
        return jsonify({"error": "Missing size_mb parameter"}), 400

    target_size_mb = data["size_mb"]
    embedding_dim = data.get("embedding_dim", 64)

    # calculate dimensions needed
    bytes_per_embedding = embedding_dim * 4
    total_embeddings_needed = (target_size_mb * 1024 * 1024) / bytes_per_embedding
    num_users = int(total_embeddings_needed * 0.5)
    num_items = int(total_embeddings_needed * 0.5)

    # get initial memory
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    result = {
        "success": False,
        "requested_size_mb": target_size_mb,
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "stage_reached": None,
        "allocation_time_ms": None,
        "inference_time_ms": None,
        "memory_used_mb": None,
        "error": None,
    }

    try:
        # stage 1: allocation
        t0 = time.perf_counter()
        model = MonolithicRecSys(num_users, num_items, embedding_dim)
        t1 = time.perf_counter()

        result["allocation_time_ms"] = (t1 - t0) * 1000
        result["stage_reached"] = "allocation"

        mem_after_alloc = process.memory_info().rss / (1024 * 1024)
        result["memory_used_mb"] = mem_after_alloc - mem_before

        # stage 2: eval
        model.eval()
        result["stage_reached"] = "eval"

        # stage 3: inference
        user_ids = torch.randint(0, num_users, (100,))
        item_ids = torch.randint(0, num_items, (100,))

        t2 = time.perf_counter()
        with torch.no_grad():
            model(user_ids, item_ids)
        t3 = time.perf_counter()

        result["inference_time_ms"] = (t3 - t2) * 1000
        result["stage_reached"] = "inference"
        result["success"] = True

        # clean up
        del model

    except RuntimeError as e:
        error_msg = str(e)
        result["error"] = error_msg

        if "out of memory" in error_msg.lower():
            result["error_type"] = "OOM"
        else:
            result["error_type"] = "RuntimeError"

    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = "UnexpectedException"

    return jsonify(result)


@app.route("/binary_search", methods=["POST"])
def binary_search_limit():
    """
    Perform binary search to find exact memory limit

    Request:
    {
        "min_mb": 10,
        "max_mb": 500,
        "tolerance_mb": 10  (optional, default 10)
    }
    """
    data = request.get_json()

    min_mb = data.get("min_mb", 10)
    max_mb = data.get("max_mb", 500)
    tolerance = data.get("tolerance_mb", 10)

    results = []

    while max_mb - min_mb > tolerance:
        mid_mb = (min_mb + max_mb) // 2

        # test this size
        test_result = test_memory_internal(mid_mb)
        results.append(test_result)

        if test_result["success"]:
            min_mb = mid_mb  # can go higher
        else:
            max_mb = mid_mb  # hit limit, go lower

        # clean up between tests
        time.sleep(1)

    return jsonify(
        {"estimated_limit_mb": min_mb, "tests_run": len(results), "details": results}
    )


def test_memory_internal(size_mb):
    """
    Internal version that doesn't require HTTP request
    """
    embedding_dim = 64
    bytes_per_embedding = embedding_dim * 4
    total_embeddings_needed = (size_mb * 1024 * 1024) / bytes_per_embedding
    num_users = int(total_embeddings_needed * 0.5)
    num_items = int(total_embeddings_needed * 0.5)

    result = {
        "size_mb": size_mb,
        "success": False,
        "stage_reached": None,
        "error": None,
    }

    try:
        model = MonolithicRecSys(num_users, num_items, embedding_dim)
        result["stage_reached"] = "allocation"

        model.eval()
        result["stage_reached"] = "eval"

        user_ids = torch.randint(0, num_users, (10,))
        item_ids = torch.randint(0, num_items, (10,))

        with torch.no_grad():
            _ = model(user_ids, item_ids)

        result["stage_reached"] = "inference"
        result["success"] = True

        del model

    except Exception as e:
        result["error"] = str(e)

    return result
