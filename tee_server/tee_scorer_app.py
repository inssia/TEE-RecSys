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
