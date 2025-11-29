# TEE-Based Recommendation System Audit Framework

Proof-of-concept implementation for auditing social media recommendation algorithms using Trusted Execution Environments (TEEs).

This project uses the MovieLens 100k dataset, collected by the GroupLens Research Project at the University of Minnesota (see: https://grouplens.org/datasets/movielens/100k/).

This project can be run either fully locally or with a cloud TEE component. For the full proof-of-concept, it is recommended that you run it with the cloud deployment.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     UNTRUSTED HOST (Platform's Infrastructure)      │
│                                                                     │
│  ┌──────────────────────────┐    ┌───────────────────────────────┐  │
│  │    EMBEDDING TABLES      │    │    CANDIDATE GENERATION       │  │
│  │                          │    │                               │  │
│  │  • User embeddings       │───▶│  1. Select candidate items    │  │
│  │    (N × D matrix)        │    │  2. Fetch user_vec by ID      │  │
│  │  • Item embeddings       │    │  3. Fetch item_vecs by IDs    │  │
│  │    (M × D matrix)        │    │  4. Send ONLY vectors to TEE  │  │
│  │                          │    │     (NO IDs cross boundary)   │  │
│  │  Size: 100s of GB        │    │                               │  │
│  └──────────────────────────┘    └───────────────────────────────┘  │
│                                              │                      │
└──────────────────────────────────────────────│──────────────────────┘
                                               │ vectors only
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TEE (Phala Cloud / SGX / TDX)                   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    RELEVANCE ESTIMATOR                         │ │
│  │                                                                │ │
│  │    score = MLP(concat(user_vector, item_vector))               │ │
│  │                                                                │ │
│  │    • Stateless scoring function                                │ │
│  │    • No knowledge of user/item identities                      │ │
│  │    • Auditable logic                                           │ │
│  │    • Size: < 100 MB                                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└──────────────────────────────────────────────│──────────────────────┘
                                               │ scores only
                                               ▼
                                     Rankings returned to host
```

## Properties

1. The TEE never sees user IDs or item IDs - only opaque numeric vectors
2. The TEE is a pure function: `score = f(user_vec, item_vec)`
3. TEE provides proof of code integrity via remote attestation
4. Scoring logic is protected from the host platform
5. The model can be cleanly partitioned
6. Scorer fits in TEE memory limits
4. Works with real TEE infrastructure (Phala Cloud)

## Overview of Project Structure

```
TEE-RecSys/
├── local_host/
│   ├── embedding_server.py    # Embedding server + TEE client
│   └── requirements.txt
└── tee_server/
    ├── tee_scorer_app.py      # Flask API for TEE scoring service
    ├── Dockerfile             # For Phala Cloud deployment
    ├── docker-compose.yml     # Phala Cloud configuration
    └── requirements.txt
```

## Running the Code

## Local Deployment

This will:
- Download MovieLens 100K dataset
- Train a recommendation model
- Partition it into embeddings (host) and scorer (TEE)
- Run recommendations through the secure pipeline

### Step 1: Clone the Repo

### Step 2: Train the Model Locally 

Embeddings aren't provided in this repo to mimic the actual architecture. 

```bash
cd local_host
pip install -r requirements.txt
python embedding_server.py --mode test_local (for sanity check)
python embedding_server.py --mode train (for training)
```

You should see a new directory called `model_weights`, containing:
- `embedding_server.pt`
- `mappings.pkl`
- `relevance_estimator.pt`

This is the partitioned model. 
To update the TEE relevance_estimator model, you will have to copy the new `relevance_estimator.pt` into the tee_server/model_weights directory.
To run the scorer locally, create a new directory called app/model and copy `relevance_estimator.pt` into that. This is because tee_scorer assumes it is running in a cloud env. 

### Step 3: Run the Scorer Model Locally

Before trying Phala Cloud deployment, test locally. 

```bash
cd tee_server
pip install -r requirements.txt
python tee_scorer_app.py
```

You are now able to specify the TEE URL to be the locally running model, by running :

```bash
cd local_host
python embedding_server.py --mode demo --tee-url http://127.0.0.1:4768
```

Once this works correctly, you are ready to deploy to a phala cloud TEE!

## Phala Cloud Deployment

### Step 1: Prepare Docker Image

```bash
cd tee_server

# build the Docker image
docker build -t <your-dockerhub-username>/tee-relevance-estimator:latest .

# push to Docker Hub
docker login
docker push <your-dockerhub-username>/tee-relevance-estimator:latest
```

### Step 2: Deploy to Phala Cloud

Follow instructions on https://docs.phala.com/phala-cloud/getting-started/start-from-cloud-ui
You can find the docker-compose spec for phala cloud in tee_server/. Just copy and paste the .yml file into the UI.

### Step 3: Get Your TEE Endpoint

After deployment, Phala provides a URL like:
```
https://<app-id>-4768.dstack-prod5.phala.network
```

### Step 4: Run Demo with Real TEE

```bash
python demo.py --mode demo --tee-url https://<app-id>-4768.dstack-prod5.phala.network
```

## API Endpoints (TEE Server)

 Endpoint        | Method | Description 
-----------------|--------|-------------
 `/health`       | GET    | Health check
 `/score`        | POST   | Score single user-item pair 
 `/score_batch`  | POST   | Score multiple items for one user 
 `/model_info`   | GET    | Get model architecture (for auditing) 


## References

- Phala Cloud Documentation: https://docs.phala.com
- Intel SGX: https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html
- MovieLens Datasets: https://grouplens.org/datasets/movielens/
