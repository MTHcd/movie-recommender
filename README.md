# Movie Recommender System

> A production-ready movie recommendation engine built with Neural Collaborative Filtering, trained on MovieLens, served via FastAPI and visualized with a Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?logo=mlflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?logo=streamlit&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-3.49-945DD6?logo=dvc&logoColor=white)

---

## Overview

This project implements a full data science pipeline for movie recommendations:

- **Data**: MovieLens (100K ratings, 9K movies, 610 users)
- **Model**: Neural Collaborative Filtering (He et al., 2017) combining Matrix Factorization and MLP branches on learned user/item embeddings
- **Training**: BPR loss with negative sampling, early stopping, learning rate scheduling
- **Tracking**: MLflow experiment tracking with per-epoch metrics
- **Serving**: REST API with automatic Swagger documentation
- **Demo**: Interactive Streamlit dashboard with user search and personalized results

---

## Results

| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| Neural Collaborative Filtering | **0.2353** | **0.0708** |

Evaluated on a chronological train/val/test split (80/10/10). Seen movies are excluded from recommendations at inference time to ensure genuine personalization.

---

## Architecture

```
MovieLens Dataset
       │
       ▼
 EDA & Preprocessing
 (pandas, seaborn)
       │
       ▼
  Neural CF (PyTorch)
  ┌────────────────────┐
  │  User Embedding    │
  │  Item Embedding    │──► MF branch (element-wise product)
  │                    │──► MLP branch (256 → 128 → 64)
  │  Fusion + Linear   │
  └────────────────────┘
       │
       ▼
  MLflow Tracking
  (params, metrics, artifacts)
       │
       ├──► FastAPI  POST /recommend
       └──► Streamlit Dashboard
```

---

## Project Structure

```
movie-recommender/
├── data/
│   ├── raw/                  # MovieLens files (tracked by DVC)
│   └── processed/            # Parquet splits + meta.pkl + model
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb # Split + PyTorch Dataset
│   ├── 03_neural_cf.ipynb    # Model training & evaluation
│   └── 04_mlflow_tracking.ipynb # Experiment tracking
├── src/
│   ├── data/                 # Data loading utilities
│   ├── models/               # Model architectures
│   ├── training/             # Training loops
│   └── evaluation/           # NDCG, Hit Rate metrics
├── api/
│   └── main.py               # FastAPI app
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── configs/
│   └── config.yaml           # Hyperparameters
├── scripts/
│   └── check_env.py          # Environment verification
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/check_env.py      # verify all packages
```

### 2. Run the notebooks in order

```
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_neural_cf.ipynb
notebooks/04_mlflow_tracking.ipynb
```

### 3. Start the API

```bash
uvicorn api.main:app --reload
```

API docs available at `http://localhost:8000/docs`

### 4. Start the dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard available at `http://localhost:8501`

### 5. (Optional) View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

MLflow UI at `http://localhost:5000`

---

## API Reference

### `GET /health`

Returns model status and dataset dimensions.

```json
{
  "status": "ok",
  "model": "NeuralCF",
  "n_users": 610,
  "n_movies": 9724
}
```

### `POST /recommend`

Returns top-N personalized movie recommendations for a given user.

**Request body:**
```json
{
  "user_id": 1,
  "n": 10
}
```

**Response:**
```json
{
  "user_id": 1,
  "recommendations": [
    {
      "rank": 1,
      "movie_id": 318,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Crime|Drama",
      "score": 4.821
    }
  ]
}
```

---

## Model Details

### Neural Collaborative Filtering

The model combines two branches on learned embeddings:

- **MF branch**: element-wise product of user and item embeddings — captures linear user-item interactions
- **MLP branch**: concatenation of embeddings passed through fully connected layers (256 → 128 → 64 with ReLU + Dropout) — captures non-linear patterns

Both branches are concatenated and passed through a final linear layer to produce a relevance score.

### Training

| Parameter | Value |
|-----------|-------|
| Embedding dim | 64 |
| MLP layers | [256, 128, 64] |
| Dropout | 0.3 |
| Loss | BPR (Bayesian Personalized Ranking) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 512 |
| Early stopping patience | 3 |

### Negative Sampling

For each positive (user, item) pair, one negative item is sampled uniformly at random from items the user has not rated. This is used to compute the BPR loss, which optimizes the ranking of positive items above negative ones.

### Inference

At inference time, all movies already rated by the user are masked with `-inf` before taking the top-K — ensuring recommendations are genuinely new to the user.

---

## Evaluation Metrics

- **Hit Rate@K (HR@K)**: proportion of users for whom the held-out positive item appears in the top-K recommendations
- **NDCG@K**: normalized discounted cumulative gain — rewards models that rank the relevant item higher in the list

Both metrics are computed on the chronological test set to simulate a realistic recommendation scenario.

---

## Key Design Decisions

**Chronological split over random split**: splitting by time prevents data leakage — the model predicts future ratings from past behavior, which mirrors real-world usage.

**BPR loss over MSE**: optimizing for ranking (is the good movie ranked above bad ones?) is more aligned with the recommendation task than minimizing rating prediction error.

**Seen-movie filtering at inference**: without this, the model tends to recommend already-watched popular movies. Filtering forces genuine personalization.

---

## About key topics

- Implementing Neural CF from scratch in PyTorch including custom Dataset with negative sampling
- The importance of chronological splitting for recommendation systems
- How BPR loss frames recommendation as a ranking problem rather than regression
- Integrating MLflow into a training loop for reproducible experiment tracking
- Building and documenting a production REST API with FastAPI and Pydantic
- The popularity bias problem and how filtering seen items mitigates it

---

## References

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031). WWW 2017.
- Harper, F. M., & Konstan, J. A. (2015). [The MovieLens Datasets](https://dl.acm.org/doi/10.1145/2827872). ACM TIIS.

---
