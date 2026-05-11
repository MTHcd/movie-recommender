import pickle
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class NeuralCF(nn.Module):
    def __init__(self, n_users, n_movies, emb_dim=64, mlp_layers=[256,128,64], dropout=0.3):
        super().__init__()
        self.user_emb_mf   = nn.Embedding(n_users,  emb_dim)
        self.movie_emb_mf  = nn.Embedding(n_movies, emb_dim)
        self.user_emb_mlp  = nn.Embedding(n_users,  emb_dim)
        self.movie_emb_mlp = nn.Embedding(n_movies, emb_dim)
        mlp_input = emb_dim * 2
        layers = []
        for out_dim in mlp_layers:
            layers += [nn.Linear(mlp_input, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            mlp_input = out_dim
        self.mlp    = nn.Sequential(*layers)
        self.output = nn.Linear(emb_dim + mlp_layers[-1], 1)

    def forward(self, user, item):
        mf_out  = self.user_emb_mf(user) * self.movie_emb_mf(item)
        mlp_out = self.mlp(torch.cat([self.user_emb_mlp(user),
                                      self.movie_emb_mlp(item)], dim=-1))
        return self.output(torch.cat([mf_out, mlp_out], dim=-1)).squeeze()


DATA_DIR = Path('data/processed')

with open(DATA_DIR / 'meta.pkl', 'rb') as f:
    meta = pickle.load(f)

N_USERS   = meta['n_users']
N_MOVIES  = meta['n_movies']
idx2movie = meta['idx2movie']
user2idx  = meta['user2idx']

DEVICE = torch.device('cpu')
model  = NeuralCF(N_USERS, N_MOVIES)
state  = torch.load(DATA_DIR / 'ncf_best.pt', map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()

movies_df     = pd.read_csv('data/raw/ml-latest-small/movies.csv')
movieid2title = dict(zip(movies_df['movieId'], movies_df['title']))
movieid2genre = dict(zip(movies_df['movieId'], movies_df['genres']))

# Charge les ratings une seule fois au démarrage
ratings_df = pd.read_parquet(DATA_DIR / 'train.parquet')


class RecommendRequest(BaseModel):
    user_id: int
    n: int = 10

class MovieRecommendation(BaseModel):
    rank:     int
    movie_id: int
    title:    str
    genres:   str
    score:    float

class RecommendResponse(BaseModel):
    user_id:         int
    recommendations: list[MovieRecommendation]


app = FastAPI(
    title='Movie Recommender API',
    description='Neural Collaborative Filtering sur MovieLens',
    version='1.0.0'
)

@app.get('/health')
def health():
    return {
        'status':   'ok',
        'model':    'NeuralCF',
        'n_users':  N_USERS,
        'n_movies': N_MOVIES
    }

@app.post('/recommend', response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if req.user_id not in user2idx:
        raise HTTPException(status_code=404, detail=f'user_id {req.user_id} inconnu')
    if not 1 <= req.n <= 50:
        raise HTTPException(status_code=400, detail='n doit être entre 1 et 50')

    uid         = user2idx[req.user_id]
    all_items   = torch.arange(N_MOVIES)
    user_tensor = torch.tensor([uid] * N_MOVIES)

    with torch.no_grad():
        scores = model(user_tensor, all_items).cpu().numpy()

    # Exclut les films déjà vus par cet utilisateur
    seen_movies = set(
        ratings_df[ratings_df['user_idx'] == uid]['movie_idx'].values
    )
    for idx in seen_movies:
        scores[idx] = -np.inf

    top_indices = np.argsort(scores)[::-1][:req.n]

    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        movie_id = idx2movie[int(idx)]
        recommendations.append(MovieRecommendation(
            rank=rank,
            movie_id=movie_id,
            title=movieid2title.get(movie_id, 'Inconnu'),
            genres=movieid2genre.get(movie_id, ''),
            score=float(scores[idx])
        ))

    return RecommendResponse(user_id=req.user_id, recommendations=recommendations)