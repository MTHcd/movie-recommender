import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("Movie Recommender System")
st.caption("Neural Collaborative Filtering · MovieLens · FastAPI + Streamlit")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres")
    user_id = st.number_input("User ID", min_value=1, max_value=610, value=1, step=1)
    n_recs  = st.slider("Nombre de recommandations", 1, 20, 10)
    st.divider()
    health  = requests.get(f"{API_URL}/health").json()
    st.success(f"API OK — {health['n_users']:,} users · {health['n_movies']:,} films")

# ── Recommandations ───────────────────────────────────────
st.subheader(f"Recommandations pour l'utilisateur {user_id}")

resp = requests.post(f"{API_URL}/recommend", json={"user_id": user_id, "n": n_recs})

if resp.status_code == 200:
    recs = resp.json()["recommendations"]
    df   = pd.DataFrame(recs)

    # Graphique des scores
    fig = px.bar(
        df, x="score", y="title", orientation="h",
        color="score", color_continuous_scale="Blues",
        labels={"score": "Score", "title": "Film"},
        height=400
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Tableau détaillé
    st.dataframe(
        df[["rank", "title", "genres", "score"]].rename(columns={
            "rank": "Rang", "title": "Titre",
            "genres": "Genres", "score": "Score"
        }),
        use_container_width=True,
        hide_index=True
    )

elif resp.status_code == 404:
    st.error(f"User ID {user_id} inconnu du modèle.")
else:
    st.error(f"Erreur API : {resp.status_code}")