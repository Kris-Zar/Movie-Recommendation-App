import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import base64
import urllib.request
import urllib.parse
import json

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="MovieRec - AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# TMDB API CONFIGURATION & CLIENT HELPERS
# ==============================================================================
# Safely check for user-provided API key in secrets, otherwise use default working key
try:
    TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8")
except Exception:
    TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

GENRE_MAP = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
    "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
    "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Sci-Fi": 878, "Thriller": 53,
    "War": 10752, "Western": 37
}
ID_TO_GENRE = {v: k for k, v in GENRE_MAP.items()}

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_tmdb_json(endpoint, params=None):
    """Fetch JSON from TMDB API with error handling and caching."""
    if params is None:
        params = {}
    params["api_key"] = TMDB_API_KEY
    url = f"{TMDB_BASE_URL}/{endpoint}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MovieRecApp/2.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=3600)
def search_movies_tmdb(query, max_results=20):
    """Search TMDB's 800,000+ movie catalog by title."""
    if not query:
        return []
    data = fetch_tmdb_json("search/movie", {"query": query, "include_adult": "false"})
    results = data.get("results", [])
    movies = []
    for m in results[:max_results]:
        genre_names = [ID_TO_GENRE.get(gid, "") for gid in m.get("genre_ids", []) if gid in ID_TO_GENRE]
        genre_str = " • ".join(genre_names[:2]) if genre_names else "Cinema"
        poster_path = m.get("poster_path")
        poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=80"
        movies.append({
            "id": m.get("id"),
            "title": m.get("title", "Untitled"),
            "genre": genre_str,
            "rating": str(round(m.get("vote_average", 0.0), 1)),
            "raw_rating": m.get("vote_average", 0.0),
            "genre_ids": m.get("genre_ids", []),
            "poster_url": poster_url,
            "release_date": m.get("release_date", "N/A"),
            "overview": m.get("overview", "")
        })
    return movies

@st.cache_data(show_spinner=False, ttl=3600)
def get_tmdb_deep_similar(movie_id, mode="recommendations", max_items=24, min_rating=0.0, filter_genre="All Genres"):
    """Fetch deep similarity catalog across multiple pages from TMDB."""
    all_movies = []
    seen_ids = {movie_id}
    # Calculate how many TMDB pages needed (each page has up to 20 results)
    pages_to_fetch = min(4, (max_items // 15) + 1)
    
    for page in range(1, pages_to_fetch + 1):
        data = fetch_tmdb_json(f"movie/{movie_id}/{mode}", {"page": page})
        results = data.get("results", [])
        
        # If recommendations is sparse, fallback to similar endpoint
        if not results and mode == "recommendations":
            data = fetch_tmdb_json(f"movie/{movie_id}/similar", {"page": page})
            results = data.get("results", [])
            
        if not results:
            break
            
        for m in results:
            mid = m.get("id")
            if mid in seen_ids or not m.get("poster_path"):
                continue
            seen_ids.add(mid)
            
            vote_avg = m.get("vote_average", 0.0)
            if vote_avg < min_rating:
                continue
                
            genre_names = [ID_TO_GENRE.get(gid, "") for gid in m.get("genre_ids", []) if gid in ID_TO_GENRE]
            if filter_genre != "All Genres" and filter_genre not in genre_names:
                continue
                
            genre_str = " • ".join(genre_names[:2]) if genre_names else "Cinema"
            poster_url = f"{TMDB_IMAGE_BASE}{m.get('poster_path')}"
            
            all_movies.append({
                "id": mid,
                "title": m.get("title", "Untitled"),
                "genre": genre_str,
                "rating": str(round(vote_avg, 1)),
                "raw_rating": vote_avg,
                "poster_url": poster_url,
                "release_date": m.get("release_date", "N/A"),
                "overview": m.get("overview", "")
            })
            if len(all_movies) >= max_items:
                break
        if len(all_movies) >= max_items:
            break
            
    return all_movies

@st.cache_data(show_spinner=False, ttl=3600)
def get_multi_movie_recommendations(movie_ids, top_n=24):
    """Aggregate recommendations from multiple favorite films across TMDB."""
    score_map = {}
    details_map = {}
    
    for mid in movie_ids:
        recs = get_tmdb_deep_similar(mid, mode="recommendations", max_items=20)
        for rank, m in enumerate(recs):
            rec_id = m["id"]
            # Weight higher ranks and multiple co-occurrences
            weight = (20 - rank) + (m.get("raw_rating", 0.0) * 1.5)
            score_map[rec_id] = score_map.get(rec_id, 0.0) + weight
            details_map[rec_id] = m
            
    # Sort by aggregate score
    sorted_ids = sorted(score_map, key=score_map.get, reverse=True)[:top_n]
    return [details_map[i] for i in sorted_ids if i not in movie_ids]

@st.cache_data(show_spinner=False, ttl=3600)
def discover_movies_tmdb(genre_name="All Genres", year_category="All Years", min_rating_str="Any", max_results=12):
    """Query TMDB live database using multi-criteria filters."""
    params = {
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "vote_count.gte": 100
    }
    if genre_name != "All Genres" and genre_name in GENRE_MAP:
        params["with_genres"] = GENRE_MAP[genre_name]
        
    if year_category == "2020s":
        params["primary_release_date.gte"] = "2020-01-01"
    elif year_category == "2010s":
        params["primary_release_date.gte"] = "2010-01-01"
        params["primary_release_date.lte"] = "2019-12-31"
    elif year_category == "2000s":
        params["primary_release_date.gte"] = "2000-01-01"
        params["primary_release_date.lte"] = "2009-12-31"
    elif year_category == "1990s":
        params["primary_release_date.gte"] = "1990-01-01"
        params["primary_release_date.lte"] = "1999-12-31"
    elif year_category == "1980s & Earlier":
        params["primary_release_date.lte"] = "1989-12-31"
        
    if min_rating_str != "Any":
        try:
            params["vote_average.gte"] = float(min_rating_str.split("+")[0])
        except Exception:
            pass
            
    data = fetch_tmdb_json("discover/movie", params)
    results = data.get("results", [])
    movies = []
    for m in results[:max_results]:
        genre_names = [ID_TO_GENRE.get(gid, "") for gid in m.get("genre_ids", []) if gid in ID_TO_GENRE]
        genre_str = " • ".join(genre_names[:2]) if genre_names else "Cinema"
        poster_path = m.get("poster_path")
        poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=80"
        movies.append({
            "id": m.get("id"),
            "title": m.get("title", "Untitled"),
            "genre": genre_str,
            "rating": str(round(m.get("vote_average", 0.0), 1)),
            "poster_url": poster_url,
            "release_date": m.get("release_date", "N/A"),
            "overview": m.get("overview", "")
        })
    return movies

@st.cache_data(show_spinner=False, ttl=86400)
def get_movie_trailer_key(movie_id):
    """Retrieve YouTube trailer embed key for a TMDB film."""
    data = fetch_tmdb_json(f"movie/{movie_id}/videos")
    results = data.get("results", [])
    trailers = [v for v in results if v.get("type") == "Trailer" and v.get("site") == "YouTube"]
    if trailers:
        return trailers[0].get("key")
    videos = [v for v in results if v.get("site") == "YouTube"]
    if videos:
        return videos[0].get("key")
    return None

# ==============================================================================
# LOCAL CACHED ASSETS & ML MODEL LOADERS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "movie_recommender")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
POSTERS_DIR = os.path.join(ASSETS_DIR, "posters")

@st.cache_data
def get_image_base64(filepath):
    """Convert local image to base64 string for zero-latency local embedding."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None
    return None

@st.cache_resource
def load_models_and_data():
    """Load local MovieLens collaborative filtering models and statistics."""
    try:
        movie_info = pd.read_csv(os.path.join(MODEL_DIR, "movie_info.csv"))
        
        def parse_title(t):
            m = re.search(r'\((\d{4})\)', t)
            year = int(m.group(1)) if m else None
            clean = re.sub(r'\s*\(\d{4}\)', '', t).strip()
            if clean.endswith(', The'):
                clean = 'The ' + clean[:-5]
            elif clean.endswith(', A'):
                clean = 'A ' + clean[:-3]
            return pd.Series([clean, year])

        parsed = movie_info['title'].apply(parse_title)
        movie_info['clean_title'] = parsed[0]
        movie_info['year'] = parsed[1]
        
        item_similarity = joblib.load(os.path.join(MODEL_DIR, "item_similarity.pkl"))
        train_matrix = joblib.load(os.path.join(MODEL_DIR, "train_matrix.pkl"))
        global_mean = float(joblib.load(os.path.join(MODEL_DIR, "global_mean.pkl")))
        
        ratings_count = (train_matrix > 0).sum(axis=0)
        ratings_sum = train_matrix.sum(axis=0)
        ratings_mean = (ratings_sum / ratings_count.replace(0, 1)).round(1)
        
        stats_df = pd.DataFrame({
            'movie_id': train_matrix.columns,
            'rating_count': ratings_count.values,
            'avg_rating': ratings_mean.values
        })
        
        movie_info = pd.merge(movie_info, stats_df, on='movie_id', how='left')
        movie_info['avg_rating'] = movie_info['avg_rating'].fillna(round(global_mean, 1))
        movie_info['rating_count'] = movie_info['rating_count'].fillna(0).astype(int)
        
        return movie_info, item_similarity, train_matrix, global_mean
    except Exception as e:
        return pd.DataFrame(), None, None, 3.5

movie_info, item_similarity, train_matrix, global_mean = load_models_and_data()

# ==============================================================================
# CURATED SHOWCASE MOVIES (MATCHING DESIGN SCREENSHOT)
# ==============================================================================
TRENDING_MOVIES_DEFAULT = [
    {"id": 157336, "title": "Interstellar", "genre": "Sci-Fi • Drama", "rating": "8.6", "poster_file": "interstellar.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg"},
    {"id": 27205, "title": "Inception", "genre": "Action • Sci-Fi", "rating": "8.4", "poster_file": "inception.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg"},
    {"id": 155, "title": "The Dark Knight", "genre": "Action • Crime", "rating": "9.0", "poster_file": "dark_knight.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg"},
    {"id": 299534, "title": "Avengers: Endgame", "genre": "Action • Adventure", "rating": "8.4", "poster_file": "endgame.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/bR8ISy1O9XQxqiy0fQFw2BX72RQ.jpg"},
    {"id": 634649, "title": "Spider-Man: No Way Home", "genre": "Action • Adventure", "rating": "8.2", "poster_file": "spiderman.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/1g0dhYtq4irTY1GPXvft6k4YLjm.jpg"},
    {"id": 278, "title": "The Shawshank Redemption", "genre": "Drama", "rating": "9.3", "poster_file": "shawshank.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg"}
]

RECOMMENDED_MOVIES_DEFAULT = [
    {"id": 286217, "title": "The Martian", "genre": "Adventure • Drama", "rating": "8.0", "poster_file": "martian.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/5BHuvQ6p9kfc091Z8RiFNhCwL4b.jpg"},
    {"id": 438631, "title": "Dune", "genre": "Sci-Fi • Adventure", "rating": "8.0", "poster_file": "dune.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/v1tRXZ4JtD2Iv6fjkPvT4GiwslV.jpg"},
    {"id": 335984, "title": "Blade Runner 2049", "genre": "Sci-Fi • Drama", "rating": "8.0", "poster_file": "bladerunner.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg"},
    {"id": 577922, "title": "Tenet", "genre": "Action • Sci-Fi", "rating": "7.3", "poster_file": "tenet.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/rrKH3VhHssGTCfPuaEQkxvEoElv.jpg"},
    {"id": 872585, "title": "Oppenheimer", "genre": "Biography • Drama", "rating": "8.3", "poster_file": "oppenheimer.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg"},
    {"id": 1124, "title": "The Prestige", "genre": "Drama • Mystery", "rating": "8.5", "poster_file": "prestige.jpg", "fallback_url": "https://image.tmdb.org/t/p/w500/Ag2B2KHKQPukjH7WutmgnnSNurZ.jpg"}
]

def get_poster_src(m_dict):
    fname = m_dict.get("poster_file")
    if fname:
        p_path = os.path.join(POSTERS_DIR, fname)
        b64 = get_image_base64(p_path)
        if b64:
            return f"data:image/jpeg;base64,{b64}"
    return m_dict.get("poster_url") or m_dict.get("fallback_url", "")

# Load hero image
hero_opt_path = os.path.join(ASSETS_DIR, "hero_cinema_opt.jpg")
hero_b64 = get_image_base64(hero_opt_path)
hero_img_src = f"data:image/jpeg;base64,{hero_b64}" if hero_b64 else "https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=1000&auto=format&fit=crop&q=80"

# ==============================================================================
# CSS STYLING INJECTION (EXACT TO DESIGN)
# ==============================================================================
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(circle at 80% 10%, #151c33 0%, #0b0e17 60%, #07090f 100%) !important;
    background-attachment: fixed !important;
}

header[data-testid="stHeader"] {
    background: rgba(11, 14, 23, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1460px !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: #090c14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    min-width: 280px !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

.brand-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 8px 20px 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 16px;
}

.brand-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #ff3366, #ff6b8b);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 14px rgba(255, 51, 102, 0.4);
}

.brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #ffffff;
    line-height: 1.15;
}

.brand-subtitle {
    font-size: 0.72rem;
    color: #7d8597;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* Sidebar Radio Navigation */
div[data-testid="stRadio"] {
    margin-bottom: 20px;
    width: 100% !important;
}

div[data-testid="stRadio"] > label {
    display: none !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 6px !important;
    width: 100% !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: all 0.2s ease-in-out !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label input {
    display: none !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] {
    width: 100% !important;
    display: block !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: inherit !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #f1f5f9 !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, #e52d53 0%, #aa1e4e 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 18px rgba(229, 45, 83, 0.45) !important;
    font-weight: 700 !important;
}

/* Sidebar Filters */
.filter-section-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: 0.3px;
    margin-top: 10px;
    margin-bottom: 12px;
}

section[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    color: #7d8597 !important;
    margin-bottom: 4px !important;
}

section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background: #111522 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
}

.filter-btn-container button {
    background: linear-gradient(90deg, #e52d53 0%, #d4274c 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    padding: 10px !important;
    margin-top: 14px !important;
    box-shadow: 0 4px 14px rgba(229, 45, 83, 0.35) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}

.filter-btn-container button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(229, 45, 83, 0.5) !important;
}

/* Hero Section */
.hero-card {
    background: linear-gradient(135deg, rgba(16, 21, 37, 0.95) 0%, rgba(20, 27, 49, 0.92) 50%, rgba(13, 17, 30, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 34px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 30px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.hero-left {
    flex: 1.1;
    z-index: 1;
    padding-right: 24px;
}

.hero-overline {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: #94a3b8;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-heading {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -1px;
    color: #ffffff;
    margin-bottom: 16px;
}

.hero-gradient-text {
    background: linear-gradient(90deg, #ff4071 0%, #ff7395 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block;
}

.hero-desc {
    font-size: 0.96rem;
    line-height: 1.6;
    color: #94a3b8;
    max-width: 520px;
    margin-bottom: 24px;
}

.hero-btn-row {
    display: flex;
    gap: 14px;
    align-items: center;
    flex-wrap: nowrap;
}

.hero-btn-primary {
    background: linear-gradient(90deg, #e52d53 0%, #c82046 100%) !important;
    color: #ffffff !important;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.92rem;
    text-decoration: none !important;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap !important;
    box-shadow: 0 6px 18px rgba(229, 45, 83, 0.4);
    transition: all 0.2s;
    cursor: pointer;
    border: none;
}

.hero-btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(229, 45, 83, 0.6);
}

.hero-btn-secondary {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #f1f5f9 !important;
    padding: 12px 22px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.92rem;
    text-decoration: none !important;
    display: inline-flex;
    align-items: center;
    white-space: nowrap !important;
    transition: all 0.2s;
    cursor: pointer;
}

.hero-btn-secondary:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    transform: translateY(-2px);
}

.hero-right {
    flex: 0.95;
    z-index: 1;
    display: flex;
    justify-content: flex-end;
}

.hero-img {
    width: 100%;
    max-width: 460px;
    height: 250px;
    object-fit: cover;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6);
}

/* Section Header */
.section-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-top: 14px;
    margin-bottom: 16px;
}

.section-title-wrap {
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.section-title {
    font-size: 1.18rem;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-subtitle {
    font-size: 0.8rem;
    color: #7d8597;
    font-weight: 500;
}

.section-link {
    font-size: 0.82rem;
    color: #94a3b8;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s;
    cursor: pointer;
}

.section-link:hover {
    color: #ff4071;
}

/* Movie Cards Grid */
.movie-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 16px;
    margin-bottom: 30px;
}

@media (max-width: 1200px) {
    .movie-grid {
        grid-template-columns: repeat(3, 1fr);
    }
}

@media (max-width: 768px) {
    .movie-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

.movie-card {
    background: #111625;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    overflow: hidden;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    display: flex;
    flex-direction: column;
    position: relative;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

.movie-card:hover {
    transform: translateY(-7px);
    border-color: rgba(229, 45, 83, 0.35);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6), 0 0 15px rgba(229, 45, 83, 0.2);
}

.card-img-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 2 / 2.75;
    overflow: hidden;
    background: #182033;
}

.card-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.3s ease;
}

.movie-card:hover .card-img {
    transform: scale(1.05);
}

.card-details {
    padding: 11px 12px 13px 12px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    background: #111625;
}

.card-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #ffffff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.25;
}

.card-genre {
    font-size: 0.74rem;
    color: #7d8597;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.card-rating-row {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-top: 2px;
}

.star-icon {
    color: #f59e0b;
    font-size: 0.8rem;
}

.card-rating {
    font-size: 0.82rem;
    font-weight: 700;
    color: #f1f5f9;
}

.metric-box {
    background: #111625;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-number {
    font-size: 2rem;
    font-weight: 800;
    color: #ff4071;
}

.metric-label {
    font-size: 0.82rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""")

# ==============================================================================
# SIDEBAR NAVIGATION & FILTER CONTROLS
# ==============================================================================
with st.sidebar:
    st.html("""
    <div class="brand-container">
        <div class="brand-icon">🎬</div>
        <div>
            <div class="brand-title">MovieRec</div>
            <div class="brand-subtitle">AI Movie Recommendation System</div>
        </div>
    </div>
    """)
    
    nav_options = [
        "🏠 Home",
        "⭐ Get Recommendations",
        "🔍 Search Movies",
        "👥 Similar Movies",
        "📊 Movie Insights",
        "ℹ️ About"
    ]
    
    # Allow programmatic tab switching via session state
    if "selected_nav" not in st.session_state:
        st.session_state.selected_nav = nav_options[0]
        
    selected_page = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state.selected_nav) if st.session_state.selected_nav in nav_options else 0,
        label_visibility="collapsed",
        key="sidebar_nav"
    )
    st.session_state.selected_nav = selected_page
    
    st.markdown("---")
    
    # Dynamic TMDB & Local Filters
    st.html('<div class="filter-section-title">Filters (Optional)</div>')
    
    genre_choices = ["All Genres"] + list(GENRE_MAP.keys())
    selected_genre = st.selectbox("Genre", options=genre_choices, index=0)
    
    year_choices = ["All Years", "2020s", "2010s", "2000s", "1990s", "1980s & Earlier"]
    selected_year = st.selectbox("Release Year", options=year_choices, index=0)
    
    rating_choices = ["Any", "9.0+ ⭐", "8.5+ ⭐", "8.0+ ⭐", "7.5+ ⭐", "7.0+ ⭐"]
    selected_rating = st.selectbox("Minimum Rating", options=rating_choices, index=0)
    
    st.html('<div class="filter-btn-container">')
    filter_applied = st.button("🎯 Apply Filters", use_container_width=True)
    st.html('</div>')
    
    # Store filter state
    if filter_applied:
        st.session_state.active_filter = {
            "genre": selected_genre,
            "year": selected_year,
            "rating": selected_rating
        }
        st.toast("Filters applied across TMDB catalog!", icon="🎬")

# Helper to render a movie card
def render_movie_card_html(title, genre, rating, img_src):
    return (
        f'<div class="movie-card">'
        f'<div class="card-img-wrap">'
        f'<img src="{img_src}" class="card-img" alt="{title}" loading="lazy"/>'
        f'</div>'
        f'<div class="card-details">'
        f'<div class="card-title" title="{title}">{title}</div>'
        f'<div class="card-genre">{genre}</div>'
        f'<div class="card-rating-row">'
        f'<span class="star-icon">★</span>'
        f'<span class="card-rating">{rating}</span>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

# ==============================================================================
# PAGE 1: HOME (EXACT MATCH TO DESIGN SCREENSHOT + LIVE TMDB FILTERS)
# ==============================================================================
if "Home" in selected_page:
    st.html(f"""
    <div class="hero-card">
        <div class="hero-left">
            <div class="hero-overline">DISCOVER • EXPLORE • ENJOY</div>
            <div class="hero-heading">
                Find Your Next
                <span class="hero-gradient-text">Favorite Movie</span>
            </div>
            <div class="hero-desc">
                Get personalized movie recommendations using machine learning and the global TMDB library based on your preferences, viewing history, and similar users.
            </div>
            <div class="hero-btn-row">
                <a href="#trending" class="hero-btn-primary">
                    <span>🪄</span> Get Recommendations →
                </a>
                <a href="#trending" class="hero-btn-secondary">
                    Explore Popular Movies
                </a>
            </div>
        </div>
        <div class="hero-right">
            <img src="{hero_img_src}" class="hero-img" alt="Cinema Popcorn and Slate"/>
        </div>
    </div>
    """)
    
    # 2. Trending Movies Section
    st.html("""
    <div id="trending" class="section-header">
        <div class="section-title-wrap">
            <div class="section-title">
                <span>🔥</span> Trending Movies
            </div>
            <div class="section-subtitle">Popular movies recommended by our system</div>
        </div>
        <div class="section-link">View All →</div>
    </div>
    """)
    
    # Check if user applied custom filters
    has_filter = (
        selected_genre != "All Genres" or 
        selected_year != "All Years" or 
        selected_rating != "Any"
    )
    
    if has_filter:
        # Fetch live filtered movies from TMDB
        filtered_movies = discover_movies_tmdb(selected_genre, selected_year, selected_rating, max_results=12)
        if filtered_movies:
            trending_display = filtered_movies[:6]
            recommended_display = filtered_movies[6:12] if len(filtered_movies) > 6 else filtered_movies[:6]
        else:
            trending_display = TRENDING_MOVIES_DEFAULT
            recommended_display = RECOMMENDED_MOVIES_DEFAULT
    else:
        trending_display = TRENDING_MOVIES_DEFAULT
        recommended_display = RECOMMENDED_MOVIES_DEFAULT
        
    trending_cards_html = "".join([
        render_movie_card_html(m["title"], m["genre"], m["rating"], get_poster_src(m))
        for m in trending_display
    ])
    st.html(f'<div class="movie-grid">{trending_cards_html}</div>')
    
    # 3. Recommended For You Section
    st.html("""
    <div class="section-header">
        <div class="section-title-wrap">
            <div class="section-title">
                <span>✨</span> Recommended For You
            </div>
            <div class="section-subtitle">Based on your selected preferences</div>
        </div>
        <div class="section-link">View All →</div>
    </div>
    """)
    
    recommended_cards_html = "".join([
        render_movie_card_html(m["title"], m["genre"], m["rating"], get_poster_src(m))
        for m in recommended_display
    ])
    st.html(f'<div class="movie-grid">{recommended_cards_html}</div>')

# ==============================================================================
# PAGE 2: GET RECOMMENDATIONS (GLOBAL TMDB + COLLABORATIVE FILTERING)
# ==============================================================================
elif "Get Recommendations" in selected_page:
    st.html("""
    <div class="section-header" style="margin-top: 0px;">
        <div class="section-title-wrap">
            <div class="section-title"><span>⭐</span> AI Movie Recommendation Engine</div>
            <div class="section-subtitle">Recommend from over 800,000 films globally via TMDB & Machine Learning models</div>
        </div>
    </div>
    """)
    
    engine_choice = st.radio(
        "Choose Recommendation Engine:",
        ["🌍 Global TMDB Engine (All 800,000+ Movies)", "🔬 Machine Learning Lab (MovieLens 100K Dataset)"],
        horizontal=True
    )
    
    if "Global TMDB" in engine_choice:
        rec_tab1, rec_tab2 = st.tabs(["🎯 Single Movie Inspiration", "🎬 Multi-Movie Taste Profile"])
        
        with rec_tab1:
            col_search, col_depth = st.columns([3, 1])
            with col_search:
                movie_search_input = st.text_input(
                    "🎬 Search any movie in TMDB's 800,000+ catalog:",
                    value="Inception",
                    placeholder="e.g. Dune: Part Two, Oppenheimer, Interstellar, The Dark Knight, Spirited Away..."
                )
            with col_depth:
                top_k_tmdb = st.select_slider("Catalogue Depth:", options=[12, 18, 24, 30, 36, 48], value=24)
                
            if movie_search_input:
                search_candidates = search_movies_tmdb(movie_search_input, max_results=8)
                if search_candidates:
                    selected_candidate_title = st.selectbox(
                        "Select matching film:",
                        [f"{c['title']} ({c['release_date'][:4] if c['release_date'] else 'N/A'}) - ⭐ {c['rating']}" for c in search_candidates]
                    )
                    selected_idx = [f"{c['title']} ({c['release_date'][:4] if c['release_date'] else 'N/A'}) - ⭐ {c['rating']}" for c in search_candidates].index(selected_candidate_title)
                    target_movie = search_candidates[selected_idx]
                    
                    # Benchmark movie banner preview
                    with st.expander("📌 Selected Film Details & Synopsis", expanded=False):
                        c_col1, c_col2 = st.columns([1, 4])
                        with c_col1:
                            st.image(target_movie["poster_url"], width=130)
                        with c_col2:
                            st.markdown(f"### **{target_movie['title']}** ({target_movie['release_date'][:4]})")
                            st.markdown(f"**Genres:** {target_movie['genre']} | **Rating:** ⭐ {target_movie['rating']}/10")
                            st.markdown(f"*{target_movie['overview']}*")
                            trailer_key = get_movie_trailer_key(target_movie["id"])
                            if trailer_key:
                                st.markdown(f"[▶ Watch Official Trailer on YouTube](https://www.youtube.com/watch?v={trailer_key})")
                    
                    col_b1, col_b2 = st.columns([2, 1])
                    with col_b1:
                        sim_algo = st.radio(
                            "Similarity Algorithm:",
                            ["Viewer Taste & Affinity (TMDB Recommendations)", "Storyline, Plot & Keyword Match (TMDB Similar)"],
                            horizontal=True
                        )
                    with col_b2:
                        min_rating_filter = st.selectbox("Min Rating:", ["Any", "6.5+ ⭐", "7.0+ ⭐", "7.5+ ⭐", "8.0+ ⭐"], index=0)
                        
                    min_r_val = 0.0 if min_rating_filter == "Any" else float(min_rating_filter.split("+")[0])
                    mode_key = "recommendations" if "Viewer Taste" in sim_algo else "similar"
                    
                    if st.button("Generate Deep AI Recommendations 🚀", type="primary", use_container_width=True):
                        with st.spinner(f"Scanning TMDB deep catalog for '{target_movie['title']}'..."):
                            recs = get_tmdb_deep_similar(target_movie["id"], mode=mode_key, max_items=top_k_tmdb, min_rating=min_r_val)
                            if recs:
                                st.markdown(f"### Found **{len(recs)}** recommended films for fans of **{target_movie['title']}**:")
                                cards_html = "".join([
                                    render_movie_card_html(m["title"], m["genre"], m["rating"], m["poster_url"])
                                    for m in recs
                                ])
                                st.html(f'<div class="movie-grid">{cards_html}</div>')
                            else:
                                st.info("No matching films found with the chosen rating threshold. Try lowering the minimum rating.")
                else:
                    st.warning(f"No movie found matching '{movie_search_input}'. Please try another search title.")
                    
        with rec_tab2:
            st.markdown("#### Combine Multiple Films for an Aggregated Taste Profile")
            st.caption("Add 2 or 3 movies you love to find films that blend their themes and audience patterns.")
            
            multi_col1, multi_col2 = st.columns(2)
            with multi_col1:
                movie1_name = st.text_input("First Favorite Film:", value="Interstellar")
            with multi_col2:
                movie2_name = st.text_input("Second Favorite Film:", value="Blade Runner 2049")
                
            multi_top_k = st.select_slider("Results to discover:", options=[12, 18, 24, 36], value=24, key="multi_slider")
            
            if st.button("Generate Blended Recommendations 🪄", type="primary", use_container_width=True):
                cand1 = search_movies_tmdb(movie1_name, max_results=1)
                cand2 = search_movies_tmdb(movie2_name, max_results=1)
                
                target_ids = []
                names = []
                if cand1:
                    target_ids.append(cand1[0]["id"])
                    names.append(cand1[0]["title"])
                if cand2:
                    target_ids.append(cand2[0]["id"])
                    names.append(cand2[0]["title"])
                    
                if target_ids:
                    with st.spinner(f"Cross-referencing TMDB recommendation graphs for {' and '.join(names)}..."):
                        blended_recs = get_multi_movie_recommendations(target_ids, top_n=multi_top_k)
                        if blended_recs:
                            st.markdown(f"### Blended Recommendations for fans of **{' & '.join(names)}**:")
                            cards_html = "".join([
                                render_movie_card_html(m["title"], m["genre"], m["rating"], m["poster_url"])
                                for m in blended_recs
                            ])
                            st.html(f'<div class="movie-grid">{cards_html}</div>')
                        else:
                            st.info("No shared recommendations found.")
                else:
                    st.warning("Please enter valid movie titles.")

    else:
        # MovieLens 100K Machine Learning Engine
        st.markdown("#### 🔬 Item-Item Collaborative Filtering (Trained on 100K User Ratings)")
        if not movie_info.empty and item_similarity is not None:
            all_titles = movie_info['title'].tolist()
            default_idx = 0
            for i, t in enumerate(all_titles):
                if "Toy Story (1995)" in t:
                    default_idx = i
                    break
                    
            selected_local_title = st.selectbox("Select a classic benchmark movie:", all_titles, index=default_idx)
            top_k_ml = st.slider("Number of Recommendations:", min_value=6, max_value=24, value=12, step=6)
            
            if st.button("Run Collaborative Filtering 🚀", type="primary", use_container_width=True):
                row = movie_info[movie_info['title'] == selected_local_title]
                if not row.empty:
                    mid = row.iloc[0]['movie_id']
                    if mid in item_similarity.index:
                        sim_scores = item_similarity.loc[mid].sort_values(ascending=False).drop(mid, errors='ignore')
                        top_ids = sim_scores.head(top_k_ml).index.tolist()
                        
                        st.markdown(f"### ML Model Output for **{selected_local_title}**:")
                        rec_movies = movie_info[movie_info['movie_id'].isin(top_ids)].copy()
                        
                        cards_html = []
                        for _, rec_row in rec_movies.iterrows():
                            clean_t = rec_row['clean_title']
                            rating = rec_row['avg_rating']
                            genre_tag = f"Released: {int(rec_row['year'])}" if pd.notnull(rec_row['year']) else "Film"
                            
                            live_meta = search_movies_tmdb(clean_t, max_results=1)
                            poster_url = live_meta[0]["poster_url"] if live_meta else "https://via.placeholder.com/300x450/111625/e52d53?text=" + urllib.parse.quote(clean_t)
                            if live_meta:
                                rating = live_meta[0]["rating"]
                                genre_tag = live_meta[0]["genre"]
                                
                            cards_html.append(render_movie_card_html(clean_t, genre_tag, str(rating), poster_url))
                            
                        st.html(f'<div class="movie-grid">{cards_html}</div>')
        else:
            st.warning("Local MovieLens model files not loaded.")

# ==============================================================================
# PAGE 3: SEARCH MOVIES (ALL 800,000+ TMDB MOVIES)
# ==============================================================================
elif "Search Movies" in selected_page:
    st.html("""
    <div class="section-header" style="margin-top: 0px;">
        <div class="section-title-wrap">
            <div class="section-title"><span>🔍</span> Explore Global Movie Library</div>
            <div class="section-subtitle">Live search across 800,000+ films with instant high-res artwork & ratings</div>
        </div>
    </div>
    """)
    
    col_s1, col_s2 = st.columns([4, 1])
    with col_s1:
        search_query = st.text_input("Enter movie title, franchise, or character:", value="Spider-Man")
    with col_s2:
        search_limit = st.selectbox("Results count:", [12, 18, 24, 36], index=1)
        
    if search_query:
        with st.spinner("Searching TMDB catalog..."):
            found_movies = search_movies_tmdb(search_query, max_results=search_limit)
            if found_movies:
                st.markdown(f"Found **{len(found_movies)}** titles matching *'{search_query}'*:")
                cards_html = "".join([
                    render_movie_card_html(m["title"], f"{m['release_date'][:4] if m['release_date'] else ''} • {m['genre']}", m["rating"], m["poster_url"])
                    for m in found_movies
                ])
                st.html(f'<div class="movie-grid">{cards_html}</div>')
            else:
                st.info(f"No results found for '{search_query}'. Try another search.")

# ==============================================================================
# PAGE 4: SIMILAR MOVIES (FULL TMDB CATALOGUE SEARCH & DEEP RETRIEVAL)
# ==============================================================================
elif "Similar Movies" in selected_page:
    st.html("""
    <div class="section-header" style="margin-top: 0px;">
        <div class="section-title-wrap">
            <div class="section-title"><span>👥</span> Deep Film Similarity Explorer</div>
            <div class="section-subtitle">Explore hundreds of matching movies across TMDB's full 800,000+ global library</div>
        </div>
    </div>
    """)
    
    col_sim_search, col_sim_limit = st.columns([3, 1])
    with col_sim_search:
        target_search = st.text_input(
            "Search any benchmark movie in the world:",
            value="The Dark Knight",
            placeholder="e.g. Dune, Interstellar, Parasite, Avengers, Pulp Fiction, Spirited Away..."
        )
    with col_sim_limit:
        similarity_depth = st.select_slider("Explore Catalog Depth:", options=[12, 18, 24, 30, 36, 48], value=24)
        
    if target_search:
        sim_candidates = search_movies_tmdb(target_search, max_results=6)
        if sim_candidates:
            chosen_candidate_str = st.selectbox(
                "Select exact movie:",
                [f"{c['title']} ({c['release_date'][:4] if c['release_date'] else 'N/A'}) - ⭐ {c['rating']}" for c in sim_candidates]
            )
            c_idx = [f"{c['title']} ({c['release_date'][:4] if c['release_date'] else 'N/A'}) - ⭐ {c['rating']}" for c in sim_candidates].index(chosen_candidate_str)
            active_benchmark = sim_candidates[c_idx]
            
            # Interactive Filter Bar for Similarity Catalog
            col_filter_mode, col_filter_genre, col_filter_rate = st.columns([2, 1, 1])
            with col_filter_mode:
                sim_mode = st.radio(
                    "Similarity Metric:",
                    ["Storyline & Keyword Similarity (TMDB Similar)", "Co-Viewer Taste & Affinity (TMDB Recommendations)"],
                    horizontal=True
                )
            with col_filter_genre:
                genre_filter = st.selectbox("Genre Filter:", ["All Genres"] + list(GENRE_MAP.keys()), index=0)
            with col_filter_rate:
                rate_filter = st.selectbox("Rating Filter:", ["Any", "6.5+ ⭐", "7.0+ ⭐", "7.5+ ⭐", "8.0+ ⭐"], index=0)
                
            min_r_val = 0.0 if rate_filter == "Any" else float(rate_filter.split("+")[0])
            mode_api_param = "similar" if "Storyline" in sim_mode else "recommendations"
            
            if st.button("Explore Full Similarity Catalogue 🚀", type="primary", use_container_width=True):
                with st.spinner(f"Extracting deep similarity network for '{active_benchmark['title']}' from TMDB..."):
                    deep_results = get_tmdb_deep_similar(
                        active_benchmark["id"],
                        mode=mode_api_param,
                        max_items=similarity_depth,
                        min_rating=min_r_val,
                        filter_genre=genre_filter
                    )
                    if deep_results:
                        st.markdown(f"### Found **{len(deep_results)}** similar films for **{active_benchmark['title']}**:")
                        cards_html = "".join([
                            render_movie_card_html(m["title"], m["genre"], m["rating"], m["poster_url"])
                            for m in deep_results
                        ])
                        st.html(f'<div class="movie-grid">{cards_html}</div>')
                    else:
                        st.info("No matching films found with the active filter. Try resetting filters.")
        else:
            st.warning(f"No movie found matching '{target_search}'.")

# ==============================================================================
# PAGE 5: MOVIE INSIGHTS
# ==============================================================================
elif "Movie Insights" in selected_page:
    st.html("""
    <div class="section-header" style="margin-top: 0px;">
        <div class="section-title-wrap">
            <div class="section-title"><span>📊</span> Dataset Metrics & Analytics</div>
            <div class="section-subtitle">System intelligence from TMDB & Collaborative Filtering matrix</div>
        </div>
    </div>
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.html("""
        <div class="metric-box">
            <div class="metric-number">800,000+</div>
            <div class="metric-label">TMDB Live Films</div>
        </div>
        """)
    with col2:
        st.html("""
        <div class="metric-box">
            <div class="metric-number">100,000</div>
            <div class="metric-label">ML Ratings Trained</div>
        </div>
        """)
    with col3:
        st.html(f"""
        <div class="metric-box">
            <div class="metric-number">{round(global_mean, 2)} ⭐</div>
            <div class="metric-label">ML Global Baseline</div>
        </div>
        """)
    with col4:
        st.html("""
        <div class="metric-box">
            <div class="metric-number">1,682</div>
            <div class="metric-label">MovieLens Items</div>
        </div>
        """)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 🔥 Most Reviewed Classic Films")
        if not movie_info.empty:
            top_rated = movie_info.sort_values(by='rating_count', ascending=False)[['clean_title', 'rating_count', 'avg_rating']].head(10)
            top_rated.columns = ['Title', 'Total Reviews', 'Avg Rating ⭐']
            st.dataframe(top_rated, use_container_width=True)
            
    with col_right:
        st.markdown("#### 🏆 Highest Rated Classics (Min. 50 Reviews)")
        if not movie_info.empty:
            critics = movie_info[movie_info['rating_count'] >= 50].sort_values(by='avg_rating', ascending=False)[['clean_title', 'avg_rating', 'rating_count']].head(10)
            critics.columns = ['Title', 'Avg Rating ⭐', 'Total Reviews']
            st.dataframe(critics, use_container_width=True)

# ==============================================================================
# PAGE 6: ABOUT
# ==============================================================================
elif "About" in selected_page:
    st.html("""
    <div class="section-header" style="margin-top: 0px;">
        <div class="section-title-wrap">
            <div class="section-title"><span>ℹ️</span> About MovieRec AI</div>
            <div class="section-subtitle">Architectural overview and system specifications</div>
        </div>
    </div>
    
    <div style="background: #111625; padding: 26px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); line-height: 1.8;">
        <h3 style="color: #ff4071; margin-top: 0;">Dual Recommendation Architecture</h3>
        <p>MovieRec combines the best of both modern web-scale APIs and custom machine learning algorithms:</p>
        <ul>
            <li><strong>TMDB Global Catalog Integration:</strong> Connects to The Movie Database REST API to access over 800,000 films with real-time recommendations, high-resolution CDN posters, user ratings, and genre filtering.</li>
            <li><strong>Item-Item Collaborative Filtering (Machine Learning):</strong> Employs precomputed cosine similarity matrices trained on the MovieLens 100K benchmark dataset (943 users × 1,653 items) to find mathematical neighbors in user preference space.</li>
            <li><strong>Zero-Latency Local Caching:</strong> All showcase artwork, posters, and base64 assets load instantly without network bottlenecks.</li>
            <li><strong>Responsive UI:</strong> Built with Streamlit, custom CSS Grid, and vibrant cinematic gradients matching modern dark streaming platforms.</li>
        </ul>
    </div>
    """)
