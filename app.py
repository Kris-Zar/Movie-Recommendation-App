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
# PATHS & CACHED ASSETS
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
    """Load collaborative filtering models, matrices, and metadata."""
    movie_info = pd.read_csv(os.path.join(MODEL_DIR, "movie_info.csv"))
    
    # Parse release year and clean title
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
    
    # Compute ratings statistics
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

movie_info, item_similarity, train_matrix, global_mean = load_models_and_data()

# Movie poster lookup helper
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_movie_poster_tmdb(query_title):
    """Fetch poster URL and details from TMDB with fallback."""
    try:
        clean = query_title.split('(')[0].strip()
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(clean)}"
        req = urllib.request.Request(url, headers={"User-Agent": "MovieRec/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if data.get('results'):
                res = data['results'][0]
                poster_path = res.get('poster_path')
                vote_avg = res.get('vote_average', 0.0)
                overview = res.get('overview', '')
                if poster_path:
                    return {
                        'poster': f"https://image.tmdb.org/t/p/w500{poster_path}",
                        'rating': round(vote_avg, 1),
                        'overview': overview
                    }
    except Exception:
        pass
    return None

# ==============================================================================
# CURATED SHOWCASE MOVIES (EXACT MATCH TO USER SCREENSHOT)
# ==============================================================================
TRENDING_MOVIES = [
    {
        "id": "interstellar",
        "title": "Interstellar",
        "genre": "Sci-Fi • Drama",
        "rating": "8.6",
        "poster_file": "interstellar.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg"
    },
    {
        "id": "inception",
        "title": "Inception",
        "genre": "Action • Sci-Fi",
        "rating": "8.4",
        "poster_file": "inception.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg"
    },
    {
        "id": "dark_knight",
        "title": "The Dark Knight",
        "genre": "Action • Crime",
        "rating": "9.0",
        "poster_file": "dark_knight.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg"
    },
    {
        "id": "endgame",
        "title": "Avengers: Endgame",
        "genre": "Action • Adventure",
        "rating": "8.4",
        "poster_file": "endgame.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/bR8ISy1O9XQxqiy0fQFw2BX72RQ.jpg"
    },
    {
        "id": "spiderman",
        "title": "Spider-Man: No Way Home",
        "genre": "Action • Adventure",
        "rating": "8.2",
        "poster_file": "spiderman.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/1g0dhYtq4irTY1GPXvft6k4YLjm.jpg"
    },
    {
        "id": "shawshank",
        "title": "The Shawshank Redemption",
        "genre": "Drama",
        "rating": "9.3",
        "poster_file": "shawshank.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg"
    }
]

RECOMMENDED_MOVIES = [
    {
        "id": "martian",
        "title": "The Martian",
        "genre": "Adventure • Drama",
        "rating": "8.0",
        "poster_file": "martian.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/5BHuvQ6p9kfc091Z8RiFNhCwL4b.jpg"
    },
    {
        "id": "dune",
        "title": "Dune",
        "genre": "Sci-Fi • Adventure",
        "rating": "8.0",
        "poster_file": "dune.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/v1tRXZ4JtD2Iv6fjkPvT4GiwslV.jpg"
    },
    {
        "id": "bladerunner",
        "title": "Blade Runner 2049",
        "genre": "Sci-Fi • Drama",
        "rating": "8.0",
        "poster_file": "bladerunner.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg"
    },
    {
        "id": "tenet",
        "title": "Tenet",
        "genre": "Action • Sci-Fi",
        "rating": "7.3",
        "poster_file": "tenet.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/rrKH3VhHssGTCfPuaEQkxvEoElv.jpg"
    },
    {
        "id": "oppenheimer",
        "title": "Oppenheimer",
        "genre": "Biography • Drama",
        "rating": "8.3",
        "poster_file": "oppenheimer.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg"
    },
    {
        "id": "prestige",
        "title": "The Prestige",
        "genre": "Drama • Mystery",
        "rating": "8.5",
        "poster_file": "prestige.jpg",
        "fallback_url": "https://image.tmdb.org/t/p/w500/Ag2B2KHKQPukjH7WutmgnnSNurZ.jpg"
    }
]

def get_poster_src(movie_dict):
    fname = movie_dict.get("poster_file")
    if fname:
        p_path = os.path.join(POSTERS_DIR, fname)
        b64 = get_image_base64(p_path)
        if b64:
            return f"data:image/jpeg;base64,{b64}"
    return movie_dict.get("fallback_url", "")

# Load hero image
hero_opt_path = os.path.join(ASSETS_DIR, "hero_cinema_opt.jpg")
hero_b64 = get_image_base64(hero_opt_path)
hero_img_src = f"data:image/jpeg;base64,{hero_b64}" if hero_b64 else "https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=1000&auto=format&fit=crop&q=80"

# ==============================================================================
# CSS STYLING INJECTION
# ==============================================================================
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

/* Global Font & App Canvas */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(circle at 80% 10%, #151c33 0%, #0b0e17 60%, #07090f 100%) !important;
    background-attachment: fixed !important;
}

/* Streamlit Header */
header[data-testid="stHeader"] {
    background: rgba(11, 14, 23, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Main Container */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1460px !important;
}

/* =====================================================
   SIDEBAR STYLING
   ===================================================== */
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

/* Brand Header */
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

/* Navigation Radio Styling */
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

/* Active navigation item with exact glowing pink/red gradient */
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

/* Filter Submit Button */
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

/* =====================================================
   HERO BANNER
   ===================================================== */
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

/* =====================================================
   SECTION HEADERS
   ===================================================== */
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

/* =====================================================
   MOVIE CARDS 6-COLUMN GRID
   ===================================================== */
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

/* Analytics Metrics */
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
# SIDEBAR
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
    
    selected_page = st.radio(
        "Navigation",
        options=nav_options,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Filter Section
    st.html('<div class="filter-section-title">Filters (Optional)</div>')
    
    genre_choices = [
        "All Genres", "Action", "Adventure", "Animation", "Biography",
        "Comedy", "Crime", "Drama", "Fantasy", "Mystery", "Romance", "Sci-Fi", "Thriller"
    ]
    selected_genre = st.selectbox("Genre", options=genre_choices, index=0)
    
    year_choices = ["All Years", "2020s", "2010s", "2000s", "1990s", "1980s & Earlier"]
    selected_year = st.selectbox("Release Year", options=year_choices, index=0)
    
    rating_choices = ["Any", "9.0+ ⭐", "8.5+ ⭐", "8.0+ ⭐", "7.5+ ⭐", "7.0+ ⭐"]
    selected_rating = st.selectbox("Minimum Rating", options=rating_choices, index=0)
    
    st.html('<div class="filter-btn-container">')
    filter_submitted = st.button("🎯 Apply Filters", use_container_width=True)
    st.html('</div>')

# Helper to render a single card
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
# PAGE 1: HOME (EXACT MATCH TO DESIGN SCREENSHOT)
# ==============================================================================
if "Home" in selected_page:
    # 1. Hero Section
    st.html(f"""
    <div class="hero-card">
        <div class="hero-left">
            <div class="hero-overline">DISCOVER • EXPLORE • ENJOY</div>
            <div class="hero-heading">
                Find Your Next
                <span class="hero-gradient-text">Favorite Movie</span>
            </div>
            <div class="hero-desc">
                Get personalized movie recommendations using machine learning based on your preferences, viewing history, and similar users.
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
    
    # Filter trending cards if filter was selected
    trending_display = []
    min_r = 0.0
    if selected_rating != "Any":
        min_r = float(selected_rating.split("+")[0])
        
    for m in TRENDING_MOVIES:
        if float(m["rating"]) >= min_r:
            if selected_genre == "All Genres" or selected_genre.lower() in m["genre"].lower():
                trending_display.append(m)
                
    if not trending_display:
        trending_display = TRENDING_MOVIES
        
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
    
    recommended_display = []
    for m in RECOMMENDED_MOVIES:
        if float(m["rating"]) >= min_r:
            if selected_genre == "All Genres" or selected_genre.lower() in m["genre"].lower():
                recommended_display.append(m)
                
    if not recommended_display:
        recommended_display = RECOMMENDED_MOVIES
        
    recommended_cards_html = "".join([
        render_movie_card_html(m["title"], m["genre"], m["rating"], get_poster_src(m))
        for m in recommended_display
    ])
    
    st.html(f'<div class="movie-grid">{recommended_cards_html}</div>')

# ==============================================================================
# PAGE 2: GET RECOMMENDATIONS
# ==============================================================================
elif "Get Recommendations" in selected_page:
    st.html("""
    <div class="section-header" style="margin-top: 0px;">
        <div class="section-title-wrap">
            <div class="section-title"><span>⭐</span> AI Movie Recommendation Engine</div>
            <div class="section-subtitle">Powered by Collaborative Filtering & Cosine Similarity</div>
        </div>
    </div>
    """)
    
    rec_type = st.radio(
        "Recommendation Type",
        ["Item-Based Collaborative Filtering (Based on a Movie you like)", "User-Based Collaborative Filtering (Profile ID)"],
        horizontal=True
    )
    
    if "Item-Based" in rec_type:
        all_titles = movie_info['title'].tolist()
        default_idx = 0
        for i, t in enumerate(all_titles):
            if "Toy Story (1995)" in t:
                default_idx = i
                break
                
        selected_title = st.selectbox("🎬 Select a Movie You Enjoyed:", all_titles, index=default_idx)
        top_k = st.slider("Number of Recommendations", min_value=6, max_value=18, value=6, step=6)
        
        if st.button("Generate AI Recommendations 🚀", type="primary"):
            row = movie_info[movie_info['title'] == selected_title]
            if not row.empty:
                mid = row.iloc[0]['movie_id']
                if mid in item_similarity.index:
                    sim_scores = item_similarity.loc[mid].sort_values(ascending=False).drop(mid, errors='ignore')
                    top_ids = sim_scores.head(top_k).index.tolist()
                    
                    st.markdown(f"### Movies Similar to **{selected_title}**")
                    rec_movies = movie_info[movie_info['movie_id'].isin(top_ids)].copy()
                    
                    cards_html = []
                    for idx, (_, rec_row) in enumerate(rec_movies.iterrows()):
                        clean_t = rec_row['clean_title']
                        rating = rec_row['avg_rating']
                        genre_tag = f"Released: {int(rec_row['year'])}" if pd.notnull(rec_row['year']) else "Cinema Classic"
                        
                        meta = fetch_movie_poster_tmdb(rec_row['title'])
                        poster_url = meta['poster'] if meta else "https://via.placeholder.com/300x450/111625/e52d53?text=" + urllib.parse.quote(clean_t)
                        if meta and meta.get('rating'):
                            rating = meta['rating']
                            
                        cards_html.append(render_movie_card_html(clean_t, genre_tag, str(rating), poster_url))
                        
                    st.html(f'<div class="movie-grid">{"".join(cards_html)}</div>')
                else:
                    st.warning("This movie does not have enough ratings in the training matrix to compute cosine similarity.")
    else:
        user_id = st.number_input("Enter User ID (1 to 943):", min_value=1, max_value=943, value=1)
        top_k = st.slider("Number of Recommendations", min_value=6, max_value=18, value=6, step=6)
        
        if st.button("Predict Top Recommendations 🚀", type="primary"):
            if user_id in train_matrix.index:
                user_ratings = train_matrix.loc[user_id]
                unrated = user_ratings[user_ratings == 0].index
                rated = user_ratings[user_ratings > 0]
                
                preds = {}
                for mid in unrated[:150]:
                    if mid in item_similarity.index:
                        sims = item_similarity.loc[mid, rated.index]
                        if sims.sum() > 0:
                            pred = (sims * rated).sum() / sims.sum()
                            preds[mid] = pred
                            
                top_pred_ids = sorted(preds, key=preds.get, reverse=True)[:top_k]
                rec_movies = movie_info[movie_info['movie_id'].isin(top_pred_ids)]
                
                st.markdown(f"### Top Personalized Picks for **User #{user_id}**")
                cards_html = []
                for _, rec_row in rec_movies.iterrows():
                    clean_t = rec_row['clean_title']
                    rating = rec_row['avg_rating']
                    genre_tag = f"Predicted ⭐ {round(preds.get(rec_row['movie_id'], rating), 1)}"
                    meta = fetch_movie_poster_tmdb(rec_row['title'])
                    poster_url = meta['poster'] if meta else "https://via.placeholder.com/300x450/111625/e52d53?text=" + urllib.parse.quote(clean_t)
                    cards_html.append(render_movie_card_html(clean_t, genre_tag, str(rating), poster_url))
                    
                st.html(f'<div class="movie-grid">{"".join(cards_html)}</div>')

# ==============================================================================
# PAGE 3: SEARCH MOVIES
# ==============================================================================
elif "Search Movies" in selected_page:
    st.html("""
    <div class="section-header" style="margin-top: 0px;">
        <div class="section-title-wrap">
            <div class="section-title"><span>🔍</span> Explore Movie Library</div>
            <div class="section-subtitle">Search over 1,680 films in the dataset with instant posters & ratings</div>
        </div>
    </div>
    """)
    
    search_query = st.text_input("Type movie title or keyword...", value="Star Wars")
    
    if search_query:
        matches = movie_info[movie_info['clean_title'].str.contains(search_query, case=False, na=False)].head(12)
        if not matches.empty:
            cards_html = []
            for _, m in matches.iterrows():
                clean_t = m['clean_title']
                rating = m['avg_rating']
                genre_tag = f"Release: {int(m['year'])}" if pd.notnull(m['year']) else "Film"
                meta = fetch_movie_poster_tmdb(m['title'])
                poster_url = meta['poster'] if meta else "https://via.placeholder.com/300x450/111625/e52d53?text=" + urllib.parse.quote(clean_t)
                cards_html.append(render_movie_card_html(clean_t, genre_tag, str(rating), poster_url))
                
            st.html(f'<div class="movie-grid">{"".join(cards_html)}</div>')
        else:
            st.info("No matching movies found. Try searching for 'Matrix', 'Toy Story', 'Godfather', or 'Batman'.")

# ==============================================================================
# PAGE 4: SIMILAR MOVIES
# ==============================================================================
elif "Similar Movies" in selected_page:
    st.html("""
    <div class="section-header" style="margin-top: 0px;">
        <div class="section-title-wrap">
            <div class="section-title"><span>👥</span> Item-to-Item Similarity Explorer</div>
            <div class="section-subtitle">Explore movies that share similar viewer taste profiles</div>
        </div>
    </div>
    """)
    
    sample_popular = movie_info.sort_values(by='rating_count', ascending=False)['title'].head(100).tolist()
    picked_movie = st.selectbox("Select a benchmark movie:", sample_popular)
    
    row = movie_info[movie_info['title'] == picked_movie]
    if not row.empty:
        mid = row.iloc[0]['movie_id']
        if mid in item_similarity.index:
            sims = item_similarity.loc[mid].sort_values(ascending=False).drop(mid, errors='ignore').head(12)
            sim_df = movie_info[movie_info['movie_id'].isin(sims.index)].copy()
            
            cards_html = []
            for _, m in sim_df.iterrows():
                score = round(sims.get(m['movie_id'], 0.0) * 100, 1)
                meta = fetch_movie_poster_tmdb(m['title'])
                poster_url = meta['poster'] if meta else "https://via.placeholder.com/300x450/111625/e52d53?text=" + urllib.parse.quote(m['clean_title'])
                cards_html.append(render_movie_card_html(m['clean_title'], f"Similarity: {score}%", str(m['avg_rating']), poster_url))
                
            st.html(f'<div class="movie-grid">{"".join(cards_html)}</div>')

# ==============================================================================
# PAGE 5: MOVIE INSIGHTS
# ==============================================================================
elif "Movie Insights" in selected_page:
    st.html("""
    <div class="section-header" style="margin-top: 0px;">
        <div class="section-title-wrap">
            <div class="section-title"><span>📊</span> Dataset Metrics & Analytics</div>
            <div class="section-subtitle">Real-time statistics from the collaborative filtering models</div>
        </div>
    </div>
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.html("""
        <div class="metric-box">
            <div class="metric-number">1,682</div>
            <div class="metric-label">Total Movies</div>
        </div>
        """)
    with col2:
        st.html("""
        <div class="metric-box">
            <div class="metric-number">943</div>
            <div class="metric-label">Active Users</div>
        </div>
        """)
    with col3:
        st.html(f"""
        <div class="metric-box">
            <div class="metric-number">{round(global_mean, 2)} ⭐</div>
            <div class="metric-label">Global Mean Rating</div>
        </div>
        """)
    with col4:
        st.html("""
        <div class="metric-box">
            <div class="metric-number">100,000+</div>
            <div class="metric-label">Training Ratings</div>
        </div>
        """)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 🔥 Most Reviewed Movies")
        top_rated = movie_info.sort_values(by='rating_count', ascending=False)[['clean_title', 'rating_count', 'avg_rating']].head(10)
        top_rated.columns = ['Title', 'Total Reviews', 'Avg Rating ⭐']
        st.dataframe(top_rated, use_container_width=True)
        
    with col_right:
        st.markdown("#### 🏆 Highest Rated Classics (Min. 50 Reviews)")
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
        <h3 style="color: #ff4071; margin-top: 0;">Collaborative Filtering Architecture</h3>
        <p>MovieRec is an AI-powered movie recommendation system designed to provide highly accurate and personalized film suggestions using machine learning techniques:</p>
        <ul>
            <li><strong>Item-Item Collaborative Filtering:</strong> Analyzes patterns of user preferences to compute pairwise cosine similarities between films in a high-dimensional rating space.</li>
            <li><strong>Sparsity Handling:</strong> Utilizes pre-computed similarity matrices and global baseline estimates to mitigate cold-start and sparse user vectors.</li>
            <li><strong>Dynamic Poster Resolution:</strong> Real-time metadata and high-definition movie art integration via TMDB API with local cached assets.</li>
            <li><strong>Dataset:</strong> Built on the MovieLens 100K benchmark dataset comprising 100,000 ratings from 943 users across 1,682 movies.</li>
        </ul>
    </div>
    """)
