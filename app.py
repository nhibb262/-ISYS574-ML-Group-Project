"""
SF Events Explorer - Streamlit App
ML-powered event discovery for San Francisco

Run locally: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="SF Events Explorer",
    page_icon="🎉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD DATA & MODEL
# ============================================================================
@st.cache_data
def load_data():
    """Load the cleaned events dataset."""
    # Try multiple possible paths
    possible_paths = [
        'data/processed/events_cleaned.csv',
        'Data/processed/events_cleaned.csv',
        'events_cleaned.csv',
        'data/events_cleaned.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    # If no local file, try to load from URL (fallback)
    st.warning("Local data not found. Loading from source...")
    url = "https://data.sfgov.org/api/views/8i3s-ih2a/rows.csv?accessType=DOWNLOAD"
    df = pd.read_csv(url)
    
    # Basic cleaning
    df['search_text'] = df.apply(
        lambda row: f"{row.get('event_name', '')} {row.get('event_description', '')} {row.get('events_category', '')}",
        axis=1
    )
    return df

@st.cache_resource
def train_model(corpus):
    """Train TF-IDF vectorizer on the corpus."""
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
FEATURE_PATTERNS = {
    'kids': r'\b(kid|kids|child|children|toddler)\b',
    'teens': r'\b(teen|teens|youth|teenager)\b',
    'families': r'\b(family|families)\b',
    'free': r'\b(free)\b',
    'morning': r'\b(morning)\b',
    'afternoon': r'\b(afternoon)\b',
    'evening': r'\b(evening|night)\b',
    'weekend': r'\b(weekend|saturday|sunday)\b'
}

def extract_features(query):
    """Extract structured features from query."""
    query_lower = query.lower()
    features = {}
    for name, pattern in FEATURE_PATTERNS.items():
        features[name] = bool(re.search(pattern, query_lower))
    return features

# ============================================================================
# SEARCH FUNCTION
# ============================================================================
def search_events(query, vectorizer, tfidf_matrix, df, top_k=20, boost_weight=0.15):
    """Search events using TF-IDF + rule-based boosting."""
    
    # TF-IDF similarity
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Extract features
    features = extract_features(query)
    
    # Apply boosts
    boost = np.zeros(len(df))
    
    # Age group boosts
    if features['kids'] and 'age_group_eligibility_tags' in df.columns:
        mask = df['age_group_eligibility_tags'].fillna('').str.contains('Children|Pre-Teens', case=False)
        boost[mask] += boost_weight
    
    if features['teens'] and 'age_group_eligibility_tags' in df.columns:
        mask = df['age_group_eligibility_tags'].fillna('').str.contains('Teens', case=False)
        boost[mask] += boost_weight
    
    if features['families'] and 'age_group_eligibility_tags' in df.columns:
        mask = df['age_group_eligibility_tags'].fillna('').str.contains('Families|Family', case=False)
        boost[mask] += boost_weight
    
    # Free events boost
    if features['free'] and 'fee' in df.columns:
        mask = df['fee'].astype(str).str.lower().isin(['false', 'no', '0', 'nan', ''])
        boost[mask] += boost_weight
    
    # Weekend boost
    if features['weekend'] and 'days_of_week' in df.columns:
        mask = df['days_of_week'].fillna('').str.contains('Sa|Su|Sat|Sun', case=False)
        boost[mask] += boost_weight * 0.5
    
    # Combine scores
    final_scores = scores + boost
    
    # Get top results
    top_idx = final_scores.argsort()[-top_k:][::-1]
    
    results = df.iloc[top_idx].copy()
    results['score'] = final_scores[top_idx]
    
    return results, features

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.title("🎉 SF Events Explorer")
    st.markdown("**ML-powered event discovery for San Francisco**")
    
    # Load data
    with st.spinner("Loading events..."):
        df = load_data()
    
    # Ensure search_text exists
    if 'search_text' not in df.columns:
        df['search_text'] = df.apply(
            lambda row: f"{row.get('event_name', '')} {row.get('event_description', '')} {row.get('events_category', '')}",
            axis=1
        )
    
    # Train model
    corpus = df['search_text'].fillna('').tolist()
    vectorizer, tfidf_matrix = train_model(corpus)
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Category filter
    if 'events_category' in df.columns:
        categories = ['All'] + sorted(df['events_category'].dropna().unique().tolist())
        selected_category = st.sidebar.selectbox("Category", categories)
    else:
        selected_category = 'All'
    
    # Neighborhood filter
    if 'analysis_neighborhood' in df.columns:
        neighborhoods = ['All'] + sorted(df['analysis_neighborhood'].dropna().unique().tolist())
        selected_neighborhood = st.sidebar.selectbox("Neighborhood", neighborhoods)
    else:
        selected_neighborhood = 'All'
    
    # Free only filter
    free_only = st.sidebar.checkbox("Free events only")
    
    # Search box
    st.markdown("---")
    query = st.text_input(
        "🔎 Search for events",
        placeholder="e.g., free art classes for kids, basketball, coding workshop...",
        help="Type naturally! Try: 'fun activities for kids', 'free weekend events', 'music performance'"
    )
    
    # Example queries
    st.markdown("**Try:** `art classes` · `sports for kids` · `free weekend activities` · `coding workshop` · `music`")
    
    # Search results
    if query:
        results, features = search_events(query, vectorizer, tfidf_matrix, df)
        
        # Apply sidebar filters
        if selected_category != 'All' and 'events_category' in results.columns:
            results = results[results['events_category'] == selected_category]
        
        if selected_neighborhood != 'All' and 'analysis_neighborhood' in results.columns:
            results = results[results['analysis_neighborhood'] == selected_neighborhood]
        
        if free_only and 'fee' in results.columns:
            results = results[results['fee'].astype(str).str.lower().isin(['false', 'no', '0', 'nan', ''])]
        
        # Show detected features
        active_features = [k for k, v in features.items() if v]
        if active_features:
            st.info(f"🎯 Detected: **{', '.join(active_features)}** — boosting relevant results")
        
        # Results count
        st.markdown(f"### Found {len(results)} events")
        
        # Display results
        for idx, row in results.head(15).iterrows():
            with st.expander(f"**{row.get('event_name', 'Unnamed Event')}** — Score: {row['score']:.3f}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Description
                    desc = row.get('event_description', 'No description available.')
                    if pd.notna(desc) and len(str(desc)) > 300:
                        desc = str(desc)[:300] + "..."
                    st.write(desc)
                    
                    # More info link
                    if pd.notna(row.get('more_info')):
                        st.markdown(f"[🔗 More Info]({row['more_info']})")
                
                with col2:
                    # Metadata
                    if pd.notna(row.get('events_category')):
                        st.write(f"📁 **Category:** {row['events_category']}")
                    
                    if pd.notna(row.get('age_group_eligibility_tags')):
                        st.write(f"👥 **Ages:** {row['age_group_eligibility_tags']}")
                    
                    if pd.notna(row.get('analysis_neighborhood')):
                        st.write(f"📍 **Location:** {row['analysis_neighborhood']}")
                    
                    if pd.notna(row.get('days_of_week')):
                        st.write(f"📅 **Days:** {row['days_of_week']}")
                    
                    if pd.notna(row.get('start_time')) and pd.notna(row.get('end_time')):
                        st.write(f"🕐 **Time:** {row['start_time']} - {row['end_time']}")
                    
                    # Fee
                    fee_val = str(row.get('fee', '')).lower()
                    if fee_val in ['false', 'no', '0', '', 'nan']:
                        st.write("💰 **Free!**")
                    else:
                        st.write("💰 **Paid event**")
                    
                    # Google Maps link
                    if pd.notna(row.get('site_address')):
                        maps_url = f"https://www.google.com/maps/search/?api=1&query={row['site_address'].replace(' ', '+')}"
                        st.markdown(f"[🗺️ View on Map]({maps_url})")
    
    else:
        # No query - show stats
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", f"{len(df):,}")
        
        if 'events_category' in df.columns:
            col2.metric("Categories", df['events_category'].nunique())
        
        if 'analysis_neighborhood' in df.columns:
            col3.metric("Neighborhoods", df['analysis_neighborhood'].nunique())
        
        # Category breakdown
        if 'events_category' in df.columns:
            st.markdown("### 📁 Events by Category")
            cat_counts = df['events_category'].value_counts()
            st.bar_chart(cat_counts)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        SF Events Explorer | ISYS 574 ML Group Project | 
        Data: <a href='https://data.sfgov.org/Economy-and-Community/Our415-Events-and-Activities/8i3s-ih2a'>Our415 SF Open Data</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
