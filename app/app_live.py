"""
NLP Product Review Analysis Web Application

A comprehensive dashboard showcasing:
1. Sentiment Classification (RoBERTa)
2. Product Clustering (Zero-shot classification)
3. Review Summarization (BART/T5)

Deploy with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import Counter
import re
import os
import json
from typing import Dict, List, Any

# NLP Libraries
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("Transformers library not available. Some features may be limited.")

# Set page configuration
st.set_page_config(
    page_title="NLP Product Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load or create sample data for demonstration."""
    try:
        # Try to load from uploaded file or GitHub repository
        if os.path.exists("data/processed/reranker_products.csv"):
            df = pd.read_csv("data/processed/reranker_products.csv")
            return df
        elif os.path.exists("reranker_products.csv"):
            df = pd.read_csv("reranker_products.csv")
            return df
        else:
            # Create sample data for demonstration
            st.info("Using sample data for demonstration. Upload your own CSV file for analysis.")
            return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration purposes."""
    np.random.seed(42)
    
    products = [
        "Fire TV Stick", "Echo Dot", "Kindle Paperwhite", "Fire Tablet", 
        "Ring Doorbell", "Echo Show", "Fire TV Cube", "Kindle Oasis",
        "Echo Buds", "Fire HD 10"
    ]
    
    categories = [
        "Fire Amazon", "Echo White Amazon", "Electronics & Technology", 
        "Fire Tablet Special", "Fire Kids Edition"
    ]
    
    reviews = [
        "Great product, works perfectly!", "Amazing quality and fast delivery",
        "Not what I expected, could be better", "Excellent value for money",
        "Poor build quality", "Outstanding performance", "Easy to use and setup",
        "Battery life is disappointing", "Perfect for my needs", "Highly recommended!",
        "Good but not great", "Exceeded my expectations", "Terrible customer service",
        "Love this device!", "Would buy again", "Not worth the price",
        "Simple and effective", "Great features", "Had some issues initially",
        "Fantastic product overall"
    ]
    
    data = []
    for _ in range(1000):
        product = np.random.choice(products)
        category = np.random.choice(categories)
        review = np.random.choice(reviews)
        rating = np.random.randint(1, 6)
        sentiment = np.random.uniform(0, 1)
        recommend = np.random.choice([True, False], p=[0.7, 0.3])
        
        data.append({
            'name': product,
            'text': review,
            'rating': rating,
            'zero_shot_label': category,
            'zero_shot_score': np.random.uniform(0.6, 0.95),
            'predicted_sentiment_roberta': sentiment,
            'doRecommend': recommend
        })
    
    return pd.DataFrame(data)

@st.cache_resource
def load_nlp_models():
    """Load NLP models with caching and error handling."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        # Use lighter models for Streamlit Cloud
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1  # Force CPU usage for Streamlit Cloud
        )
        
        # Use a lighter summarization model
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",  # Lighter BART model
            device=-1,
            max_length=100,
            min_length=30
        )
        
        return sentiment_classifier, summarizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Running in demo mode without live NLP models.")
        return None, None

def analyze_sentiment_live(text: str, classifier) -> Dict[str, Any]:
    """Analyze sentiment of input text using RoBERTa."""
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0.5}
    
    if classifier is None:
        # Fallback demo sentiment analysis
        words = text.lower().split()
        positive_words = ['good', 'great', 'excellent', 'amazing', 'perfect', 'love', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'disappointing', 'poor', 'worst']
        
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        if pos_count > neg_count:
            return {"label": "POSITIVE", "score": 0.8, "confidence": "80%"}
        elif neg_count > pos_count:
            return {"label": "NEGATIVE", "score": 0.2, "confidence": "80%"}
        else:
            return {"label": "NEUTRAL", "score": 0.5, "confidence": "50%"}
    
    try:
        result = classifier(text[:512])[0]  # Limit text length
        return {
            "label": result["label"],
            "score": result["score"],
            "confidence": f"{result['score']:.1%}"
        }
    except Exception as e:
        st.error(f"Sentiment analysis error: {str(e)}")
        return {"label": "ERROR", "score": 0.0}

def classify_product_category(text: str) -> Dict[str, Any]:
    """Classify product category using keyword-based classification."""
    text_lower = text.lower()
    
    category_keywords = {
        "Fire Amazon": ["fire", "tv", "stick", "streaming", "video", "entertainment"],
        "Echo White Amazon": ["echo", "alexa", "voice", "smart", "speaker", "assistant"],
        "Electronics & Technology": ["phone", "computer", "tablet", "camera", "headphones", "electronic", "tech"],
        "Fire Tablet Special": ["tablet", "ipad", "screen", "touch", "portable", "reading"],
        "Fire Kids Edition": ["kids", "children", "child", "family", "educational", "parental"]
    }
    
    scores = {}
    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        scores[category] = score / len(keywords)
    
    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]
    
    return {
        "category": best_category,
        "confidence": max(0.3, min(0.95, best_score + 0.3)),
        "all_scores": scores
    }

def generate_category_summary(df: pd.DataFrame, category: str, summarizer) -> str:
    """Generate summary for a specific category."""
    category_data = df[df['zero_shot_label'] == category]
    
    if len(category_data) == 0:
        return f"No data available for {category} category."
    
    # Get top products
    top_products = category_data.groupby('name').agg({
        'rating': ['mean', 'count'],
        'zero_shot_score': 'mean'
    }).round(2)
    top_products.columns = ['avg_rating', 'review_count', 'confidence']
    top_products = top_products[top_products['review_count'] >= 3].nlargest(3, 'avg_rating')
    
    summary_text = f"**{category} Category Analysis**\n\n"
    summary_text += f"Total Products: {category_data['name'].nunique()}\n"
    summary_text += f"Average Rating: {category_data['rating'].mean():.1f}/5.0\n"
    summary_text += f"Total Reviews: {len(category_data)}\n\n"
    
    if len(top_products) > 0:
        summary_text += "**Top Rated Products:**\n"
        for i, (product_name, stats) in enumerate(top_products.iterrows(), 1):
            summary_text += f"{i}. {product_name} - {stats['avg_rating']:.1f}/5.0 ({int(stats['review_count'])} reviews)\n"
    
    return summary_text

def create_sentiment_distribution_chart(df: pd.DataFrame, category: str = None):
    """Create sentiment distribution visualization."""
    if category:
        data = df[df['zero_shot_label'] == category]
        title = f"Sentiment Distribution - {category}"
    else:
        data = df
        title = "Overall Sentiment Distribution"
    
    # Convert sentiment scores to labels
    data_copy = data.copy()
    data_copy['sentiment_label'] = data_copy['predicted_sentiment_roberta'].apply(
        lambda x: 'Positive' if x > 0.6 else 'Negative' if x < 0.4 else 'Neutral'
    )
    
    sentiment_counts = data_copy['sentiment_label'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title=title,
        color_discrete_map={
            'Positive': '#2E8B57',
            'Neutral': '#FFD700', 
            'Negative': '#DC143C'
        }
    )
    
    return fig

def create_rating_distribution_chart(df: pd.DataFrame):
    """Create rating distribution by category."""
    rating_by_category = df.groupby(['zero_shot_label', 'rating']).size().reset_index(name='count')
    
    fig = px.bar(
        rating_by_category,
        x='zero_shot_label',
        y='count',
        color='rating',
        title="Rating Distribution by Category",
        labels={'zero_shot_label': 'Category', 'count': 'Number of Reviews'},
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_cluster_overview(df: pd.DataFrame):
    """Create cluster overview visualization."""
    cluster_stats = df.groupby('zero_shot_label').agg({
        'name': 'nunique',
        'rating': 'mean',
        'predicted_sentiment_roberta': 'mean',
        'zero_shot_score': 'mean'
    }).round(3)
    
    cluster_stats.columns = ['Unique_Products', 'Avg_Rating', 'Avg_Sentiment', 'Avg_Confidence']
    cluster_stats = cluster_stats.reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Products per Category', 'Average Rating', 'Average Sentiment', 'Classification Confidence'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Products per category
    fig.add_trace(
        go.Bar(x=cluster_stats['zero_shot_label'], y=cluster_stats['Unique_Products'], name='Products'),
        row=1, col=1
    )
    
    # Average rating
    fig.add_trace(
        go.Bar(x=cluster_stats['zero_shot_label'], y=cluster_stats['Avg_Rating'], name='Rating'),
        row=1, col=2
    )
    
    # Average sentiment
    fig.add_trace(
        go.Bar(x=cluster_stats['zero_shot_label'], y=cluster_stats['Avg_Sentiment'], name='Sentiment'),
        row=2, col=1
    )
    
    # Classification confidence
    fig.add_trace(
        go.Bar(x=cluster_stats['zero_shot_label'], y=cluster_stats['Avg_Confidence'], name='Confidence'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Category Analysis Dashboard")
    fig.update_xaxes(tickangle=-45)
    
    return fig

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">NLP Product Review Analysis Platform</h1>', unsafe_allow_html=True)
    
    # Load data and models
    df = load_sample_data()
    if df is None:
        st.error("Unable to load data. Please check your data file.")
        st.stop()
    
    sentiment_classifier, summarizer = load_nlp_models()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Dashboard Overview", "Live Text Analysis", "Category Deep Dive", "Product Search", "Upload & Analyze"]
    )
    
    if page == "Dashboard Overview":
        show_dashboard_overview(df)
    elif page == "Live Text Analysis":
        show_live_analysis(sentiment_classifier, summarizer)
    elif page == "Category Deep Dive":
        show_category_analysis(df, summarizer)
    elif page == "Product Search":
        show_product_search(df)
    elif page == "Upload & Analyze":
        show_upload_analysis(sentiment_classifier, summarizer)

def show_dashboard_overview(df: pd.DataFrame):
    """Display main dashboard with overview statistics."""
    st.subheader("Platform Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    with col2:
        st.metric("Unique Products", f"{df['name'].nunique():,}")
    with col3:
        st.metric("Categories", f"{df['zero_shot_label'].nunique()}")
    with col4:
        st.metric("Avg Rating", f"{df['rating'].mean():.1f}/5.0")
    
    st.markdown("---")
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = create_sentiment_distribution_chart(df)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = create_rating_distribution_chart(df)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Cluster overview
    st.subheader("Category Analysis")
    fig3 = create_cluster_overview(df)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Category breakdown table
    st.subheader("Category Statistics")
    category_stats = df.groupby('zero_shot_label').agg({
        'name': 'nunique',
        'rating': ['mean', 'count'],
        'predicted_sentiment_roberta': 'mean',
        'doRecommend': 'mean'
    }).round(3)
    
    category_stats.columns = ['Products', 'Avg_Rating', 'Review_Count', 'Avg_Sentiment', 'Recommend_Rate']
    st.dataframe(category_stats, use_container_width=True)

def show_live_analysis(sentiment_classifier, summarizer):
    """Show live text analysis interface."""
    st.subheader("Live Text Analysis")
    st.write("Test the NLP models with your own text input")
    
    # Text input
    user_text = st.text_area(
        "Enter product review or description:",
        placeholder="Example: This tablet has an amazing display and great battery life. Perfect for reading and streaming videos.",
        height=150
    )
    
    if st.button("Analyze Text", type="primary"):
        if user_text.strip():
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Classification")
                sentiment_result = analyze_sentiment_live(user_text, sentiment_classifier)
                
                st.success(f"Sentiment: **{sentiment_result['label']}**")
                st.info(f"Confidence: **{sentiment_result.get('confidence', 'N/A')}**")
                
                # Sentiment gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = sentiment_result['score'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sentiment Score"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Category Classification")
                category_result = classify_product_category(user_text)
                
                st.success(f"Category: **{category_result['category']}**")
                st.info(f"Confidence: **{category_result['confidence']:.1%}**")
                
                # Category scores
                if 'all_scores' in category_result:
                    scores_df = pd.DataFrame(
                        list(category_result['all_scores'].items()),
                        columns=['Category', 'Score']
                    ).sort_values('Score', ascending=True)
                    
                    fig = px.bar(
                        scores_df,
                        x='Score',
                        y='Category',
                        orientation='h',
                        title="Category Classification Scores"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Text summarization
            st.subheader("Text Summary")
            if summarizer and len(user_text) > 50:
                try:
                    with st.spinner("Generating summary..."):
                        summary = summarizer(user_text, max_length=60, min_length=20)
                        st.write(f"**Summary:** {summary[0]['summary_text']}")
                except Exception as e:
                    st.write("Text too short for meaningful summarization or model unavailable.")
            else:
                st.write("Enter longer text (50+ characters) for AI summarization, or running in demo mode.")
        else:
            st.warning("Please enter some text to analyze.")

def show_category_analysis(df: pd.DataFrame, summarizer):
    """Show detailed category analysis."""
    st.subheader("Category Deep Dive")
    
    # Category selection
    categories = df['zero_shot_label'].unique()
    selected_category = st.selectbox("Select Category for Analysis:", categories)
    
    if selected_category:
        category_data = df[df['zero_shot_label'] == selected_category]
        
        # Category metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Products", category_data['name'].nunique())
        with col2:
            st.metric("Reviews", len(category_data))
        with col3:
            st.metric("Avg Rating", f"{category_data['rating'].mean():.1f}/5.0")
        with col4:
            st.metric("Positive Reviews", f"{(category_data['predicted_sentiment_roberta'] > 0.6).mean():.1%}")
        
        st.markdown("---")
        
        # Category summary
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Category Summary")
            summary = generate_category_summary(df, selected_category, summarizer)
            st.markdown(summary)
        
        with col2:
            st.subheader("Sentiment Distribution")
            fig = create_sentiment_distribution_chart(df, selected_category)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top products table
        st.subheader("Top Products in Category")
        top_products = category_data.groupby('name').agg({
            'rating': ['mean', 'count'],
            'predicted_sentiment_roberta': 'mean',
            'doRecommend': 'mean'
        }).round(3)
        
        top_products.columns = ['Avg_Rating', 'Review_Count', 'Sentiment', 'Recommend_Rate']
        top_products = top_products[top_products['Review_Count'] >= 3].nlargest(10, 'Avg_Rating')
        
        st.dataframe(top_products, use_container_width=True)

def show_product_search(df: pd.DataFrame):
    """Show product search functionality."""
    st.subheader("Product Search & Analysis")
    
    # Search interface
    search_query = st.text_input("Search for products:", placeholder="e.g., kindle, tablet, phone")
    
    if search_query:
        # Filter products based on search
        mask = df['name'].str.contains(search_query, case=False, na=False)
        search_results = df[mask]
        
        if len(search_results) > 0:
            st.success(f"Found {len(search_results)} reviews for products matching '{search_query}'")
            
            # Search results summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Products", search_results['name'].nunique())
            with col2:
                st.metric("Average Rating", f"{search_results['rating'].mean():.1f}/5.0")
            with col3:
                st.metric("Most Common Category", search_results['zero_shot_label'].mode().iloc[0])
            
            # Product breakdown
            st.subheader("Search Results")
            product_summary = search_results.groupby('name').agg({
                'rating': ['mean', 'count'],
                'predicted_sentiment_roberta': 'mean',
                'zero_shot_label': lambda x: x.mode().iloc[0]
            }).round(3)
            
            product_summary.columns = ['Avg_Rating', 'Review_Count', 'Sentiment', 'Category']
            product_summary = product_summary.sort_values('Avg_Rating', ascending=False)
            
            st.dataframe(product_summary, use_container_width=True)
            
            # Visualization
            fig = px.scatter(
                product_summary.reset_index(),
                x='Avg_Rating',
                y='Sentiment',
                size='Review_Count',
                color='Category',
                hover_name='name',
                title=f"Product Performance: {search_query}",
                labels={'Avg_Rating': 'Average Rating', 'Sentiment': 'Sentiment Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning(f"No products found matching '{search_query}'. Try different keywords.")

def show_upload_analysis(sentiment_classifier, summarizer):
    """Show file upload and analysis interface."""
    st.subheader("Upload & Analyze Custom Dataset")
    st.write("Upload your own CSV file with product reviews for analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            upload_df = pd.read_csv(uploaded_file)
            
            st.success(f"File uploaded successfully! Shape: {upload_df.shape}")
            st.subheader("Data Preview")
            st.dataframe(upload_df.head(), use_container_width=True)
            
            # Column mapping
            st.subheader("Column Mapping")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                text_column = st.selectbox("Review Text Column:", upload_df.columns)
            with col2:
                rating_column = st.selectbox("Rating Column:", [None] + list(upload_df.columns))
            with col3:
                product_column = st.selectbox("Product Name Column:", [None] + list(upload_df.columns))
            
            if st.button("Analyze Uploaded Data", type="primary"):
                with st.spinner("Processing your data..."):
                    analyze_uploaded_data(upload_df, text_column, rating_column, product_column, sentiment_classifier)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def analyze_uploaded_data(df: pd.DataFrame, text_col: str, rating_col: str, product_col: str, sentiment_classifier):
    """Analyze uploaded dataset."""
    
    # Basic statistics
    st.subheader("Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", len(df))
    with col2:
        if product_col:
            st.metric("Unique Products", df[product_col].nunique())
    with col3:
        if rating_col and df[rating_col].dtype in ['int64', 'float64']:
            st.metric("Average Rating", f"{df[rating_col].mean():.1f}")
    
    # Sentiment analysis on sample
    st.subheader("Sentiment Analysis Sample")
    sample_size = min(100, len(df))
    sample_df = df.sample(sample_size)
    
    sentiments = []
    progress_bar = st.progress(0)
    
    for i, text in enumerate(sample_df[text_col].astype(str)):
        if pd.notna(text) and text.strip():
            result = analyze_sentiment_live(text, sentiment_classifier)
            sentiments.append(result['label'])
        else:
            sentiments.append('UNKNOWN')
        
        progress_bar.progress((i + 1) / sample_size)
    
    # Display results
    sentiment_counts = Counter(sentiments)
    
    fig = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title=f"Sentiment Distribution (Sample of {sample_size} reviews)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show sample results
    sample_results = sample_df.copy()
    sample_results['Predicted_Sentiment'] = sentiments
    st.subheader("Sample Results")
    st.dataframe(sample_results[[text_col, 'Predicted_Sentiment']].head(10), use_container_width=True)

if __name__ == "__main__":
    main()