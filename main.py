import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from textblob import TextBlob  # For sentiment analysis
import numpy as np
import streamlit.components.v1 as components
import os

# --- Configuration ---
# IMPORTANT: Replace 'YOUR_GA_MEASUREMENT_ID' with your actual Google Analytics 4 (GA4) Measurement ID.
# It typically starts with "G-".
# You can also get this from an environment variable for better security and deployment practices.
GA_MEASUREMENT_ID = os.environ.get("GA_MEASUREMENT_ID", "YOUR_GA_MEASUREMENT_ID")

if GA_MEASUREMENT_ID == "YOUR_GA_MEASUREMENT_ID":
    st.warning("Please replace 'YOUR_GA_MEASUREMENT_ID' with your actual Google Analytics 4 Measurement ID.")
    st.info("You can also set it as an environment variable named `GA_MEASUREMENT_ID`.")


def inject_google_analytics():
    """
    Injects the Google Analytics 4 (GA4) tracking code into the Streamlit app
    using st.components.v1.html().
    The height is set to 0 to make the component invisible.
    """
    if GA_MEASUREMENT_ID == "YOUR_GA_MEASUREMENT_ID" or not GA_MEASUREMENT_ID:
        # Don't inject if the ID is not set or is the placeholder
        return

    ga_code = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());

      gtag('config', '{GA_MEASUREMENT_ID}');
    </script>
    """
    # Use components.html to inject the raw HTML.
    # Setting height=0 makes it invisible, as it's just a script.
    components.html(ga_code, height=0)
    st.success(f"Google Analytics (ID: {GA_MEASUREMENT_ID}) injected successfully!")


# --- Your Streamlit App ---

# Call the injection function at the very beginning of your app script.
# This ensures it runs whenever the app is loaded or re-executed.
inject_google_analytics()


# Optional imports for advanced features
try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Set page config
st.set_page_config(page_title="Twitter Analytics Dashboard", layout="wide", page_icon="📊")

st.title("My Streamlit App")
st.write("Welcome to my app!")
# ... your app content ...

# Main content
st.title("📊 Twitter Analytics Dashboard")
st.markdown("*Transform your Twitter data into actionable insights*")

st.markdown("Made by [Felipe Gabriel](https://x.com/FelG_research), if you want to contribute, send me an [email](felipe.g.datascience@gmail.com) or a DM on my [LinkedIn](https://www.linkedin.com/in/felipe-gabriel0/)")

# Tutorial Section
st.header("💼 How to use this dashboard?")

# Expandable tutorial section
with st.expander("📚 **Quick Start Guide - Click to expand**", expanded=True):
    st.markdown("""
    ### **Must have Premium/Professional Features!**
    
    **Access Advanced Analytics:**
    
    * Go to **Premium** section in sidebar → **Analytics**
    * More detailed metrics available
    * Longer historical data retention
    
    #### **📥 Enhanced Export Options**
    * **Longer date ranges** (up to 1 year on overview)
    * **More detailed metrics** per tweet (on content)
    * **Video performance** data (being developed)
    
    ---
    
    ### **📋 Step-by-Step Data Download**
    
    #### **Step 1: Go to Premium Section** 
    * Click on **"Premium"** in your Twitter sidebar
    
    #### **Step 2: Click Analytics**
    * In Premium section, click **"Analytics"**
    
    #### **Step 3: Download Overview Data**
    * Go to **"Overview"** tab
    * Select **"1Y"** (1 year) time interval  
    * Click the **download button** (📥) in top-right corner
    
    #### **Step 4: Download Content Data**
    * Go to **"Content"** tab
    * Select **"1Y"** (1 year) time interval
    * Click the **download button** (📥) in top-right corner
    
    ---
    
    ### **📂 Required Files for Dashboard**
    
    **You'll need 2 CSV files:**
    
    1. **`account_analytics_content_*.csv`** (Tweet-level data)
       * Columns: Date, Tweet text, Impressions, Likes, Engagements, etc.
    
    2. **`account_overview_analytics_*.csv`** (Daily summary data)  
       * Columns: Date, Total impressions, Total engagements, Followers, etc.
    
    **Expected Column Names:**
    * **Portuguese:** `Data`, `Texto do post`, `Impressões`, `Curtidas`, `Engajamentos`
    * **English:** `Date`, `Post text`, `Impressions`, `Likes`, `Engagements`
    """)
    
    # Success message
    st.success("🎉 **Once you have both CSV files, upload them below to start analyzing your Twitter performance!**")

st.divider()

# File uploaders section
st.header("📁 Upload Your Twitter Data")

tweets_sheet = st.file_uploader("📊 Upload your **account_analytics_content** data CSV", type=["csv"], help="This file contains individual tweet performance data")

account_analytics = st.file_uploader("📈 Upload your **account_overview_analytics** performance CSV", type=["csv"], help="This file contains daily account summary data")

# Preprocess data
if account_analytics is not None:
    account_analytics= pd.read_csv(account_analytics)

    account_analytics['Date'] = pd.to_datetime(account_analytics['Date'], format='mixed')
    account_analytics = account_analytics.sort_values(by='Date')

    # Handle both English and Portuguese column names
    if 'Novos seguidores' in account_analytics.columns and 'Deixar de seguir' in account_analytics.columns:
        account_analytics['followers'] = (account_analytics['Novos seguidores'] - account_analytics['Deixar de seguir']).cumsum()
    elif 'New follows' in account_analytics.columns and 'Unfollows' in account_analytics.columns:
        account_analytics['followers'] = (account_analytics['New follows'] - account_analytics['Unfollows']).cumsum()

if tweets_sheet is not None:
    tweets_sheet = pd.read_csv(tweets_sheet)

    # Handle both English and Portuguese column names for date
    if 'Data' in tweets_sheet.columns:
        tweets_sheet['Date'] = pd.to_datetime(tweets_sheet['Data'], format='mixed')
        tweets_sheet['tweet_length'] = tweets_sheet['Texto do post'].apply(lambda x: len(str(x)))
    elif 'Date' in tweets_sheet.columns:
        tweets_sheet['Date'] = pd.to_datetime(tweets_sheet['Date'], format='mixed')
        tweets_sheet['tweet_length'] = tweets_sheet['Post text'].apply(lambda x: len(str(x)))

    tweets_sheet = tweets_sheet.sort_values(by='Date')

if (account_analytics is not None) and (tweets_sheet is not None) :
        
    # Main content
    st.title("Twitter Analytics Dashboard")

    # Account Overview Section
    st.header("Account Overview")

    # Key metrics cards - handle both languages
    col1, col2, col3, col4 = st.columns(4)
    
    # Determine column names based on language
    if 'Curtidas' in tweets_sheet.columns:
        likes_col = 'Curtidas'
        impressions_col = 'Impressões'
        engagements_col = 'Engajamentos'
        reposts_col = 'Compartilhamentos'
        replies_col = 'Respostas'
        bookmarks_col = 'Itens salvos'
        post_text_col = 'Texto do post'
    else:
        likes_col = 'Likes'
        impressions_col = 'Impressions'
        engagements_col = 'Engagements'
        reposts_col = 'Reposts'
        replies_col = 'Replies'
        bookmarks_col = 'Bookmarks'
        post_text_col = 'Post text'
    
    with col1:
        total_tweets = len(tweets_sheet)
        st.metric("Total Tweets", total_tweets)
    
    with col2:
        total_likes = tweets_sheet[likes_col].sum()
        st.metric("Total Likes", f"{total_likes:,}")
    
    with col3:
        total_impressions = tweets_sheet[impressions_col].sum()
        st.metric("Total Impressions", f"{total_impressions:,}")
    
    with col4:
        avg_engagement_rate = tweets_sheet[engagements_col].sum() / tweets_sheet[impressions_col].sum()
        st.metric("Avg Engagement Rate", f"{avg_engagement_rate:.2%}")

    # Follower growth chart
    fig_followers = go.Figure()

    fig_followers.add_trace(go.Scatter(x=account_analytics['Date'], y=account_analytics['followers'],
                                    mode='lines', name='Followers'))
    fig_followers.update_layout(title="Follower Growth Over Time",
                            xaxis_title="Date", yaxis_title="Followers")
    st.plotly_chart(fig_followers, use_container_width=True)

    # Correlation heatmap using Plotly - handle both languages
    if 'Impressões' in account_analytics.columns:
        corr_cols = ['Impressões', 'Curtidas', 'Engajamentos', 'followers']
    else:
        corr_cols = ['Impressions', 'Likes', 'Engagements', 'followers']
    
    corr_matrix = account_analytics[corr_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        labels=dict(x="Metrics", y="Metrics", color="Correlation"),
                        x=corr_matrix.columns, y=corr_matrix.columns)
    st.plotly_chart(fig_corr, use_container_width=True)


    # Engagement Analysis Section
    st.header("Engagement Analysis")

    # Daily aggregated engagement metrics
    daily_engagement = tweets_sheet.set_index('Date').resample('D').agg({
        likes_col: 'sum',
        reposts_col: 'sum',
        replies_col: 'sum',
        bookmarks_col: 'sum',
        impressions_col: 'sum'
    }).reset_index()

    # Create cumulative engagement
    for metric in [likes_col, reposts_col, replies_col, bookmarks_col]:
        daily_engagement[f'cum_{metric}'] = daily_engagement[metric].cumsum()

    # Add the import statement at the top
    try:
        # Time series of engagement metrics with subplots
        from plotly.subplots import make_subplots
        
        metrics = [likes_col, reposts_col, replies_col, bookmarks_col]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        fig_engagement = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for i, (metric, color, pos) in enumerate(zip(metrics, colors, positions)):
            fig_engagement.add_trace(
                go.Scatter(
                    x=daily_engagement['Date'], 
                    y=daily_engagement[metric],
                    name=metric,
                    line=dict(color=color, width=3),
                    fill='tozeroy',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)'
                ),
                row=pos[0], col=pos[1]
            )
        
        fig_engagement.update_layout(
            title="Daily Engagement Metrics - Detailed View",
            height=600,
            showlegend=False,
            font=dict(size=12)
        )
        
        fig_engagement.update_xaxes(title_text="Date")
        fig_engagement.update_yaxes(title_text="Count")
        
        st.plotly_chart(fig_engagement, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating engagement metrics chart: {str(e)}")
        st.info("Using fallback visualization...")
        
        # Fallback: Simple line chart
        fig_simple = go.Figure()
        for metric, color in zip([likes_col, reposts_col], ['#FF6B6B', '#4ECDC4']):
            fig_simple.add_trace(go.Scatter(
                x=daily_engagement['Date'], 
                y=daily_engagement[metric],
                name=metric,
                line=dict(color=color, width=3)
            ))
        
        fig_simple.update_layout(title="Daily Engagement Metrics (Simplified)")
        st.plotly_chart(fig_simple, use_container_width=True)

    # Engagement Rate Analysis
    st.subheader("Engagement Rate Analysis")
    tweets_sheet['engagement_rate'] = tweets_sheet[engagements_col]/tweets_sheet[impressions_col]

    # Calculate mean engagement rate
    mean_engagement_rate = tweets_sheet['engagement_rate'].mean()

    # Create the histogram with Plotly
    fig_engagement_rate = go.Figure()

    # Add histogram trace
    fig_engagement_rate.add_trace(go.Histogram(
        x=tweets_sheet['engagement_rate'],
        nbinsx=50,
        name='Engagement Rate',
        marker_color='#1f77b4'
    ))

    # Add mean line
    fig_engagement_rate.add_vline(
        x=mean_engagement_rate,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_engagement_rate:.2%}",
        annotation_position="top right"
    )

    # Update layout
    fig_engagement_rate.update_layout(
        title="Distribution of Engagement Rates",
        xaxis_title="Engagement Rate",
        yaxis_title="Count",
        showlegend=False
    )

    st.plotly_chart(fig_engagement_rate, use_container_width=True)

    # NEW: Post Performance Comparison Section
    st.header("Post Performance Comparison")
    
    # Controls for the comparison chart
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_posts = st.slider("Number of top posts to compare", min_value=5, max_value=20, value=10)
    
    with col2:
        metrics_options = [likes_col, reposts_col, replies_col, bookmarks_col, impressions_col, engagements_col]
        selected_metrics = st.multiselect(
            "Select metrics to compare", 
            metrics_options, 
            default=[likes_col, reposts_col, replies_col]
        )
    
    with col3:
        # NEW: Allow user to select sorting parameter
        sort_by_metric = st.selectbox(
            "Sort posts by:", 
            options=metrics_options,
            index=0,  # Default to likes
            help="Choose which metric to use for ranking the top posts"
        )
    
    if selected_metrics:
        # Get top N posts based on selected sorting metric
        top_posts = tweets_sheet.nlargest(n_posts, sort_by_metric).copy()
        
        # Create enhanced labels and hover data
        top_posts['post_label'] = top_posts.apply(
            lambda x: f"Post {x.name}: " + str(x[post_text_col])[:50] + ("..." if len(str(x[post_text_col])) > 50 else ""), 
            axis=1
        )
        
        # Add content preview and link information
        top_posts['content_preview'] = top_posts[post_text_col].apply(
            lambda x: str(x)[:100] + ("..." if len(str(x)) > 100 else "")
        )
        
        # Check if link column exists and add to hover data
        link_col = 'Postar link' if 'Postar link' in top_posts.columns else 'Post link' if 'Post link' in top_posts.columns else None
        
        # Create grouped bar chart with enhanced hover information
        fig_comparison = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, metric in enumerate(selected_metrics):
            # Create hover text with content preview
            hover_text = []
            for idx, row in top_posts.iterrows():
                hover_info = f"<b>{metric}: {row[metric]:,}</b><br>"
                hover_info += f"<b>Content:</b><br>{row['content_preview']}<br>"
                hover_info += f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d %H:%M')}<br>"
                hover_info += f"<b>Engagement Rate:</b> {row['engagement_rate']:.2%}<br>"
                if link_col and pd.notna(row[link_col]) and row[link_col] != '':
                    hover_info += f"<b>Link:</b> {row[link_col]}"
                hover_text.append(hover_info)
            
            fig_comparison.add_trace(go.Bar(
                name=metric,
                x=top_posts['post_label'],
                y=top_posts[metric],
                marker_color=colors[i % len(colors)],
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_text
            ))
        
        fig_comparison.update_layout(
            barmode='group',
            title=f"Top {n_posts} Posts Performance Comparison (Sorted by {sort_by_metric})",
            xaxis_title="Posts (hover for full content)",
            yaxis_title="Count",
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Add a detailed table below the chart
        st.subheader("📋 Detailed Post Information")
        
        # Prepare display columns
        display_cols = ['Date', 'content_preview']
        if link_col:
            display_cols.append(link_col)
        display_cols.extend(selected_metrics)
        display_cols.append('engagement_rate')
        
        # Create display dataframe
        display_df = top_posts[display_cols].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['engagement_rate'] = display_df['engagement_rate'].apply(lambda x: f"{x:.2%}")
        
        # Rename columns for better display
        column_names = {
            'content_preview': 'Content Preview (100 chars)',
            'engagement_rate': 'Engagement Rate'
        }
        if link_col:
            column_names[link_col] = 'Post Link'
            
        display_df = display_df.rename(columns=column_names)
        
        st.dataframe(display_df, use_container_width=True, height=300)
        
        # Add download button for detailed data
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Top Posts Data as CSV",
            data=csv,
            file_name=f"top_{n_posts}_posts_analysis.csv",
            mime="text/csv"
        )

    # NEW: Time-based Performance Analysis
    st.subheader("Performance by Time Patterns")
    
    # Extract day of week and month
    tweets_sheet['day_of_week'] = tweets_sheet['Date'].dt.day_name()
    tweets_sheet['month'] = tweets_sheet['Date'].dt.month_name()
    
    # Performance by day of week with rainbow colors
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    rainbow_colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3']  # Red to Violet
    
    daily_performance = tweets_sheet.groupby('day_of_week').agg({
        likes_col: 'mean',
        reposts_col: 'mean',
        replies_col: 'mean',
        impressions_col: 'mean',
        'engagement_rate': 'mean'
    }).reindex(day_order).reset_index()
    
    # Remove days with no data
    daily_performance = daily_performance.dropna()
    
    if len(daily_performance) > 0:
        # Create custom color mapping for days
        day_color_map = dict(zip(day_order, rainbow_colors))
        daily_performance['colors'] = daily_performance['day_of_week'].map(day_color_map)
        
        fig_daily = go.Figure()
        
        fig_daily.add_trace(go.Bar(
            x=daily_performance['day_of_week'],
            y=daily_performance['engagement_rate'],
            marker_color=daily_performance['colors'],
            name='Engagement Rate',
            hovertemplate='<b>%{x}</b><br>Engagement Rate: %{y:.2%}<extra></extra>'
        ))
        
        fig_daily.update_layout(
            title="🌈 Average Engagement Rate by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Avg Engagement Rate",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500,
            showlegend=False
        )
        
        # Add annotations for best days
        max_engagement_day = daily_performance.loc[daily_performance['engagement_rate'].idxmax()]
        fig_daily.add_annotation(
            x=max_engagement_day['day_of_week'],
            y=max_engagement_day['engagement_rate'],
            text=f"🔥 Best Day!<br>{max_engagement_day['engagement_rate']:.2%}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gold",
            arrowwidth=2,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gold",
            borderwidth=2
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Add insights
        st.info(f"📊 **Best performing day:** {max_engagement_day['day_of_week']} with {max_engagement_day['engagement_rate']:.2%} engagement rate")
    else:
        st.info("No daily data available for analysis")

    # Content Analysis Section
    st.header("Content Analysis")

    # Extract mentions and hashtags
    def extract_features(text, pattern):
        return [mention for mention in re.findall(pattern, str(text))]

    # Update the extract_features function to handle case sensitivity
    def extract_features2(text, pattern):
        return [mention.lower() for mention in re.findall(pattern, str(text))]

    tweets_sheet['mentions'] = tweets_sheet[post_text_col].apply(
        lambda x: extract_features2(x, r'@(\w+)'))
    tweets_sheet['hashtags'] = tweets_sheet[post_text_col].apply(
        lambda x: extract_features(x, r'#(\w+)'))

    # NEW: Hashtag Analysis
    st.subheader("Hashtag Performance Analysis")
    
    # Get hashtag counts and performance
    hashtag_counts = pd.Series([tag for sublist in tweets_sheet['hashtags'] for tag in sublist]).value_counts().head(15)
    
    if len(hashtag_counts) > 0:
        # Simple hashtag usage chart with better colors
        fig_hashtags = px.bar(
            x=hashtag_counts.index, 
            y=hashtag_counts.values,
            labels={'x': 'Hashtag', 'y': 'Usage Count'},
            title="Top 15 Hashtags by Usage",
            color=hashtag_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_hashtags.update_layout(
            xaxis_tickangle=-45,
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hashtags, use_container_width=True)
        
        # Enhanced hashtag impact analysis with multi-dimensional bubble chart
        hashtag_effect = []
        for hashtag in hashtag_counts.head(12).index:  # Top 12 for better visualization
            hashtag_tweets = tweets_sheet[tweets_sheet['hashtags'].apply(lambda x: hashtag in x)]
            if len(hashtag_tweets) > 1:  # Only include hashtags with multiple uses
                avg_engagement = hashtag_tweets['engagement_rate'].mean()
                avg_likes = hashtag_tweets[likes_col].mean()
                avg_impressions = hashtag_tweets[impressions_col].mean()
                usage_count = len(hashtag_tweets)
                hashtag_effect.append({
                    'Hashtag': f"#{hashtag}", 
                    'Avg Engagement Rate': avg_engagement,
                    'Avg Likes': avg_likes,
                    'Avg Impressions': avg_impressions,
                    'Usage Count': usage_count
                })
        
        if len(hashtag_effect) > 0:
            hashtag_effect_df = pd.DataFrame(hashtag_effect)
            
            # Multi-dimensional bubble chart
            fig_hashtag_impact = px.scatter(
                hashtag_effect_df, 
                x='Usage Count', 
                y='Avg Engagement Rate',
                size='Avg Likes',
                color='Avg Impressions',
                hover_data=['Hashtag'],
                title="Hashtag Performance Matrix",
                labels={
                    'Usage Count': 'How Often Used',
                    'Avg Engagement Rate': 'Average Engagement Rate',
                    'Avg Likes': 'Average Likes (Bubble Size)',
                    'Avg Impressions': 'Average Impressions (Color)'
                },
                color_continuous_scale='Plasma',
                size_max=60
            )
            
            fig_hashtag_impact.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            
            fig_hashtag_impact.update_traces(
                marker=dict(line=dict(width=2, color='white')),
                selector=dict(mode='markers')
            )
            
            st.plotly_chart(fig_hashtag_impact, use_container_width=True)
            
            # Add explanation
            st.info("💡 **How to read this chart:** Larger bubbles = higher likes, warmer colors = more impressions, higher position = better engagement rate, further right = used more often")
    else:
        st.info("No hashtags found in your posts.")

    # Top mentions analysis with better styling
    mentions_counts = pd.Series([mention for sublist in tweets_sheet['mentions'] for mention in sublist]).value_counts().head(20)

    if len(mentions_counts) > 0:
        # Create a vibrant color palette for better visibility
        fig_mentions = px.bar(
            x=mentions_counts.index,
            y=mentions_counts.values,
            labels={'x': 'Mention', 'y': 'Count'},
            title="Top 20 Mentions",
            color=mentions_counts.values,
            color_continuous_scale='Turbo'  # Very vibrant and visible color scale
        )
        
        fig_mentions.update_layout(
            xaxis_tickangle=-45,
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        # Add custom colors for better contrast
        vibrant_colors = ['#FF1744', '#FF6D00', '#FFD600', '#00E676', '#00BCD4', 
                         '#2196F3', '#3F51B5', '#9C27B0', '#E91E63', '#F44336',
                         '#FF9800', '#FFEB3B', '#8BC34A', '#4CAF50', '#009688',
                         '#00ACC1', '#1976D2', '#512DA8', '#7B1FA2', '#C2185B']
        
        fig_mentions.update_traces(
            marker_color=vibrant_colors[:len(mentions_counts)]
        )
        
        st.plotly_chart(fig_mentions, use_container_width=True)

        # Impact of mentions on impressions with count-based colors
        mention_effect = []
        for mention in mentions_counts.head(15).index:  # Limit to top 15 for better visualization
            mention_tweets = tweets_sheet[tweets_sheet['mentions'].apply(lambda x: mention in x)]
            if len(mention_tweets) > 0:
                avg_impression = mention_tweets[impressions_col].mean()
                mention_count = len(mention_tweets)  # Number of times mentioned
                mention_effect.append({
                    'Mention': f"@{mention}", 
                    'Avg Impressions': avg_impression,
                    'Mention Count': mention_count
                })

        if mention_effect:
            mention_effect_df = pd.DataFrame(mention_effect)

            fig_mention_impact = px.bar(
                mention_effect_df, 
                x='Mention', 
                y='Avg Impressions',
                title="💬 Average Impressions by Mention",
                color='Mention Count',  # Color based on how many times mentioned
                color_continuous_scale='Plasma',  # High contrast color scale
                hover_data={'Mention Count': True},
                labels={'Mention Count': 'Times Mentioned'}
            )
            
            fig_mention_impact.update_layout(
                xaxis_tickangle=-45,
                coloraxis_showscale=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=500
            )
            
            st.plotly_chart(fig_mention_impact, use_container_width=True)
            st.info("💡 **How to read:** Darker colors = mentioned more often. Height = average impressions when mentioned.")
    else:
        st.info("No mentions found in your posts.")

    # Enhanced Tweet Length vs Engagement Analysis
    st.subheader("📏 Tweet Length vs Performance Analysis")
    
    # Add explanation
    st.info("💡 **How to read these charts:** Bubble size = impressions, color intensity = engagement rate (left) or likes (right). Look for patterns in successful content length!")
    
    # Create enhanced scatter plot with better scaling and multiple metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter outliers for better visualization (remove top 5% of likes)
        likes_threshold = tweets_sheet[likes_col].quantile(0.95)
        filtered_tweets = tweets_sheet[tweets_sheet[likes_col] <= likes_threshold]
        
        fig_length = px.scatter(
            filtered_tweets, 
            x='tweet_length', 
            y=likes_col,
            color='engagement_rate',
            size=impressions_col,
            hover_data=[post_text_col],
            title="Tweet Length vs Likes (Outliers Removed)",
            labels={
                'tweet_length': 'Tweet Length (characters)',
                likes_col: 'Likes',
                'engagement_rate': 'Engagement Rate'
            },
            color_continuous_scale='Viridis',
            size_max=30
        )
        
        # Add trendline with error handling
        try:
            if SKLEARN_AVAILABLE:
                # Prepare data for trendline
                X = filtered_tweets['tweet_length'].values.reshape(-1, 1)
                y = filtered_tweets[likes_col].values
                
                # Create polynomial regression
                poly_reg = make_pipeline(PolynomialFeatures(2), LinearRegression())
                poly_reg.fit(X, y)
                
                # Generate smooth line
                x_range = np.linspace(filtered_tweets['tweet_length'].min(), filtered_tweets['tweet_length'].max(), 100)
                y_pred = poly_reg.predict(x_range.reshape(-1, 1))
                
                # Add trendline
                fig_length.add_trace(go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=3, dash='dash')
                ))
            else:
                st.info("📊 Install scikit-learn for advanced trendline analysis: `pip install scikit-learn`")
        except Exception as e:
            st.warning(f"Could not add trendline: {str(e)}")
        
        fig_length.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        st.plotly_chart(fig_length, use_container_width=True)
    
    with col2:
        # Alternative view: Length vs Engagement Rate
        fig_engagement_length = px.scatter(
            tweets_sheet, 
            x='tweet_length', 
            y='engagement_rate',
            color=likes_col,
            size=impressions_col,
            title="Tweet Length vs Engagement Rate",
            labels={
                'tweet_length': 'Tweet Length (characters)',
                'engagement_rate': 'Engagement Rate',
                likes_col: 'Likes (Color)'
            },
            color_continuous_scale='Plasma',
            size_max=30
        )
        
        fig_engagement_length.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        st.plotly_chart(fig_engagement_length, use_container_width=True)

    # NEW: Content Length Categories Analysis with enhanced styling
    st.subheader("Content Length Performance Analysis")
    
    # Categorize tweets by length
    def categorize_length(length):
        if length <= 50:
            return "Very Short (≤50)"
        elif length <= 100:
            return "Short (51-100)"
        elif length <= 200:
            return "Medium (101-200)"
        else:
            return "Long (>200)"
    
    tweets_sheet['length_category'] = tweets_sheet['tweet_length'].apply(categorize_length)
    
    length_performance = tweets_sheet.groupby('length_category').agg({
        likes_col: 'mean',
        reposts_col: 'mean',
        replies_col: 'mean',
        'engagement_rate': 'mean',
        impressions_col: 'mean'
    }).reset_index()
    
    # Create side-by-side charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_length_cat = px.bar(
            length_performance, 
            x='length_category', 
            y='engagement_rate',
            title="Engagement Rate by Tweet Length",
            labels={'length_category': 'Tweet Length Category', 'engagement_rate': 'Avg Engagement Rate'},
            color='engagement_rate',
            color_continuous_scale='Viridis'
        )
        
        fig_length_cat.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_length_cat, use_container_width=True)
    
    with col2:
        # Additional metric: Impressions by length
        fig_length_imp = px.bar(
            length_performance, 
            x='length_category', 
            y=impressions_col,
            title="Average Impressions by Tweet Length",
            labels={'length_category': 'Tweet Length Category'},
            color=impressions_col,
            color_continuous_scale='Plasma'
        )
        
        fig_length_imp.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_length_imp, use_container_width=True)

    # Sentiment Analysis (requires TextBlob)
    tweets_sheet['sentiment'] = tweets_sheet[post_text_col].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # NEW: Enhanced Sentiment Analysis
    st.subheader("🎭 Sentiment Analysis & Word Insights")
    
    # Categorize sentiment
    def categorize_sentiment(score):
        if score > 0.3:
            return "Positive"
        elif score < -0.3:
            return "Negative"
        else:
            return "Neutral"
    
    tweets_sheet['sentiment_category'] = tweets_sheet['sentiment'].apply(categorize_sentiment)
    
    # Create word frequency analysis
    def extract_words(text):
        """Extract meaningful words from text, excluding common stop words and mentions"""
        import re
        # Remove URLs, mentions, hashtags, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Common stop words in Portuguese and English
        stop_words = {
            'de', 'da', 'do', 'das', 'dos', 'em', 'na', 'no', 'nas', 'nos', 'para', 'por', 'com', 'sem', 'um', 'uma', 'uns', 'umas',
            'o', 'a', 'os', 'as', 'e', 'ou', 'mas', 'que', 'se', 'te', 'me', 'nos', 'lhe', 'lhes', 'eu', 'tu', 'ele', 'ela',
            'nós', 'vós', 'eles', 'elas', 'meu', 'minha', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'seu', 'sua', 'seus', 'suas',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'hers', 'its', 'our', 'their', 'rt', '', 'não', 'é', 'só', 'já', 'mais', 'muito', 'bem', 'então'
        }
        
        words = [word.lower() for word in text.split() if len(word) > 2 and word.lower() not in stop_words]
        return words
    
    # Extract words from all posts
    all_words = []
    for text in tweets_sheet[post_text_col]:
        all_words.extend(extract_words(text))
    
    # Count word frequencies
    word_freq = pd.Series(all_words).value_counts().head(30)
    
    # Create enhanced visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive sentiment distribution over time
        sentiment_over_time = tweets_sheet.groupby([tweets_sheet['Date'].dt.date, 'sentiment_category']).size().unstack(fill_value=0)
        
        if len(sentiment_over_time) > 0:
            fig_sentiment_time = go.Figure()
            
            colors_sentiment = {'Positive': '#00D4AA', 'Neutral': '#FFA726', 'Negative': '#FF5252'}
            
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                if sentiment in sentiment_over_time.columns:
                    fig_sentiment_time.add_trace(go.Scatter(
                        x=sentiment_over_time.index,
                        y=sentiment_over_time[sentiment],
                        mode='lines+markers',
                        name=sentiment,
                        line=dict(color=colors_sentiment[sentiment], width=3),
                        fill='tonexty' if sentiment != 'Positive' else 'tozeroy',
                        fillcolor=f'rgba({int(colors_sentiment[sentiment][1:3], 16)}, {int(colors_sentiment[sentiment][3:5], 16)}, {int(colors_sentiment[sentiment][5:7], 16)}, 0.3)'
                    ))
            
            fig_sentiment_time.update_layout(
                title="📈 Sentiment Evolution Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Posts",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_sentiment_time, use_container_width=True)
    
    with col2:
        # Enhanced sentiment metrics
        sentiment_stats = tweets_sheet['sentiment_category'].value_counts()
        
        # Create a donut chart instead of pie
        fig_sentiment_donut = go.Figure(data=[go.Pie(
            labels=sentiment_stats.index, 
            values=sentiment_stats.values,
            hole=0.5,
            marker_colors=['#00D4AA', '#FFA726', '#FF5252'],
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig_sentiment_donut.update_layout(
            title="🎯 Overall Sentiment Distribution",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add center text
        fig_sentiment_donut.add_annotation(
            text=f"Total<br>{len(tweets_sheet)}<br>Posts",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )
        
        st.plotly_chart(fig_sentiment_donut, use_container_width=True)
    
    # Word frequency visualization
    if len(word_freq) > 0:
        st.subheader("🔤 Most Frequent Words in Your Posts")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create an attractive word frequency chart
            fig_words = px.bar(
                x=word_freq.values[:20],
                y=word_freq.index[:20],
                orientation='h',
                title="Top 20 Words by Frequency",
                labels={'x': 'Frequency', 'y': 'Words'},
                color=word_freq.values[:20],
                color_continuous_scale='Viridis'
            )
            
            fig_words.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=500,
                coloraxis_showscale=False,
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig_words, use_container_width=True)
        
        with col2:
            # Actual Word Cloud if available, otherwise text preview
            st.markdown("### ☁️ Word Cloud")
            
            if WORDCLOUD_AVAILABLE and len(word_freq) > 0:
                try:
                    # Create word frequency dictionary
                    word_freq_dict = word_freq.head(50).to_dict()
                    
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=400, 
                        height=300,
                        background_color='white',
                        colormap='viridis',
                        max_words=50,
                        relative_scaling=0.5,
                        random_state=42
                    ).generate_from_frequencies(word_freq_dict)
                    
                    # Create matplotlib figure
                    fig_wc, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout(pad=0)
                    
                    # Display in streamlit
                    st.pyplot(fig_wc, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating word cloud: {e}")
                    # Fallback to text preview
                    wordcloud_text = ""
                    for i, (word, freq) in enumerate(word_freq.head(15).items()):
                        size = max(12, min(24, int(freq * 2)))
                        wordcloud_text += f"<span style='font-size: {size}px; color: hsl({i*25}, 70%, 50%); margin: 5px;'>{word}</span> "
                    
                    st.markdown(f"<div style='text-align: center; padding: 20px; background: linear-gradient(45deg, #f0f0f0, #e0e0e0); border-radius: 10px;'>{wordcloud_text}</div>", unsafe_allow_html=True)
            else:
                # Text-based word cloud representation
                wordcloud_text = ""
                for i, (word, freq) in enumerate(word_freq.head(15).items()):
                    size = max(12, min(24, int(freq * 2)))
                    wordcloud_text += f"<span style='font-size: {size}px; color: hsl({i*25}, 70%, 50%); margin: 5px;'>{word}</span> "
                
                st.markdown(f"<div style='text-align: center; padding: 20px; background: linear-gradient(45deg, #f0f0f0, #e0e0e0); border-radius: 10px;'>{wordcloud_text}</div>", unsafe_allow_html=True)
                
                if not WORDCLOUD_AVAILABLE:
                    st.info("💡 **Enhanced word cloud available!** Install with: `pip install wordcloud matplotlib`")
    
    # Sentiment vs Performance Analysis
    st.subheader("📊 How Sentiment Affects Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Simplified sentiment performance analysis
        sentiment_performance = tweets_sheet.groupby('sentiment_category').agg({
            likes_col: 'mean',
            reposts_col: 'mean',
            replies_col: 'mean',
            'engagement_rate': 'mean',
            impressions_col: 'mean'
        }).reset_index()
        
        # Create separate charts for clarity
        fig_sentiment_eng = px.bar(
            sentiment_performance, 
            x='sentiment_category', 
            y='engagement_rate',
            title="🎯 Engagement Rate by Sentiment",
            labels={'sentiment_category': 'Sentiment', 'engagement_rate': 'Avg Engagement Rate'},
            color='engagement_rate',
            color_continuous_scale='RdYlGn',
            text='engagement_rate'
        )
        
        fig_sentiment_eng.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_sentiment_eng.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            height=400
        )
        
        st.plotly_chart(fig_sentiment_eng, use_container_width=True)
        
        st.info("📈 **How to read:** Higher bars = better engagement for that sentiment type")
    
    with col2:
        # Average impressions by sentiment
        fig_sentiment_imp = px.bar(
            sentiment_performance, 
            x='sentiment_category', 
            y=impressions_col,
            title="👁️ Average Impressions by Sentiment",
            labels={'sentiment_category': 'Sentiment'},
            color=impressions_col,
            color_continuous_scale='Blues',
            text=impressions_col
        )
        
        fig_sentiment_imp.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_sentiment_imp.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            height=400
        )
        
        st.plotly_chart(fig_sentiment_imp, use_container_width=True)
        
        st.info("👀 **How to read:** Higher bars = more people see posts with that sentiment")
    
    # NEW: Advanced Analytics Section
    st.header("Advanced Analytics")
    
    # NEW: Advanced Analytics Section
    st.header("🚀 Advanced Analytics")
    
    # Enhanced Viral potential analysis
    st.subheader("🔥 Viral Potential Analysis")
    
    # Explanation of what viral potential means
    with st.expander("🤔 What is Viral Potential Analysis?"):
        st.markdown("""
        **Viral Potential Analysis** helps you understand what makes your content go viral and reach wider audiences.
        
        **How it works:**
        - We define "viral" posts as those in the **top 10%** of impressions
        - We then analyze what characteristics these high-performing posts share
        - This helps you replicate successful content patterns
        
        **Key Metrics:**
        - **Viral Rate**: Percentage of your posts that achieve viral status
        - **Viral Characteristics**: Common features of your viral content
        - **Viral Timing**: When your viral posts were published
        - **Viral Content Patterns**: What type of content performs best
        """)
    
    # Define viral threshold (top 10% of impressions)
    viral_threshold = tweets_sheet[impressions_col].quantile(0.9)
    tweets_sheet['is_viral'] = tweets_sheet[impressions_col] >= viral_threshold
    
    viral_rate = tweets_sheet['is_viral'].mean()
    viral_posts_count = tweets_sheet['is_viral'].sum()
    
    # Enhanced metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔥 Viral Post Rate", 
            f"{viral_rate:.1%}", 
            help=f"Posts with ≥{viral_threshold:,.0f} impressions"
        )
    
    with col2:
        st.metric(
            "📈 Viral Posts Count", 
            f"{viral_posts_count}",
            help="Total number of viral posts"
        )
    
    with col3:
        if viral_posts_count > 0:
            avg_viral_engagement = tweets_sheet[tweets_sheet['is_viral']]['engagement_rate'].mean()
            st.metric(
                "⚡ Avg Viral Engagement", 
                f"{avg_viral_engagement:.2%}",
                help="Average engagement rate of viral posts"
            )
    
    with col4:
        if viral_posts_count > 0:
            max_impressions = tweets_sheet[impressions_col].max()
            st.metric(
                "🎯 Peak Impressions", 
                f"{max_impressions:,.0f}",
                help="Highest impressions achieved"
            )
    
    if viral_posts_count > 1:
        # Viral vs Non-Viral Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Characteristics comparison with normalized scales
            viral_analysis = tweets_sheet.groupby('is_viral').agg({
                'tweet_length': 'mean',
                'sentiment': 'mean',
                'engagement_rate': 'mean',
                'hashtags': lambda x: np.mean([len(tags) for tags in x]),
                'mentions': lambda x: np.mean([len(mentions) for mentions in x]),
                likes_col: 'mean',
                reposts_col: 'mean'
            }).round(3)
            
            # Check if both viral and non-viral posts exist
            if len(viral_analysis) < 2:
                st.warning("⚠️ Not enough data to compare viral vs non-viral characteristics. Need both viral and non-viral posts.")
            else:
                # Normalize characteristics for fair comparison (convert to percentage differences)
                normalized_data = []
                characteristics = ['tweet_length', 'sentiment', 'engagement_rate']
                
                for char in characteristics:
                    if False in viral_analysis.index and True in viral_analysis.index:
                        non_viral_val = viral_analysis.loc[False, char]
                        viral_val = viral_analysis.loc[True, char]
                        
                        # Calculate percentage difference
                        if non_viral_val != 0:
                            pct_diff = ((viral_val - non_viral_val) / abs(non_viral_val)) * 100
                        else:
                            pct_diff = 0
                        
                        normalized_data.append({
                            'Characteristic': char.replace('_', ' ').title(),
                            'Percentage Difference': pct_diff,
                            'Direction': 'Higher' if pct_diff > 0 else 'Lower',
                            'Non-Viral Value': non_viral_val,
                            'Viral Value': viral_val
                        })
                
                if normalized_data:
                    norm_df = pd.DataFrame(normalized_data)
                    
                    # Create normalized comparison chart
                    colors = ['#FF6B6B' if x < 0 else '#00D4AA' for x in norm_df['Percentage Difference']]
                    
                    fig_viral_comparison = go.Figure()
                    
                    fig_viral_comparison.add_trace(go.Bar(
                        x=norm_df['Characteristic'],
                        y=norm_df['Percentage Difference'],
                        marker_color=colors,
                        text=[f"{x:+.1f}%" for x in norm_df['Percentage Difference']],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>' +
                                    'Difference: %{y:.1f}%<br>' +
                                    '<extra></extra>'
                    ))
                    
                    fig_viral_comparison.update_layout(
                        title="🔍 Viral Posts vs Non-Viral Posts<br><sub>Percentage Difference in Key Characteristics</sub>",
                        xaxis_title="Characteristics",
                        yaxis_title="% Difference (Viral vs Non-Viral)",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=450,
                        showlegend=False
                    )
                    
                    # Add zero line
                    fig_viral_comparison.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig_viral_comparison, use_container_width=True)
                    
                    st.info("📊 **How to read:** Green bars = viral posts have MORE of this characteristic. Red bars = viral posts have LESS. The percentage shows how much difference.")
                else:
                    st.warning("⚠️ Unable to create comparison - insufficient data.")
        
        with col2:
            # Viral posts timing analysis
            if viral_posts_count > 0:
                viral_posts_timing = tweets_sheet[tweets_sheet['is_viral']]['day_of_week'].value_counts()
                
                fig_viral_timing = px.pie(
                    values=viral_posts_timing.values,
                    names=viral_posts_timing.index,
                    title="📅 When Do Your Posts Go Viral?",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                fig_viral_timing.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                
                st.plotly_chart(fig_viral_timing, use_container_width=True)
            else:
                st.info("📅 Not enough viral posts to analyze timing patterns")
        
        # Detailed viral analysis table
        st.subheader("📋 Detailed Viral vs Non-Viral Comparison")
        
        try:
            if len(viral_analysis) >= 2 and False in viral_analysis.index and True in viral_analysis.index:
                viral_analysis.index = ['🔵 Non-Viral Posts', '🔥 Viral Posts']
                viral_analysis_display = viral_analysis.T
                viral_analysis_display.columns = ['Non-Viral', 'Viral']
                
                # Calculate differences
                viral_analysis_display['Difference'] = viral_analysis_display['Viral'] - viral_analysis_display['Non-Viral']
                
                # Avoid division by zero
                viral_analysis_display['% Change'] = viral_analysis_display.apply(
                    lambda row: (row['Difference'] / abs(row['Non-Viral']) * 100) if row['Non-Viral'] != 0 else 0, axis=1
                ).round(1)
                
                # Format the display
                viral_analysis_display = viral_analysis_display.round(3)
                
                st.dataframe(viral_analysis_display, use_container_width=True)
            else:
                st.info("📊 **Insufficient data for detailed comparison.** Need both viral and non-viral posts to generate this analysis.")
        except Exception as e:
            st.warning(f"⚠️ **Analysis unavailable** - Data structure doesn't support detailed comparison yet.")
        
        # Insights and recommendations
        st.subheader("💡 Viral Content Insights")
        
        # Generate insights based on the data - with error checking
        insights = []
        
        try:
            # Check if we have both viral and non-viral posts
            if len(viral_analysis) >= 2 and False in viral_analysis.index and True in viral_analysis.index:
                if viral_analysis.loc[True, 'tweet_length'] > viral_analysis.loc[False, 'tweet_length']:
                    insights.append("✅ **Longer posts tend to go viral** - Consider adding more detail to your content")
                else:
                    insights.append("✅ **Shorter posts tend to go viral** - Keep your content concise and punchy")
                
                if viral_analysis.loc[True, 'sentiment'] > viral_analysis.loc[False, 'sentiment']:
                    insights.append("✅ **Positive sentiment drives virality** - Focus on uplifting, positive content")
                elif viral_analysis.loc[True, 'sentiment'] < viral_analysis.loc[False, 'sentiment']:
                    insights.append("✅ **Controversial/negative content can go viral** - Consider your brand voice carefully")
                
                if viral_analysis.loc[True, 'hashtags'] > viral_analysis.loc[False, 'hashtags']:
                    insights.append("✅ **More hashtags increase viral potential** - Use relevant hashtags strategically")
                
                if viral_analysis.loc[True, 'mentions'] > viral_analysis.loc[False, 'mentions']:
                    insights.append("✅ **Mentioning others boosts virality** - Engage with your community")
            else:
                insights.append("📊 **Limited viral data** - Keep posting consistently to build better insights")
            
            # Best viral day
            if viral_posts_count > 0:
                viral_posts_timing = tweets_sheet[tweets_sheet['is_viral']]['day_of_week'].value_counts()
                if len(viral_posts_timing) > 0:
                    best_viral_day = viral_posts_timing.index[0]
                    insights.append(f"📅 **{best_viral_day} is your best viral day** - Schedule important content then")
            
            if len(insights) == 0:
                insights.append("📈 **Keep creating!** - More data needed to generate specific insights")
                
        except Exception as e:
            insights.append(f"⚠️ **Analysis in progress** - Some insights unavailable due to data structure")
        
        for insight in insights:
            st.info(insight)
        
        # Show top viral posts
        if viral_posts_count > 0:
            st.subheader("🏆 Your Top Viral Posts")
            viral_posts = tweets_sheet[tweets_sheet['is_viral']].nlargest(5, impressions_col)
            
            viral_display = viral_posts[['Date', post_text_col, impressions_col, likes_col, 'engagement_rate']].copy()
            viral_display['Date'] = viral_display['Date'].dt.strftime('%Y-%m-%d')
            viral_display['engagement_rate'] = viral_display['engagement_rate'].apply(lambda x: f"{x:.2%}")
            viral_display[post_text_col] = viral_display[post_text_col].apply(lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x))
            
            st.dataframe(viral_display, use_container_width=True)
    
    else:
        st.info("📊 **Need more data for viral analysis.** Keep posting consistently to build a robust viral potential analysis!")
        
        # Show potential viral posts (top 20% instead)
        potential_threshold = tweets_sheet[impressions_col].quantile(0.8)
        potential_viral = tweets_sheet[tweets_sheet[impressions_col] >= potential_threshold]
        
        if len(potential_viral) > 0:
            st.subheader("🌟 Your Best Performing Posts (Top 20%)")
            st.dataframe(
                potential_viral[['Date', post_text_col, impressions_col, likes_col, 'engagement_rate']].head(5),
                use_container_width=True
            )

    # Top Performing Tweets
    st.subheader("Top Performing Tweets")
    
    # Allow user to select sorting metric
    sort_options = [likes_col, reposts_col, replies_col, impressions_col, engagements_col]
    sort_metric = st.selectbox("Sort by:", sort_options)
    
    top_tweets = tweets_sheet.nlargest(10, sort_metric)[['Date', post_text_col, likes_col, reposts_col, replies_col, impressions_col, 'engagement_rate']]
    top_tweets['engagement_rate'] = top_tweets['engagement_rate'].apply(lambda x: f"{x:.2%}")
    st.dataframe(top_tweets, use_container_width=True)
    
    # NEW: Performance Insights Summary
    st.header("📊 Key Insights Summary")
    
    # Calculate insights with error handling
    try:
        # Remove hourly insights since we removed the hourly chart
        best_day = daily_performance.loc[daily_performance['engagement_rate'].idxmax(), 'day_of_week'] if len(daily_performance) > 0 else "N/A"
        best_length = length_performance.loc[length_performance['engagement_rate'].idxmax(), 'length_category'] if len(length_performance) > 0 else "N/A"
        best_sentiment = sentiment_performance.loc[sentiment_performance['engagement_rate'].idxmax(), 'sentiment_category'] if len(sentiment_performance) > 0 else "N/A"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **🗓️ Timing Insights:**
            - Best day to post: **{best_day}**
            - Viral post rate: **{viral_rate:.1%}**
            """)
            
            st.success(f"""
            **📝 Content Insights:**
            - Optimal tweet length: **{best_length}**
            - Best performing sentiment: **{best_sentiment}**
            """)
        
        with col2:
            if 'hashtag_effect_df' in locals() and len(hashtag_effect_df) > 0:
                best_hashtag = hashtag_effect_df.loc[hashtag_effect_df['Avg Engagement Rate'].idxmax(), 'Hashtag']
                st.success(f"""
                **📈 Engagement Insights:**
                - Average engagement rate: **{avg_engagement_rate:.2%}**
                - Best performing hashtag: **{best_hashtag}**
                """)
            else:
                st.success(f"""
                **📈 Engagement Insights:**
                - Average engagement rate: **{avg_engagement_rate:.2%}**
                - Best performing hashtag: **No hashtags found**
                """)
            
            st.success(f"""
            **🚀 Growth Insights:**
            - Total posts analyzed: **{total_tweets:,}**
            - Most frequent words: **{', '.join(word_freq.head(3).index.tolist()) if 'word_freq' in locals() and len(word_freq) > 0 else 'N/A'}**
            """)
            
        # Action recommendations
        st.subheader("🎯 Recommended Actions")
        
        recommendations = []
        
        if viral_rate < 0.1:
            recommendations.append("🔥 **Increase viral potential** - Study your top-performing posts and replicate their characteristics")
        
        if avg_engagement_rate < 0.05:
            recommendations.append("📈 **Boost engagement** - Try asking questions, using calls-to-action, or posting at peak times")
        
        if best_day != "N/A":
            recommendations.append(f"📅 **Optimize timing** - Post more content on {best_day} when your audience is most engaged")
        
        if len(word_freq) > 0 and 'word_freq' in locals():
            top_words = ', '.join(word_freq.head(3).index.tolist())
            recommendations.append(f"🔤 **Leverage top words** - Your audience resonates with: {top_words}")
        
        recommendations.append("📊 **Track progress** - Return weekly to monitor improvements and adjust strategy")
        
        for rec in recommendations:
            st.info(rec)
            
    except Exception as e:
        st.error(f"Error generating insights summary: {str(e)}")
        st.info("Some insights may not be available due to insufficient data.")
