import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from textblob import TextBlob
import numpy as np
import streamlit.components.v1 as components
import os

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

# --- Page Configuration ---
st.set_page_config(page_title="Twitter Analytics Dashboard", layout="wide", 
                   page_icon="🐦")

# --- Custom CSS for modern UI ---
st.markdown("""
<style>
    .stApp {
        background-color: #F0F2F6;
    }
    .stMetric {
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        background-color: #FFFFFF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1, h2 {
        color: #1DA1F2; /* Twitter Blue */
    }
    .st-expander {
        border: 1px solid #E0E0E0 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    /* Style for the controls container */
    .controls-container {
        border: 1px solid #1DA1F2;
        border-radius: 10px;
        padding: 20px;
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar for Controls and Information ---
with st.sidebar:
    st.title("📊 Dashboard Controls")
    st.info("Upload your Twitter data files below to get started.")

    tweets_sheet_file = st.file_uploader("1. Upload Content Data CSV", type=["csv"], help="This is the `account_analytics_content_...` file.")
    account_analytics_file = st.file_uploader("2. Upload Overview Data CSV", type=["csv"], help="This is the `account_overview_analytics_...` file.")

    st.markdown("---")
    st.markdown("Made by [Felipe Gabriel](https://x.com/FelG_research)")

    with st.expander("📚 Quick Start Guide"):
        st.markdown("""
        ### **Must have Premium/Professional Features!**
        Access advanced analytics on Twitter to download the necessary files.
        1.  **`account_analytics_content_*.csv`**
        2.  **`account_overview_analytics_*.csv`**
        """)

# --- Data Loading and Preprocessing ---
account_analytics = None
tweets_sheet = None

# Added engine='python' to prevent ParserError from malformed CSVs.
if account_analytics_file is not None:
    try:
        account_analytics = pd.read_csv(account_analytics_file, engine='python')
        account_analytics['Date'] = pd.to_datetime(account_analytics['Date'], format='mixed')
        account_analytics = account_analytics.sort_values(by='Date')

        if 'Novos seguidores' in account_analytics.columns and 'Deixar de seguir' in account_analytics.columns:
            account_analytics['followers'] = (account_analytics['Novos seguidores'] - account_analytics['Deixar de seguir']).cumsum()
        elif 'New follows' in account_analytics.columns and 'Unfollows' in account_analytics.columns:
            account_analytics['followers'] = (account_analytics['New follows'] - account_analytics['Unfollows']).cumsum()
    except Exception as e:
        st.error(f"Error processing overview file: {e}")
        account_analytics = None


if tweets_sheet_file is not None:
    try:
        tweets_sheet = pd.read_csv(tweets_sheet_file, engine='python')
        if 'Data' in tweets_sheet.columns:
            tweets_sheet['Date'] = pd.to_datetime(tweets_sheet['Data'], format='mixed')
            tweets_sheet['tweet_length'] = tweets_sheet['Texto do post'].apply(lambda x: len(str(x)))
        elif 'Date' in tweets_sheet.columns:
            tweets_sheet['Date'] = pd.to_datetime(tweets_sheet['Date'], format='mixed')
            tweets_sheet['tweet_length'] = tweets_sheet['Post text'].apply(lambda x: len(str(x)))

        tweets_sheet = tweets_sheet.sort_values(by='Date')
    except Exception as e:
        st.error(f"Error processing content file: {e}")
        tweets_sheet = None

# --- Main Dashboard ---
if account_analytics is not None and tweets_sheet is not None:
    # --- Language and Column Name Handling ---
    if 'Curtidas' in tweets_sheet.columns:
        likes_col, impressions_col, engagements_col, reposts_col, replies_col, bookmarks_col, post_text_col = 'Curtidas', 'Impressões', 'Engajamentos', 'Compartilhamentos', 'Respostas', 'Itens salvos', 'Texto do post'
    else:
        likes_col, impressions_col, engagements_col, reposts_col, replies_col, bookmarks_col, post_text_col = 'Likes', 'Impressions', 'Engagements', 'Reposts', 'Replies', 'Bookmarks', 'Post text'

    if 'New follows' in account_analytics.columns:
        new_followers_col = 'New follows'
    else:
        new_followers_col = 'Novos seguidores'

    # --- Pre-computation for Analyses ---
    tweets_sheet.dropna(subset=[impressions_col, engagements_col], inplace=True)
    tweets_sheet = tweets_sheet[tweets_sheet[impressions_col] > 0] # Avoid division by zero
    tweets_sheet['engagement_rate'] = tweets_sheet[engagements_col] / tweets_sheet[impressions_col]
    avg_engagement_rate = tweets_sheet[engagements_col].sum() / tweets_sheet[impressions_col].sum()

    # --- Tabbed Interface ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚀 Overview", "📈 Engagement Analysis", "✍️ Content Analysis", "😊 Sentiment Analysis", "🔥 Viral Potential", "🎯 Insights Summary"])

    with tab1:
        st.header("Account Overview")
        st.markdown("*A high-level look at your Twitter performance.*")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("Total Tweets", len(tweets_sheet))
        with col2: st.metric("Total Likes", f"{tweets_sheet[likes_col].sum():,}")
        with col3: st.metric("Total Impressions", f"{tweets_sheet[impressions_col].sum():,}")
        with col4: st.metric("Total New Followers", f"{account_analytics[new_followers_col].sum():,}")
        with col5: st.metric("Avg. Engagement Rate", f"{avg_engagement_rate:.2%}")
        st.markdown("---")
        if 'followers' in account_analytics.columns:
            fig_followers = go.Figure(go.Scatter(x=account_analytics['Date'], y=account_analytics['followers'], mode='lines', name='Followers', line=dict(color='#1DA1F2', width=3)))
            fig_followers.update_layout(title="Follower Growth Over Time", xaxis_title="Date", yaxis_title="Total Followers", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_followers, use_container_width=True)

            
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

    with tab2:
        st.header("Engagement Deep-Dive")
        st.markdown("*Understanding how your audience interacts with your content.*")
        daily_engagement = tweets_sheet.set_index('Date').resample('D').agg({likes_col: 'sum', reposts_col: 'sum', replies_col: 'sum', bookmarks_col: 'sum'}).reset_index()
        metrics = [likes_col, reposts_col, replies_col, bookmarks_col]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        fig_engagement = make_subplots(rows=2, cols=2, subplot_titles=metrics, vertical_spacing=0.15, horizontal_spacing=0.1)
        positions = [(1,1), (1,2), (2,1), (2,2)]
        for i, (metric, color, pos) in enumerate(zip(metrics, colors, positions)):
            fig_engagement.add_trace(go.Scatter(x=daily_engagement['Date'], y=daily_engagement[metric], name=metric, line=dict(color=color, width=3), fill='tozeroy'), row=pos[0], col=pos[1])
        fig_engagement.update_layout(title="Daily Engagement Metrics", height=600, showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_engagement, use_container_width=True)

            
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
        

            
        # Update the extract_features function to handle case sensitivity
        def extract_features2(text, pattern):
            return [mention.lower() for mention in re.findall(pattern, str(text))]

        tweets_sheet['mentions'] = tweets_sheet[post_text_col].apply(
            lambda x: extract_features2(x, r'@(\w+)'))
        
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
                color_continuous_scale='Plasma'#'Turbo'  # Very vibrant and visible color scale
            )
            
            fig_mentions.update_layout(
                xaxis_tickangle=-45,
                coloraxis_showscale=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=500
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

        
    with tab3:
        st.header("Content and Performance Analysis")
        st.markdown("*Analyzing what type of content works best.*")

        # Tweet Length vs Engagement
        st.subheader("Tweet Length vs. Performance")

        col1, col2 = st.columns(2)
        with col1:
            fig_length = px.scatter(
                tweets_sheet, x='tweet_length', y=likes_col,
                color='engagement_rate', size=impressions_col,
                hover_data=[post_text_col], title="Tweet Length vs. Likes",
                labels={'tweet_length': 'Tweet Length (chars)'},
                color_continuous_scale='Viridis', size_max=30
            )

            # Ensure trendline appears and give clear instructions.
            if SKLEARN_AVAILABLE and len(tweets_sheet) > 10:
                try:
                    X = tweets_sheet[['tweet_length']].values
                    y = tweets_sheet[likes_col].values
                    poly_reg = make_pipeline(PolynomialFeatures(2), LinearRegression())
                    poly_reg.fit(X, y)
                    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    y_pred = poly_reg.predict(x_range)
                    fig_length.add_trace(go.Scatter(x=x_range.ravel(), y=y_pred, mode='lines', name='Trend', line=dict(color='red', width=3, dash='dash')))
                except Exception as e:
                    st.warning(f"Could not draw trendline: {e}")
            elif not SKLEARN_AVAILABLE:
                 st.info("💡 Install scikit-learn (`pip install scikit-learn`) to see a performance trendline on this chart.")


            fig_length.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_length, use_container_width=True)

        with col2:
            fig_engagement_length = px.scatter(tweets_sheet, x='tweet_length', y='engagement_rate', color=likes_col, size=impressions_col, title="Tweet Length vs. Engagement Rate", labels={'tweet_length': 'Tweet Length (chars)'}, color_continuous_scale='Plasma', size_max=30)
            fig_engagement_length.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_engagement_length, use_container_width=True)

        
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

    with tab4:

            
        # Sentiment Analysis (requires TextBlob)
        tweets_sheet['sentiment'] = tweets_sheet[post_text_col].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity)
        
        # NEW: Enhanced Sentiment Analysis
        st.subheader("🎭 Sentiment Analysis & Word Insights")
        
        # Categorize sentiment
        def categorize_sentiment(score):
            if score > 0.25:
                return "Positive"
            elif score < -0.25:
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
            # REFINEMENT 5: Sentiment time series without shadows/fills
            sentiment_over_time = tweets_sheet.groupby([tweets_sheet['Date'].dt.date, 'sentiment_category']).size().unstack(fill_value=0)
            
            if len(sentiment_over_time) > 0:
                fig_sentiment_time = go.Figure()
                
                # REFINEMENT 5: Updated colors and NO FILLS
                colors_sentiment = {'Positive': '#2ca02c', 'Neutral': '#ffdd00', 'Negative': '#d62728'}
                
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    if sentiment in sentiment_over_time.columns:
                        fig_sentiment_time.add_trace(go.Scatter(
                            x=sentiment_over_time.index,
                            y=sentiment_over_time[sentiment],
                            mode='lines+markers',
                            name=sentiment,
                            line=dict(color=colors_sentiment[sentiment], width=3)
                            # REFINEMENT 5: NO fill property - removed shadows
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
            # Define color mapping for sentiments
            color_mapping = {
                'positive': '#00D4AA',  # Green
                'neutral': '#FFA726',   # Yellow/Orange  
                'negative': '#FF5252'   # Red
            }

            # Create colors list in the same order as sentiment_stats.index
            colors = [color_mapping.get(sentiment.lower(), '#CCCCCC') for sentiment in sentiment_stats.index]

            # Create a donut chart instead of pie
            fig_sentiment_donut = go.Figure(data=[go.Pie(
                labels=sentiment_stats.index, 
                values=sentiment_stats.values,
                hole=0.5,
                marker_colors=colors,  # Use the mapped colors
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
            
            #st.plotly_chart(fig_sentiment_donut, use_container_width=True)
        
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
        
        # REFINEMENT 5: Sentiment vs Performance Analysis with fixed colors
        st.subheader("📊 How Sentiment Affects Performance")
        
        col1, col2 = st.columns(2)
        
        # REFINEMENT 5: Define exact color mapping
        sentiment_color_map = {
            'Positive': '#2ca02c',  # Green
            'Negative': '#d62728',  # Red
            'Neutral': '#ffdd00'    # Yellow
        }
        
        with col1:
            # Sentiment performance analysis
            sentiment_performance = tweets_sheet.groupby('sentiment_category').agg({
                likes_col: 'mean',
                reposts_col: 'mean',
                replies_col: 'mean',
                'engagement_rate': 'mean',
                impressions_col: 'mean'
            }).reset_index()
            
            # REFINEMENT 5: Use exact colors specified
            fig_sentiment_eng = px.bar(
                sentiment_performance, 
                x='sentiment_category', 
                y='engagement_rate',
                title="🎯 Engagement Rate by Sentiment",
                labels={'sentiment_category': 'Sentiment', 'engagement_rate': 'Avg Engagement Rate'},
                color='sentiment_category',
                color_discrete_map=sentiment_color_map,
                text='engagement_rate'
            )
            
            fig_sentiment_eng.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig_sentiment_eng.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_sentiment_eng, use_container_width=True)
            
            st.info("📈 **How to read:** Higher bars = better engagement for that sentiment type")
        
        with col2:
            # REFINEMENT 5: Average impressions by sentiment with exact colors
            fig_sentiment_imp = px.bar(
                sentiment_performance, 
                x='sentiment_category', 
                y=impressions_col,
                title="👁️ Average Impressions by Sentiment",
                labels={'sentiment_category': 'Sentiment'},
                color='sentiment_category',
                color_discrete_map=sentiment_color_map,
                text=impressions_col
            )
            
            fig_sentiment_imp.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_sentiment_imp.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_sentiment_imp, use_container_width=True)
            
            st.info("👀 **How to read:** Higher bars = more people see posts with that sentiment")
        


    with tab5:
        st.header("Viral Potential Analysis")
        st.markdown("*Identifying the characteristics of your most successful posts.*")

        # Make the controls more prominent.
        st.markdown("---")
        with st.container():
            #st.markdown('<div class="controls-container">', unsafe_allow_html=True)
            st.subheader("🔬 Analysis Controls")
            viral_metric = st.selectbox(
                "Define 'Viral'🦠 based on top 10% of:",
                [impressions_col, likes_col, engagements_col, reposts_col],
                index=0,
                help="Choose which metric defines a 'viral' post."
            )
            #st.markdown('</div>', unsafe_allow_html=True)

        viral_threshold = tweets_sheet[viral_metric].quantile(0.9)
        tweets_sheet['is_viral'] = tweets_sheet[viral_metric] >= viral_threshold
        viral_rate = tweets_sheet['is_viral'].mean()
        viral_posts_count = tweets_sheet['is_viral'].sum()

        st.markdown("### Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("🔥 Viral Post Rate", f"{viral_rate:.1%}", help=f"Posts with ≥{viral_threshold:,.0f} {viral_metric}")
        col2.metric("📈 Viral Posts Count", f"{viral_posts_count}")
        if viral_posts_count > 0:
            avg_viral_engagement = tweets_sheet[tweets_sheet['is_viral']]['engagement_rate'].mean()
            col3.metric("⚡ Avg. Viral Engagement", f"{avg_viral_engagement:.2%}")

        if viral_posts_count > 1:
            st.markdown("---")
            st.subheader("Viral vs. Non-Viral Characteristics")
            col1, col2 = st.columns(2)

            with col1:
                tweets_sheet['sentiment'] = tweets_sheet[post_text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
                viral_analysis = tweets_sheet.groupby('is_viral').agg({'tweet_length': 'mean', 'sentiment': 'mean', 'engagement_rate': 'mean'}).round(3)

                if len(viral_analysis) == 2:
                    norm_data = []
                    for char in ['tweet_length', 'sentiment', 'engagement_rate']:
                        non_viral_val, viral_val = viral_analysis.loc[False, char], viral_analysis.loc[True, char]
                        pct_diff = ((viral_val - non_viral_val) / abs(non_viral_val)) * 100 if non_viral_val != 0 else 0
                        norm_data.append({'Characteristic': char.replace('_', ' ').title(), 'Percentage Difference': pct_diff})

                    norm_df = pd.DataFrame(norm_data)
                    colors = ['#00D4AA' if x > 0 else '#FF6B6B' for x in norm_df['Percentage Difference']]
                    fig_viral_comp = px.bar(norm_df, x='Characteristic', y='Percentage Difference', title="Key Differences in Viral Posts", text=[f"{x:+.1f}%" for x in norm_df['Percentage Difference']])
                    fig_viral_comp.update_traces(marker_color=colors)
                    fig_viral_comp.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_viral_comp, use_container_width=True)
                else:
                    st.info("Not enough data to compare viral vs non-viral characteristics.")

            with col2:
                tweets_sheet['day_of_week'] = tweets_sheet['Date'].dt.day_name()
                viral_timing = tweets_sheet[tweets_sheet['is_viral']]['day_of_week'].value_counts()
                fig_viral_timing = px.pie(values=viral_timing.values, names=viral_timing.index, title="🗓️ When Do Posts Go Viral?", color_discrete_sequence=px.colors.qualitative.Set3)
                fig_viral_timing.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_viral_timing, use_container_width=True)

            st.markdown("---")
            st.subheader("🏆 Your Top 5 Viral Posts")
            viral_posts_df = tweets_sheet[tweets_sheet['is_viral']].nlargest(5, viral_metric)
            st.dataframe(viral_posts_df[[post_text_col, viral_metric, 'engagement_rate', 'Date']], use_container_width=True)
        else:
            st.info("Not enough viral posts to perform a detailed comparison. Keep posting!")


    with tab6:
        st.header("🎯 Key Insights & Recommendations")
        # Pre-computation for this tab to avoid errors on small datasets
        tweets_sheet['day_of_week'] = tweets_sheet['Date'].dt.day_name()
        daily_performance = tweets_sheet.groupby('day_of_week').agg({'engagement_rate': 'mean'})
        best_day = daily_performance['engagement_rate'].idxmax() if not daily_performance.empty else "N/A"

        tweets_sheet['length_category'] = pd.cut(tweets_sheet['tweet_length'], bins=[0, 70, 140, 280], labels=["Short", "Medium", "Long"])
        length_performance = tweets_sheet.groupby('length_category', observed=False).agg({'engagement_rate': 'mean'})
        best_length = length_performance['engagement_rate'].idxmax() if not length_performance.empty else "N/A"
        
        #tweets_sheet['sentiment'] = tweets_sheet[post_text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        def categorize_sentiment(score):
            if score > 0.2: return "Positive"
            elif score < -0.2: return "Negative"
            else: return "Neutral"
        tweets_sheet['sentiment_category'] = tweets_sheet['sentiment'].apply(categorize_sentiment)
        sentiment_performance = tweets_sheet.groupby('sentiment_category').agg({'engagement_rate': 'mean'})
        best_sentiment = sentiment_performance['engagement_rate'].idxmax() if not sentiment_performance.empty else "N/A"

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Summary")
            st.success(f"**🗓️ Best Day:** Your posts perform best on **{best_day}**.")
            st.success(f"**📝 Optimal Length:** **{best_length}** posts tend to get the highest engagement.")
            st.success(f"**😊 Best Sentiment:** **{best_sentiment}** content resonates most with your audience.")
            st.success(f"**🔥 Viral Rate:** **{viral_rate:.1%}** of your posts achieve viral status.")
        
        with col2:
            st.subheader("Recommended Actions")
            if best_day != "N/A":
                st.info(f"📅 **Optimize Timing:** Schedule more of your important content to be posted on **{best_day}s**.")
            if best_length != "N/A":
                 st.info(f"✍️ **Refine Content:** Focus on creating more **{best_length.lower()}** posts to boost engagement.")
            if viral_rate < 0.05:
                st.info("🚀 **Boost Virality:** Analyze your top-performing posts and try to replicate their style and topics.")
            st.info("📊 **Track Progress:** Return weekly to monitor improvements and adjust your strategy accordingly.")

else:
    st.title("📊 Welcome to the Twitter Analytics Dashboard")
    st.markdown("---")
    st.info("👈 Please upload your **Content** and **Overview** data files using the sidebar to begin your analysis.")
    st.image("https://images.unsplash.com/photo-1611605698335-8b1569810432?q=80&w=1974", caption="Unlock insights from your Twitter data.")
