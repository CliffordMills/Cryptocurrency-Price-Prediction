import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

# Loading the model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Setting the Streamlit app
st.set_page_config(page_title='Cryptocurrency Closing Price Prediction', page_icon='bitcoin.svg',
                   layout='centered', initial_sidebar_state='expanded')

# Custom CSS to add a background image
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        background-size: cover;
        background-position: center;
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.8);
    }
    .st-bm {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Cryptocurrency Closing Price Prediction')

# Load and display the image using PIL
image = Image.open('crypto_img.jpg')
st.image(image, use_column_width=True)

# User choice for input method
input_method = st.sidebar.radio("Select Input Method", ("Manual Input", "Upload CSV/Excel"))

input_df = None
scaler = MinMaxScaler()  # Instantiate scaler once

# Required columns list
required_columns = ['open', 'high', 'low', 'volume', 'market_cap', 'url_shares', 'unique_url_shares', 
                    'reddit_posts', 'reddit_posts_score', 'reddit_comments', 'reddit_comments_score', 
                    'tweets', 'tweet_spam', 'tweet_followers', 'tweet_quotes', 'tweet_retweets', 
                    'tweet_replies', 'tweet_favorites', 'tweet_sentiment1', 'tweet_sentiment2', 
                    'tweet_sentiment3', 'tweet_sentiment4', 'tweet_sentiment5', 
                    'tweet_sentiment_impact1', 'tweet_sentiment_impact2', 'tweet_sentiment_impact3', 
                    'tweet_sentiment_impact4', 'tweet_sentiment_impact5', 'social_score', 
                    'average_sentiment', 'news', 'price_score', 'social_impact_score', 'correlation_rank', 
                    'galaxy_score', 'volatility', 'market_cap_rank', 'percent_change_24h_rank', 
                    'volume_24h_rank', 'social_volume_24h_rank', 'social_score_24h_rank', 'social_volume', 
                    'percent_change_24h', 'market_cap_global']

if input_method == "Upload CSV/Excel":
    # File upload form
    with st.sidebar.form(key='file_upload_form'):
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        submit_file_button = st.form_submit_button(label='Predict (Upload File)')

    if uploaded_file is not None and submit_file_button:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            input_df = pd.read_excel(uploaded_file)
        
        # Ensure the required columns are present
        if not all(col in input_df.columns for col in required_columns):
            st.error("Uploaded file does not contain all required columns.")
            input_df = None

else:
    # Manual input fields using a form
    with st.sidebar.form(key='manual_input_form'):
        st.header('Enter parameters below')
        
        open = st.number_input('Open Price', min_value=0.0, step=0.01)
        high = st.number_input('High Price', min_value=0.0, step=0.01)
        low = st.number_input('Low Price', min_value=0.0, step=0.01)
        volume = st.number_input('Volume', min_value=0.0, step=0.01)
        market_cap = st.number_input('Market Cap', min_value=0.0, step=0.01)
        url_shares = st.number_input('URL Shares', min_value=0.0, step=0.01)
        unique_url_shares = st.number_input('Unique URL Shares', min_value=0.0, step=0.01)
        reddit_posts = st.number_input('Reddit Posts', min_value=0.0, step=0.01)
        reddit_posts_score = st.number_input('Reddit Posts Score', min_value=0.0, step=0.01)
        reddit_comments = st.number_input('Reddit Comments', min_value=0.0, step=0.01)
        reddit_comments_score = st.number_input('Reddit Comments Score', min_value=0.0, step=0.01)
        tweets = st.number_input('Tweets', min_value=0.0, step=0.01)
        tweet_spam = st.number_input('Tweet Spam', min_value=0.0, step=0.01)
        tweet_followers = st.number_input('Tweet Followers', min_value=0.0, step=0.01)
        tweet_quotes = st.number_input('Tweet Quotes', min_value=0.0, step=0.01)
        tweet_retweets = st.number_input('Tweet Retweets', min_value=0.0, step=0.01)
        tweet_replies = st.number_input('Tweet Replies', min_value=0.0, step=0.01)
        tweet_favorites = st.number_input('Tweet Favorites', min_value=0.0, step=0.01)
        tweet_sentiment1 = st.number_input('Tweet Sentiment 1', min_value=0.0, step=0.01)
        tweet_sentiment2 = st.number_input('Tweet Sentiment 2', min_value=0.0, step=0.01)
        tweet_sentiment3 = st.number_input('Tweet Sentiment 3', min_value=0.0, step=0.01)
        tweet_sentiment4 = st.number_input('Tweet Sentiment 4', min_value=0.0, step=0.01)
        tweet_sentiment5 = st.number_input('Tweet Sentiment 5', min_value=0.0, step=0.01)
        tweet_sentiment_impact1 = st.number_input('Tweet Sentiment Impact 1', min_value=0.0, step=0.01)
        tweet_sentiment_impact2 = st.number_input('Tweet Sentiment Impact 2', min_value=0.0, step=0.01)
        tweet_sentiment_impact3 = st.number_input('Tweet Sentiment Impact 3', min_value=0.0, step=0.01)
        tweet_sentiment_impact4 = st.number_input('Tweet Sentiment Impact 4', min_value=0.0, step=0.01)
        tweet_sentiment_impact5 = st.number_input('Tweet Sentiment Impact 5', min_value=0.0, step=0.01)
        social_score = st.number_input('Social Score', min_value=0.0, step=0.01)
        average_sentiment = st.number_input('Average Sentiment', min_value=0.0, step=0.01)
        news = st.number_input('News', min_value=0.0, step=0.01)
        price_score = st.number_input('Price Score', min_value=0.0, step=0.01)
        social_impact_score = st.number_input('Social Impact Score', min_value=0.0, step=0.01)
        correlation_rank = st.number_input('Correlation Rank', min_value=0.0, step=0.01)
        galaxy_score = st.number_input('Galaxy Score', min_value=0.0, step=0.01)
        volatility = st.number_input('Volatility', min_value=0.0, step=0.01)
        market_cap_rank = st.number_input('Market Cap Rank', min_value=0.0, step=0.01)
        percent_change_24h_rank = st.number_input('Percent Change 24H Rank', min_value=0.0, step=0.01)
        volume_24h_rank = st.number_input('Volume 24H Rank', min_value=0.0, step=0.01)
        social_volume_24h_rank = st.number_input('Social Volume 24H Rank', min_value=0.0, step=0.01)
        social_score_24h_rank = st.number_input('Social Score 24H Rank', min_value=0.0, step=0.01)
        social_volume = st.number_input('Social Volume', min_value=0.0, step=0.01)
        percent_change_24h = st.number_input('Percent Change 24H', min_value=0.0, step=0.01)
        market_cap_global = st.number_input('Market Cap Global', min_value=0.0, step=0.01)
        
        submit_manual_button = st.form_submit_button(label='Predict (Manual Input)')

def preprocess_input(input_data, is_manual=True):
    # Ensure required columns are in the input data
    input_df = pd.DataFrame([input_data]) if is_manual else input_data
    input_df = input_df.reindex(columns=required_columns, fill_value=0)

    # Scaling the input data
    input_df_scaled = pd.DataFrame(scaler.fit_transform(input_df), columns=required_columns)
    return input_df_scaled

if input_method == "Upload CSV/Excel" and input_df is not None:
    # Preprocess and predict
    processed_data = preprocess_input(input_df, is_manual=False)
    predictions = model.predict(processed_data)
    input_df['Predicted Closing Price'] = predictions
    st.write("Predictions made for uploaded data:")
    st.write(input_df)

    # Download button for the prediction results
    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Predictions')
        writer.close()
        processed_data = output.getvalue()
        return processed_data

    df_xlsx = to_excel(input_df)
    st.download_button(label='📥 Download Predictions as Excel', data=df_xlsx, file_name='predictions.xlsx')

elif input_method == "Manual Input" and submit_manual_button:
    # Collect manual input values
    input_data = {
        'open': open,
        'high': high,
        'low': low,
        'volume': volume,
        'market_cap': market_cap,
        'url_shares': url_shares,
        'unique_url_shares': unique_url_shares,
        'reddit_posts': reddit_posts,
        'reddit_posts_score': reddit_posts_score,
        'reddit_comments': reddit_comments,
        'reddit_comments_score': reddit_comments_score,
        'tweets': tweets,
        'tweet_spam': tweet_spam,
        'tweet_followers': tweet_followers,
        'tweet_quotes': tweet_quotes,
        'tweet_retweets': tweet_retweets,
        'tweet_replies': tweet_replies,
        'tweet_favorites': tweet_favorites,
        'tweet_sentiment1': tweet_sentiment1,
        'tweet_sentiment2': tweet_sentiment2,
        'tweet_sentiment3': tweet_sentiment3,
        'tweet_sentiment4': tweet_sentiment4,
        'tweet_sentiment5': tweet_sentiment5,
        'tweet_sentiment_impact1': tweet_sentiment_impact1,
        'tweet_sentiment_impact2': tweet_sentiment_impact2,
        'tweet_sentiment_impact3': tweet_sentiment_impact3,
        'tweet_sentiment_impact4': tweet_sentiment_impact4,
        'tweet_sentiment_impact5': tweet_sentiment_impact5,
        'social_score': social_score,
        'average_sentiment': average_sentiment,
        'news': news,
        'price_score': price_score,
        'social_impact_score': social_impact_score,
        'correlation_rank': correlation_rank,
        'galaxy_score': galaxy_score,
        'volatility': volatility,
        'market_cap_rank': market_cap_rank,
        'percent_change_24h_rank': percent_change_24h_rank,
        'volume_24h_rank': volume_24h_rank,
        'social_volume_24h_rank': social_volume_24h_rank,
        'social_score_24h_rank': social_score_24h_rank,
        'social_volume': social_volume,
        'percent_change_24h': percent_change_24h,
        'market_cap_global': market_cap_global
    }

    # Preprocess the manual input data
    input_df_scaled = preprocess_input(input_data, is_manual=True)
    
    # Make predictions
    predictions = model.predict(input_df_scaled)
    st.write("Predictions:", predictions)
