import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Local file paths
DATA_FILE_PATH = "processed_audible_data(1).csv"
MODEL_FILE_PATH = "book_recommender_models_dbscan_lsa.pkl"

# Load the model from local file
@st.cache_resource
def load_model():
    try:
        with open(MODEL_FILE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load dataset from local CSV
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        # Clean the data
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df.dropna(subset=['Author', 'Book Name'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

def main():
    st.title('📚 Audible Book Recommendation System')
    
    # Load data and model
    model_data = load_model()
    df = load_data()

    # Sidebar filters
    st.sidebar.header('Your Preferences')
    min_rating = st.sidebar.slider('Minimum Rating (out of 5):', 1.0, 5.0, 4.0, 0.1)
    max_price = st.sidebar.slider('Maximum Price (in $):', 0.0, 500.0, 50.0)

    # Book recommendations based on filters
    if st.button('Recommend Books'):
        filtered_books = df[(df['Rating'] >= min_rating) & (df['Price'] <= max_price)]
        if not filtered_books.empty:
            st.subheader("Recommended Books")
            for _, book in filtered_books.iterrows():
                with st.expander(f"{book['Book Name']} - {book['Author']}"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**Rating:** ⭐ {float(book['Rating']):.1f}")
                        st.write(f"**Price:** ${book['Price']:.2f}")
                    with col2:
                        st.write(f"**Description:** {book.get('Description', 'No description available')}")
        else:
            st.warning("No books found matching your criteria. Please adjust your filters.")

    st.markdown('---')
    st.header("Explore Books")

    # Search options
    search_method = st.radio("How would you like to find books?", ["Search by Book Name", "Browse All Books"])
    
    if search_method == "Search by Book Name":
        available_books = df['Book Name'].dropna().unique().tolist()
        book_name = st.selectbox('Search for a book:', available_books)
        
        if st.button('Find Similar Books'):
            try:
                book_idx = df[df['Book Name'] == book_name].index[0]
                features = model_data['features']
                
                # Handle both dense and sparse matrices
                similarities = cosine_similarity(
                    features[book_idx:book_idx+1] if isinstance(features, np.ndarray) else features[book_idx:book_idx+1].toarray(),
                    features if isinstance(features, np.ndarray) else features.toarray()
                )
                
                similar_indices = similarities.argsort()[0][-6:-1][::-1]
                similar_books = df.iloc[similar_indices]

                for _, book in similar_books.iterrows():
                    with st.expander(f"{book['Book Name']} by {book['Author']}"):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.write(f"**Rating:** ⭐ {float(book['Rating']):.1f}")
                            st.write(f"**Price:** ${book['Price']:.2f}")
                        with col2:
                            st.write(f"**Description:** {book.get('Description', 'No description available')}")
            except Exception as e:
                st.error(f"Error finding similar books: {str(e)}")
    else:
        # Browse top rated books
        filtered_books = df[df['Rating'] >= min_rating].sort_values('Rating', ascending=False)
        st.subheader('Top Rated Books')
        for _, book in filtered_books.head(10).iterrows():
            with st.expander(f"{book['Book Name']} by {book['Author']}"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**Rating:** ⭐ {float(book['Rating']):.1f}")
                    st.write(f"**Price:** ${book['Price']:.2f}")
                with col2:
                    st.write(f"**Description:** {book.get('Description', 'No description available')}")

if __name__ == "__main__":
    main()
