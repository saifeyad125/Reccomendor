from setup import load_data, show_dataframe_popup, pd, np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dataframe = load_data()

# Preprocessing
dataframe['Translated_Review'] = dataframe['Translated_Review'].fillna('')

# Group reviews by game
grouped_data = dataframe.groupby('App')['Translated_Review'].apply(' '.join).reset_index()

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(grouped_data['Translated_Review'])

# Create mapping of game names to indices
indices = pd.Series(grouped_data.index, index=grouped_data['App']).drop_duplicates()

def get_recommendations(keywords):
    try:
        # Process keywords
        words = [word.strip().lower() for word in keywords.split(',')]
        words = ' '.join(words)
        
        # Transform keywords into TF-IDF vector
        words_matrix = tfidf.transform([words])
        
        # Calculate similarity
        similarity_scores = cosine_similarity(words_matrix, tfidf_matrix)[0]
        
        # Get similar games
        sim_scores = list(enumerate(similarity_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[:5]  # Get top 5 similar games
        
        # Get game indices
        app_indices = [i[0] for i in sim_scores]
        
        return grouped_data['App'].iloc[app_indices]
    
    except Exception as e:
        return f"Error processing keywords: {str(e)}"


df1 = pd.read_csv('/Users/saif/Desktop/University Saif/aldar internship/vscodealdar/googleplaystore/archive/googleplaystore.csv')
print(df1['Last Updated'].max())

# Test the recommendations
print("\nRecommendations for keywords 'scary, , multiplayer':")
recommendations = get_recommendations('scary, puzzle, monsters')
print(recommendations)