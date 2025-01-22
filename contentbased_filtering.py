from setup import load_data, show_dataframe_popup, pd, np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import gc  # for garbage collection

dataframe = load_data()

# Use a smaller max_features value
tfidf = TfidfVectorizer(stop_words='english', max_features=50)

dataframe['Translated_Review'] = dataframe['Translated_Review'].fillna('')
dataframe['Sentiment'] = dataframe['Sentiment'].fillna('Neutral')

# Instead of creating separate matrices, process all reviews together
grouped_data = dataframe.groupby('App').agg({
    'Translated_Review': lambda x: ' '.join(x),
    'Sentiment': lambda x: list(x)  # Keep sentiments as a list for potential weighting
}).reset_index()

# Create TF-IDF matrix once
tfidf_matrix = tfidf.fit_transform(grouped_data['Translated_Review'])

# Calculate similarity matrix
print("Calculating similarity matrix...")
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create reverse mapping
indices = pd.Series(grouped_data.index, index=grouped_data['App']).drop_duplicates()

def get_recommendations(app_name, cosine_sim=cosine_sim):
    try:
        idx = indices[app_name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Get top 5 similar games
        app_indices = [i[0] for i in sim_scores]
        return grouped_data['App'].iloc[app_indices]
    except KeyError:
        return f"Game '{app_name}' not found in the database."

# Test the recommendations
print("\nRecommendations for Candy Crush Saga:")
print(get_recommendations('Candy Crush Saga'))





# Clean up to free memory
del tfidf_matrix
del cosine_sim
gc.collect()