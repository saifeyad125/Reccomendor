#based on a game they present, i will give them the top 5 games that are similar to the game they presented
#i will use the reviews, TFIDF for text vectorization and cosine similarity to determine the similarity between the games

from setup import load_data, show_dataframe_popup, pd, np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel #using linear kernel cuz its faster

dataframe = load_data()

tfidf = TfidfVectorizer(stop_words='english', max_features=100)

dataframe['Translated_Review'] = dataframe['Translated_Review'].fillna('')
dataframe['Sentiment'] = dataframe['Sentiment'].fillna('Neutral')


#i want to make 3 grouped datasets, one for each sentiment
positive_data = dataframe[dataframe['Sentiment'] == 'Positive']
positive_data = positive_data.groupby('App').agg({
    'Translated_Review': lambda x: ' '.join(x) #the %*$# is just a random string that i will use to split the reviews later
}).reset_index()
negative_data = dataframe[dataframe['Sentiment'] == 'Negative']
negative_data = negative_data.groupby('App').agg({
    'Translated_Review': lambda x: ' '.join(x)
}).reset_index()
neutral_data = dataframe[dataframe['Sentiment'] == 'Neutral']
negative_data = negative_data.groupby('App').agg({
    'Translated_Review': lambda x: ' '.join(x)
}).reset_index()



#i want to compare each review with every other review that has the same sentiment 

tfidf_matrix_p = tfidf.fit_transform(positive_data['Translated_Review'])
tfidf_matrix_n = tfidf.fit_transform(negative_data['Translated_Review'])
tfidf_matrix_ne = tfidf.fit_transform(neutral_data['Translated_Review'])

#now i will find the cosine similarity between each review

cosine_sim_p = linear_kernel(tfidf_matrix_p, tfidf_matrix_p)
cosine_sim_n = linear_kernel(tfidf_matrix_n, tfidf_matrix_n)
cosine_sim_ne = linear_kernel(tfidf_matrix_ne, tfidf_matrix_ne)
print(cosine_sim_ne.shape, "shape")

#now i will get the average for the 3 cosine similarities
#we have to give them all the same shape first
def biggest_shape(matrix1, matrix2, matrix3):
    # Find the matrix with the largest dimensions
    if matrix1.shape[0] > matrix2.shape[0]:
        if matrix1.shape[0] > matrix3.shape[0]:
            return matrix1.shape, 1
        else:
            return matrix3.shape, 3
    else:
        if matrix2.shape[0] > matrix3.shape[0]:
            return matrix2.shape, 2
        else:
            return matrix3.shape, 3


# Find the biggest shape
biggest_shape, ind = biggest_shape(cosine_sim_p, cosine_sim_n, cosine_sim_ne)
print(f"Largest matrix shape: {biggest_shape}, {ind}")

# Pad matrices to the biggest shape
if ind == 1:  # Positive is the largest
    cosine_sim_n = np.pad(cosine_sim_n, ((0, biggest_shape[0] - cosine_sim_n.shape[0]),
                                         (0, biggest_shape[1] - cosine_sim_n.shape[1])), 'constant')
    cosine_sim_ne = np.pad(cosine_sim_ne, ((0, biggest_shape[0] - cosine_sim_ne.shape[0]),
                                           (0, biggest_shape[1] - cosine_sim_ne.shape[1])), 'constant')
elif ind == 2:  # Negative is the largest
    cosine_sim_p = np.pad(cosine_sim_p, ((0, biggest_shape[0] - cosine_sim_p.shape[0]),
                                         (0, biggest_shape[1] - cosine_sim_p.shape[1])), 'constant')
    cosine_sim_ne = np.pad(cosine_sim_ne, ((0, biggest_shape[0] - cosine_sim_ne.shape[0]),
                                           (0, biggest_shape[1] - cosine_sim_ne.shape[1])), 'constant')
else:  # Neutral is the largest
    cosine_sim_p = np.pad(cosine_sim_p, ((0, biggest_shape[0] - cosine_sim_p.shape[0]),
                                         (0, biggest_shape[1] - cosine_sim_p.shape[1])), 'constant')
    cosine_sim_n = np.pad(cosine_sim_n, ((0, biggest_shape[0] - cosine_sim_n.shape[0]),
                                         (0, biggest_shape[1] - cosine_sim_n.shape[1])), 'constant')
print(cosine_sim_p.shape, "pos")
print(cosine_sim_n.shape, "neg")

# Verify the shapes
print(f"Shapes after padding: {cosine_sim_p.shape}, {cosine_sim_n.shape}, {cosine_sim_ne.shape}")

# Combine the matrices
cosine_sim = (cosine_sim_p + cosine_sim_n + cosine_sim_ne) / 3


#now i will reverse map the indices to the app names

grouped_data = dataframe.groupby('App').agg({
    'Translated_Review': lambda x: ' '.join(x),
    'Sentiment': lambda x: ', '.join(x)
}).reset_index()

indices = pd.Series(positive_data.index, index=positive_data['App']).drop_duplicates()

def get_recommendations(app_name, cosine_sim=cosine_sim):
    idx = indices[app_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    app_indices = [i[0] for i in sim_scores]
    return grouped_data['App'].iloc[app_indices]


show_dataframe_popup(get_recommendations('Candy Crush Saga'))
#show_dataframe_popup(group_data.head(5))