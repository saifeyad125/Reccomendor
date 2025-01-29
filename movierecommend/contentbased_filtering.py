#for this one, we will use the overview of the movies and use TFIDF for text processing:
#there is  a built in TFIDF vectorizer in scikit learn:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from setup import pd, np, plt, load_data


dataframe = load_data()

tfidf = TfidfVectorizer(stop_words='english')

#now we will replace any null strings with ''
dataframe['overview'] = dataframe['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(dataframe['overview'])

#print(tfidf_matrix.shape)

#we will use the cosine similarity to calculate the similarity between the movies:
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#now we will make a reverse mapping of movie titles and indices:
indices = pd.Series(dataframe.index, index=dataframe['title']).drop_duplicates()

def get_reccomendations(title, cosine_sim=cosine_sim):
    #get the index that matches the title:
    idx = indices[title]

    #get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    #sort the movies based on the similarity scores (lambda accesses the second element which is the score)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #get the scores of the 10 most similar movies (starting from the second movie)
    sim_scores = sim_scores[1:11]

    #get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    #return the top 10 most similar movies
    return dataframe['title'].iloc[movie_indices]


#print(get_reccomendations('The Dark Knight Rises'))