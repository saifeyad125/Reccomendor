from setup import pd, np, plt, load_data

dataframe = load_data()

#now we will use imdb's weighted rating formula to construct the recommendation system:
C= dataframe['vote_average'].mean()
m= dataframe['vote_count'].quantile(0.9)

#now we will filter out the movies which qualify for this:
q_movies = dataframe.copy().loc[dataframe['vote_count'] >= m]




def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

#now we will make a new feature called score
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#now we will sort the movies based on score:
q_movies = q_movies.sort_values('score', ascending=False)

#finally, we will print the first 15 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))


pop= dataframe.sort_values('popularity', ascending=False)
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.show()