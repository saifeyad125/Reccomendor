from ast import literal_eval
from setup import pd, np, plt, load_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from contentbased_filtering import get_reccomendations



dataframe = load_data()
features = ['genres', 'keywords', 'cast', 'crew']


#right now, these contain stringfield lists, so we want to convert them to actual lists
for feature in features:
    dataframe[feature] = dataframe[feature].apply(literal_eval)



def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
        #if its null we return null (NaN)
        return 'Unknown'

def get_name_list(x):
    #check if its a list type
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #if there are more than 3 names, we will return the first 3 only
        if len(names) > 3:
            names = names[:3]
        return names
    return []


dataframe['director'] = dataframe['crew'].apply(get_director)
features = ['genres', 'keywords', 'cast']
for feature in features:
    dataframe[feature] = dataframe[feature].apply(get_name_list)


#now we will remove all spaces and make everythign lowercase

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


features = ['cast', 'keywords', 'director', 'genres']

for features in features:
    dataframe[feature] = dataframe[feature].apply(clean_data)


#now we can create the metadata soup which we will feed into the vectorizor
def create_soup(x):
    keywords = ' '.join(x['keywords']) if isinstance(x['keywords'], list) else ''
    cast = ' '.join(x['cast']) if isinstance(x['cast'], list) else ''
    director = str(x['director']) if pd.notna(x['director']) else ''
    genres = ' '.join(x['genres']) if isinstance(x['genres'], list) else ''
    
    return keywords + ' ' + cast + ' ' + director + ' ' + genres

dataframe['soup'] = dataframe.apply(create_soup, axis=1)


#we will use countvectorizer instead of tfidf

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(dataframe['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

#reverse map like before
dataframe = dataframe.reset_index()
indices = pd.Series(dataframe.index, index=dataframe['title'])


#now we can use the getreommendations function from the contentbased filtering file
print(get_reccomendations('The Godfather', cosine_sim2))