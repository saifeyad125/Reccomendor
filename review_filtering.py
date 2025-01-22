from setup import load_data, show_dataframe_popup, pd, np

dataframe = load_data()

#firstly i want to show the top 10 best games based on the positive reviews

#there is #ofreviews, sentiment, sentiment polarity, and sentiment subjectivity I will use all 4 to determine the best games

#for score i will use the equation: 0.5x num of reviews + 0.25 x sentiment + 0.2x sentiment polarity + 0.05x sentiment subjectivity


dataframe['Translated_Review'] = dataframe['Translated_Review'].fillna('')
dataframe['Sentiment'] = dataframe['Sentiment'].fillna('Neutral')
dataframe['Sentiment_Polarity'] = dataframe['Sentiment_Polarity'].fillna(0)
dataframe['Sentiment_Subjectivity'] = dataframe['Sentiment_Subjectivity'].fillna(0.5)

grouped_data = dataframe.groupby('App').agg({
    'Sentiment': lambda x: ', '.join(x),  # Combine all sentiments into a string
    'Sentiment_Polarity': 'mean',         # Average polarity
    'Sentiment_Subjectivity': 'mean',     # Average subjectivity
    'Reviews': 'max'                   # Total number of reviews
}).reset_index()

# Rename columns for better clarity
grouped_data.rename(columns={
    'Sentiment': 'Combined_Sentiments',
    'Sentiment_Polarity': 'Average_Polarity',
    'Sentiment_Subjectivity': 'Average_Subjectivity',
    'Reviews': 'Total_Reviews'
}, inplace=True)

#ensure that the total reviews is a number
grouped_data['Total_Reviews'] = pd.to_numeric(grouped_data['Total_Reviews'], errors='coerce').fillna(0)


def calculate_score(row):
    return 0.5 * row['Total_Reviews'] + 0.25 * row['Average_Polarity'] + 0.2 * row['Average_Subjectivity']

grouped_data['Score'] = grouped_data.apply(calculate_score, axis=1)

# Sort by score in descending order
grouped_data = grouped_data.sort_values('Score', ascending=False)


#show the top 10 games
show_dataframe_popup(grouped_data.head(10))