import os
os.chdir('YOURPATH')    # Set working directory
os. getcwd() 
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from wordcloud import WordCloud,STOPWORDS

warnings.filterwarnings("ignore")


netflix_dataset = pd.read_csv('netflix_titles.csv')

netflix_dataset.info()
# Identify the unique values
dict = {}
for i in list(netflix_dataset.columns):
    dict[i] = netflix_dataset[i].value_counts().shape[0]

print(pd.DataFrame(dict, index=["Unique counts"]).transpose())
#Identify the missing values

temp = netflix_dataset.isnull().sum()
uniq = pd.DataFrame({'Columns': temp.index, 'Numbers of Missing Values': temp.values})
uniq
#Analysis of Movies vs TV Shows

netflix_shows=netflix_dataset[netflix_dataset['type']=='TV Show']
netflix_movies=netflix_dataset[netflix_dataset['type']=='Movie']

plt.figure(figsize=(8,6))
ax= sns.countplot(x = "type", data = netflix_dataset,palette="Set1")
ax.set_title("TV Shows VS Movies")
#plt.show()
plt.savefig('barcharttvmovies.png')
# This shows that there are more Movies than TV Shows on Netflix

netflix_date= netflix_shows[['date_added']].dropna()
netflix_date['year'] = netflix_date['date_added'].apply(lambda x: x.split(',')[-1])
netflix_date['month'] = netflix_date['date_added'].apply(lambda x: x.split(' ')[0])
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'] #::-1 just reverse this nigga

df = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0)[month_order].T
plt.subplots(figsize=(10,10))
sns.heatmap(df,cmap='Blues') #heatmap
plt.savefig("heatmapyear.png")
#plt.show()
# This heatmap shows frequencies of TV shows added to Netflix throughout the years.

# Year wise analysis

Last_fifteen_years = netflix_dataset[netflix_dataset['release_year']>2005 ]
Last_fifteen_years.head()
#Year wise analysis in graph
plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(y="release_year", data=Last_fifteen_years, palette="Set2", order=netflix_dataset['release_year'].value_counts().index[0:15])
#plt.show()
plt.savefig('releaseyearcount.png')

# Analysis of duration of TV shows

features=['title','duration']
durations= netflix_shows[features]
durations['no_of_seasons']=durations['duration'].str.replace(' Season','')
durations['no_of_seasons']=durations['no_of_seasons'].str.replace('s','')

durations['no_of_seasons']=durations['no_of_seasons'].astype(str).astype(int)
#TV shows with largest number of seasons
t=['title','no_of_seasons']
top=durations[t]

top=top.sort_values(by='no_of_seasons', ascending=False)

top20=top[0:20]
print(top20)
plt.figure(figsize=(80,60))
top20.plot(kind='bar',x='title',y='no_of_seasons', color='blue')
plt.savefig('tvshowsmaxseasons.png')
new_df = netflix_dataset['description']
words = ' '.join(new_df)
cleaned_word = " ".join([word for word in words.split()
                                                     ])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word)
plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('netflixwordcloud.png')
#Filling null values with empty string.
filledna=netflix_dataset.fillna('')
filledna.head()

#Cleaning the data - making all the words lower case
def clean_data(x):
        return str.lower(x.replace(" ", ""))

#Identifying features on which the model is to be filtered.
features=['title','director','cast','listed_in','description']
filledna=filledna[features]

for feature in features:
    filledna[feature] = filledna[feature].apply(clean_data)

filledna.head()

def create_soup(x):
    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']

filledna['soup'] = filledna.apply(create_soup, axis=1)
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filledna['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of our main DataFrame and construct reverse mapping as before
filledna=filledna.reset_index()
indices = pd.Series(filledna.index, index=filledna['title'])
def get_recommendations_new(title, cosine_sim = cosine_sim2):
    title=title.replace(' ','').lower()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return netflix_dataset['title'].iloc[movie_indices]
recommendations = get_recommendations_new('NCIS', cosine_sim2)
print(recommendations)
country=df["country"]
country=country.dropna()

country=", ".join(country)
country=country.replace(',, ',', ')


country=country.split(", ")
country= list(Counter(country).items())
country.remove(('Vatican City', 1))
country.remove(('East Germany', 1))
print(country)
max_show_country=country[0:11]
max_show_country = pd.DataFrame(max_show_country) 
max_show_country= max_show_country.sort_values(1)

fig, ax = plt.subplots(1, figsize=(8, 6))
fig.suptitle('Plot of country vs shows')
ax.barh(max_show_country[0],max_show_country[1],color='blue')
plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.savefig('plotcountryshow.png')
df1=pd.read_csv('country_code.csv')
df1=df1.drop(columns=['Unnamed: 2'])
df1.head()
country_map = pd.DataFrame(country) 
country_map=country_map.sort_values(1,ascending=False)
location = pd.DataFrame(columns = ['CODE']) 
search_name=df1['COUNTRY']

for i in country_map[0]:
    x=df1[search_name.str.contains(i,case=False)] 
    x['CODE'].replace(' ','')
    location=location.append(x)


print(location)
locations=[]
temp=location['CODE']
for i in temp:
    locations.append(i.replace(' ',''))
genre=df["listed_in"]
genre=", ".join(genre)
genre=genre.replace(',, ',', ')
genre=genre.split(", ")
genre= list(Counter(genre).items())
print(genre)

max_genre=genre[0:11]
max_genre = pd.DataFrame(max_genre) 
max_genre= max_genre.sort_values(1)

plt.figure(figsize=(8,6))
plt.xlabel('COUNT')
plt.ylabel('GENRE')
plt.barh(max_genre[0],max_genre[1], color='red')
df = df.dropna(how='any',subset=['cast', 'director'])
df = df.dropna()
df["date_added"] = pd.to_datetime(df['date_added'])
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month
df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)
df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)
