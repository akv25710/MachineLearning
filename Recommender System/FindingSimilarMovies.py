# downloading the raw dataset from site and making recommendation system

import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('u.data', sep='\t', names=r_cols, usecols=range(3))
m_cols = ['movie_id', 'title']
movies = pd.read_csv('u.item', sep='|', names=m_cols, usecols=range(2))
# importing and create table
ratings = pd.merge(movies, ratings)

ratings.head()

# making matrix to find correlation
movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
movieRatings.head()

# getting rating of a perticular movie
starWarsRatings = movieRatings['Star Wars (1977)']
starWarsRatings.head()

# finding correlation with a perticular movie
similarMovies = movieRatings.corrwith(starWarsRatings)
# removing the null values ( or NaN )
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
df.head(10)

# sort with according with high to low correlation
similarMovies.sort_values(ascending=False)
# this recommendation fails 
# give perfect correlation to movies according to very few peoples

# checking size and mean to get the reviews from many peoples 
import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
movieStats.head()

# minimum 100 ratings to take participate in the recommendation system
popularMovies = movieStats['rating']['size'] >= 100
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]

# join with the new system
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
# show now the system
df.head()

# sort according to the maximum correlation 
# factors : mean rating = mean of all the ratings
#         : rating size = no of people rated
#         : similarity = correlation between the movies
print df.sort_values(['similarity'], ascending=False)[1:15]

