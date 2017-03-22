# item based collaborative filtering 

import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('u.data', sep='\t', names=r_cols, usecols=range(3))
m_cols = ['movie_id', 'title']
movies = pd.read_csv('u.item', sep='|', names=m_cols, usecols=range(2))
ratings = pd.merge(movies, ratings)
ratings.head()

# creating matrix
userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()

# making correlation matrix
corrMatrix = userRatings.corr()
corrMatrix.head()

# removing the rating which have less than 100 ratings
corrMatrix = userRatings.corr(method='pearson', min_periods=100)
corrMatrix.head()

# adding a fake user which is already added in the data set
myRatings = userRatings.loc[0].dropna()
myRatings

# multiply with the correlation matrix so high rating gives the good result
# and sorting the result data
simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print "Adding sims for " + myRatings.index[i] + "..."
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)
    
#Glance at our results so far:
print "sorting..."
simCandidates.sort_values(inplace = True, ascending = False)
#print simCandidates.head(10)

# removing duplicates
simCandidates = simCandidates.groupby(simCandidates.index).sum()
simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)

# removing my rated and watched movies 
filteredSims = simCandidates.drop(myRatings.index)
print filteredSims.head(10)

