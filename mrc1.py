import pandas as pd
import numpy as np
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
movies=pd.read_csv("c:\\py.projects\\tmdb_5000_movies.csv")
credits=pd.read_csv("c:\\py.projects\\tmdb_5000_credits.csv")
movies=movies.merge(credits,on='title')
movies.dropna(inplace=True)
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
def convert3(obj):
    L3=[]
    countr=0
    for i in ast.literal_eval(obj):
        if countr != 3:
          L3.append(i['name'])
          countr=countr+1
        else:
            break  
    return L3
def fetch_director(obj):
    FD=[]
    for i in ast.literal_eval(obj):
      if i['job']=='Director':
        FD.append(i['name'])
        break
    return FD
movies['genres']=movies['genres'].apply(convert)  
movies['keywords']=movies['keywords'].apply(convert)  
movies['cast']=movies['cast'].apply(convert3)
movies['crew']=movies['crew'].apply(fetch_director)
movies['overview']=movies['overview'].apply(lambda x:x.split())
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies['tags']=movies['overview']+movies['cast']+movies['crew']+movies['genres']+movies['keywords']
new_df=movies[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
# print(new_df['tags'][0])
# we are going to convert text to vectors
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
ps = PorterStemmer()
def stem(text):
   y=[]
   for i in text.split():
      y.append(ps.stem(i))
      z=" ".join(y)
   return z   
new_df['tags']=new_df['tags'].apply(stem)
vectors=cv.fit_transform(new_df['tags']).toarray()
w=cv.get_feature_names_out()
similarity=cosine_similarity(vectors)
def recommend(movie):
   movie_index= new_df[new_df['title']==movie].index[0]
   distances= similarity[movie_index]
   movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
   for i in movies_list:
    zz= new_df.iloc[i[0]].title
   return zz 
recommend('Avatar')
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
