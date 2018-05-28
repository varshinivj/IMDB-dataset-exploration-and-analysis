#Project data visualisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *

#importing the dataset
df = pd.read_csv("IMDB.csv")
df
df.columns

#dropping recurring or unecesssary columns
df = df.drop(['color','genres2','genres3','genres4','genres5','genres6','genres7','Unnamed: 38','plot_keywords1', 'plot_keywords2',
       'plot_keywords3', 'plot_keywords4', 'plot_keywords5','movie_imdb_link'], axis=1)

#Distribution of IMDB ratings
sns.set()
_ = plt.hist(df['imdb_score'])
_ = plt.xlabel('IMBD scores')
_ = plt.ylabel('Frequency')
_ = plt.title('Distribution of IMDB rating')

#IMDB score vs Country
plt.figure(figsize=(20,20))
plt.xticks(rotation = 'vertical')
ax = sns.boxplot(x='country',y='imdb_score',data=df)

plt.show()

#IMDB score vs Movie year
plt.figure(figsize=(20,20))
plt.xticks(rotation = 'vertical')

ax = sns.boxplot(x='title_year',y='imdb_score',data=df)
plt.show()

#IMDB score vs movie facebook popularity
plt.figure(figsize=(20,10))
sns.regplot(x='imdb_score',y='movie_facebook_likes', data=df,fit_reg=False)
plt.show()

#IMDB score vs director facebook popularity
#removing null director names
df[pd.notnull(df['director_name'])]
plt.figure(figsize=(20,10))
sns.regplot(x='imdb_score',y='director_facebook_likes', data=df,fit_reg=False)
plt.show()

print "\n\n\n\n\n\n"


#Project Linear Regression
import pandas as pd 
import statsmodels as s
import numpy as ny
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import os 
import math as math

z=pd.read_csv("IMDB.csv")
z=z[pd.notnull(z['director_name'])]
z=z[pd.notnull(z['gross'])]
lm1=sm.ols(formula="imdb_score~language", data=z).fit()
lm1.params
print(lm1.summary())
lm=sm.ols(formula="imdb_score~color",data=z).fit()
lm.params
print(lm.summary())
z.gross=z.gross/1000000
lm2=sm.ols(formula="imdb_score~gross",data=z).fit()
lm2.params
print(lm2.summary())
lm3=sm.ols(formula="imdb_score~content_rating",data=z).fit()
lm3.params
print(lm3.summary())
lm4=sm.ols(formula="imdb_score~num_critic_for_reviews",data=z).fit()
lm4.params
print(lm4.summary())
lm5=sm.ols(formula="imdb_score~genres1",data=z).fit()
lm5.params
print(lm5.summary())
lm6=sm.ols(formula="imdb_score~country",data=z).fit()
lm6.params
print(lm6.summary())
x=z[['actor_2_facebook_likes',
'actor_1_facebook_likes',
'actor_3_facebook_likes',
'cast_total_facebook_likes' ,
'num_voted_users',
'director_facebook_likes',
'title_year']]
ml=sm.ols('imdb_score~x',data=z).fit()
ml.params
print(ml.summary())
c=z[['num_critic_for_reviews',
'num_voted_users',]]
ml1=sm.ols('imdb_score~c',data=z).fit()
ml1.params
print(ml1.summary())

print "\n\n\n\n\n\n"


#Project Random Forest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
from sklearn import *
from sklearn.metrics import *
from sklearn.ensemble import *
data = pd.read_csv("IMDB.csv")
data_model = data.drop(['director_name','actor_1_name','actor_2_name','actor_3_name','movie_title','genres2','genres3','genres4','genres5','genres6','genres7','Unnamed: 38','plot_keywords1', 'plot_keywords2',
       'plot_keywords3', 'plot_keywords4', 'plot_keywords5','movie_imdb_link'],axis = 1)
data_model = data_model.dropna()
data_X = data_model.drop(['imdb_score','content_rating','color','language','country','genres1'],axis = 1)
data_Y = data_model.iloc[:,17]

data_X_content_rating = pd.get_dummies(data_model['content_rating'])
data_X = data_X.join(data_X_content_rating)

data_X_color = pd.get_dummies(data_model['color'])
data_X = data_X.join(data_X_color)

data_X_language = pd.get_dummies(data_model['language'])
data_X = data_X.join(data_X_language)

data_X_country = pd.get_dummies(data_model['country'])
data_X = data_X.join(data_X_country)

data_X_genres1 = pd.get_dummies(data_model['genres1'])
data_X = data_X.join(data_X_genres1)


rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
data_X1 = data_X.astype(int)
data_Y1 = data_Y.astype(int)
rf.fit(data_X1, data_Y1)
rf_predict = rf.predict(data_X1)
accuracy = accuracy_score(data_Y1, rf_predict)
print 'Mean squared error:', mean((rf_predict-data_Y)**2)
print 'Root Mean squared error:', math.sqrt(mean((rf_predict-data_Y)**2))
for name, importance in zip(list(data_X1), rf.feature_importances_):
    print name, "=", importance
features = list(data_X1)
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features)
plt.xlabel('Relative Importance')
plt.show()

print "\n\n\n\n\n\n"

#Project Text Mining
from nltk.tokenize import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import enchant
import csv
dataX_text = data.ix[:,10]
dataY_text = data.ix[:,23]
d = enchant.Dict("en_US")

title = []
for element in dataX_text:
    title.append(element.split('??')[0])

scores = []
for element in dataY_text:
    scores.append(element)

tokens = []
for element in title:
        tokens.append(word_tokenize(element.lower()))

StopWords = stopwords.words("english")
filtered_sentence = [w for w in tokens if not w in StopWords]
filtered_sentence = []
for w in tokens:
    u = []
    for v in w:
        if v not in StopWords:
            if len(v) >= 3:
                u.append(v)
    filtered_sentence.append(u)

ps = PorterStemmer()
stemming_sentence = []
for w in filtered_sentence:
    u = []
    for v in w:
        u.append(str(ps.stem(v)))
    stemming_sentence.append(u)

text_file = []
for item in stemming_sentence:
    text_file.append(str(item).strip("['").strip("']").replace("', '",' '))

vectorizer = TfidfVectorizer(min_df=1)
vectorizer.fit_transform(text_file)
name1 = vectorizer.get_feature_names()
words = []
for element in name1:
    words.append(str(element).strip("u'").strip("'"))

TFIDF_matrix = vectorizer.fit_transform(text_file).toarray()
TFIDF_sum = TFIDF_matrix.sum(axis=0)
TFIDF = TFIDF_sum.tolist()
TFIDF_indices = np.argsort(-TFIDF_sum)

words_name = []
for element in TFIDF_indices:
    words_name.append(words[element])

TFIDF_value = []
for element in TFIDF_indices:
    TFIDF_value.append(TFIDF[element])

TFIDF_list = zip(words_name,TFIDF_value)
print pd.DataFrame(TFIDF_list[:10], columns = ['Words', 'TFIDF'])
print pd.DataFrame(TFIDF_list[-10:], columns = ['Words', 'TFIDF'])

indices1 = [i for i, s in enumerate(words) if 'man' in s]
indices2 = [i for i, s in enumerate(words) if 'love' in s]
indices3 = [i for i, s in enumerate(words) if 'day' in s]
indices4 = [i for i, s in enumerate(words) if 'dead' in s]
indices5 = [i for i, s in enumerate(words) if 'movi' in s]

score_sum1 = 0
for element in indices1:
    score_sum1 = score_sum1 + scores[element]

for element in indices2:
    score_sum1 = score_sum1 + scores[element]

for element in indices3:
    score_sum1 = score_sum1 + scores[element]

print "Average Rating of Top Three Frequency Words:", score_sum1/(len(indices1)+len(indices2)+len(indices3))

for element in indices4:
    score_sum1 = score_sum1 + scores[element]

for element in indices5:
    score_sum1 = score_sum1 + scores[element]
    
print "Average Rating of Top Five Frequency Words:", score_sum1/(len(indices1)+len(indices2)+len(indices3)+len(indices4)+len(indices5))

indices6 = [i for i, s in enumerate(words) if 'colon' in s]
indices7 = [i for i, s in enumerate(words) if 'theater' in s]
indices8 = [i for i, s in enumerate(words) if 'alan' in s]
indices9 = [i for i, s in enumerate(words) if 'smithe' in s]
indices10 = [i for i, s in enumerate(words) if 'felt' in s]

score_sum2 = 0
for element in indices6:
    score_sum2 = score_sum2 + scores[element]

for element in indices7:
    score_sum2 = score_sum2 + scores[element]

for element in indices8:
    score_sum2 = score_sum2 + scores[element]

print "Average Rating of Least Three Frequency Words:", score_sum2/(len(indices6)+len(indices7)+len(indices8))

for element in indices9:
    score_sum2 = score_sum2 + scores[element]

for element in indices10:
    score_sum2 = score_sum2 + scores[element]
    
print "Average Rating of Least Five Frequency Words:", score_sum2/(len(indices6)+len(indices7)+len(indices8)+len(indices9)+len(indices10))




