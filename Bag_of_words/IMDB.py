#coding=utf-8
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


def review_to_text(review, remove_stopwords):
    raw_text = BeautifulSoup(review, 'lxml').get_text() #去掉html标记
    letters = re.sub('[^a-zA-Z]','',raw_text) #去掉非字母字符
    words = letters.lower().split()
    if remove_stopwords:  #remove_stopwords 是boolean类型，是否去掉停用词
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    return words #返回每条评论经过三项处理的词汇列表

train = pd.read_csv('../data/labeledTrainData1.tsv', delimiter='\t')
test = pd.read_csv('../data/testData1.tsv', delimiter='\t')
# print train.head()
# print test.head()
X_train=[]
for review in train['review']:
    X_train.append(''.join(review_to_text(review, True)))
X_test=[]
for review in test['review']:
    X_test.append(''.join(review_to_text(review, True)))
y_train = train['sentiment']
pip_count = Pipeline([('count_vec', CountVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
params_count = {'count_vec_binary':[True, False], 'count_vec_ngram_range':[(1, 1), (1, 2)], 'mnb_alpha':[0.1, 1.0, 10.0]} #参数意思查看sklearn官网
params_tfidf = {'tfidf_vec_binary':[True, False], 'tifid_vec_ngram_range':[(1, 1), (1, 2)], 'mnb_alpha':[0.1, 1.0, 10.0]}
gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=-1, verbose=1)
gs_count.fit(X_train, y_train)
print gs_count.best_score_
print gs_count.best_params_
count_y_predict= gs_count.predict(X_test)
gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv=4, n_jobs=-1, verbose=1)
gs_tfidf.fit(X_train, y_train)
tfidf_y_predict = gs_tfidf.predict(X_test)

submission_count = pd.DataFrame({'id':test['id'],'sentiment':count_y_predict})
submission_tfidf = pd.DataFrame({'id':test['id'],'sentiment':tfidf_y_predict})
