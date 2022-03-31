import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('train_stances_modified.csv')

# df = df.dropna()

x = df.drop('Stance', axis=1)

y = df['Stance']

messages = x.copy()

# unnecessary
messages.reset_index(inplace=True)
print(messages)

nltk.download('stopwords')

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['Headline'][i])
    review = review.lower()
    review = review.split()

    review_new = []
    for word in review:
        if not word in stopwords.words('english'):
            review_new.append(ps.stem(word))

    review_new = ' '.join(review_new)
    corpus.append(review_new)

# print(corpus)

voc_size = 6000
onehot_repr = [one_hot(words,voc_size)for words in corpus]
# print(onehot_repr)

sent_length=20
embedded_docs = pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

len(embedded_docs)

## creating model
embedding_vector_features= 40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length)) # making embedding layer
model.add(LSTM(100))  # one LSTM Layer with 100 neurons
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


x_final= np.array(embedded_docs)
y_final= np.array(y)

print(x_final.dtype)
print(y_final.dtype)

x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,test_size=0.33,random_state=0)

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)
# y_pred= model.predict_classes(x_test)

