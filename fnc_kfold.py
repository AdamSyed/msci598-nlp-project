import timeit
start = timeit.default_timer()

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import gen_or_load_tfidf_feats,get_tfidf_headlines, get_tfidf_bodies, refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.system import parse_params, check_version
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import Embedding, LSTM, Dropout, SimpleRNN, MaxPooling1D, Conv1D, Bidirectional
import sklearn.feature_extraction as feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Function to generate baseline features - provided to us in the baseline repo
def generate_features_baseline(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])


    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]

    return X,y

# Function to generate Tf-Idf features
def generate_features_tfidf(stances,dataset,name):
    h, b, y, n = [],[],[], []

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        n.append(stance['Body ID'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    # get the tfidf vectors for the headliens and bodies
    X_headlines = gen_or_load_tfidf_feats(get_tfidf_headlines, h, b)
    X_bodies = gen_or_load_tfidf_feats(get_tfidf_bodies, h, b)
    
    # create a vectorizer that is trained on the headlines AND bodies
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=6000).fit(h + b)

    # find the cosine similarity between a headline and it's body
    cos_d = []
    for i in range(len(X_overlap)):
        cos_d.append(cosine_similarity(vectorizer.transform([h[i]]),vectorizer.transform([b[i]])))

    # add all the features together to form the X array
    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, np.asarray(cos_d).reshape(-1,1)]

    return X,y[:len(X_overlap)],n,h

# Function to generate one-hot features
def generate_features_one_hot(dataset):
    corpus_headline = []
    corpus_body = []
    corpus_stances = []
    bodyIDs = []
    for a in dataset.stances:
        corpus_headline.append(a['Headline'])
        corpus_body.append(dataset.articles[int(a['Body ID'])])
        if a['Stance'] == 'agree':
            corpus_stances.append(0)
        elif a['Stance'] == 'disagree':
            corpus_stances.append(1)
        elif a['Stance'] == 'discuss':
            corpus_stances.append(2)
        else:
            corpus_stances.append(3)
        bodyIDs.append(int(a['Body ID']))

    print('1')
    # create one hot objects for the headlines and bodies
    voc_size = 250
    onehot_repr_headline = [one_hot(words,voc_size)for words in corpus_headline]
    onehot_repr_body = [one_hot(words,voc_size)for words in corpus_body]
    print('2')
    # padded the one hot vectors for consistence
    sent_length=20
    embedded_docs_headline = pad_sequences(onehot_repr_headline,padding='pre',maxlen=sent_length)
    embedded_docs_body = pad_sequences(onehot_repr_body,padding='pre',maxlen=sent_length)
    print('3')
    y = np.array(corpus_stances)
    
    # add all the features together to form the X array
    X = np.c_[embedded_docs_headline, embedded_docs_body]

    return X,y,bodyIDs,corpus_headline

# Function to generate word2vec features
def generate_features_word2vec(dataset):
    vocabulary_size = 5000
    num_embedding_matrix_vectors = 100
    maximum_headline_length = 20
    maxmum_body_length = 100
    corpus_headline = []
    corpus_body = []
    corpus_stances = []

    for a in dataset.stances:
        corpus_headline.append(a['Headline'])
        corpus_body.append(dataset.articles[int(a['Body ID'])])
        if a['Stance'] == 'agree':
            corpus_stances.append(0)
        elif a['Stance'] == 'disagree':
            corpus_stances.append(1)
        elif a['Stance'] == 'discuss':
            corpus_stances.append(2)
        else:
            corpus_stances.append(3)


    # # ----------------------------- WORD2VEC -----------------------------
    # t = corpus_headline + corpus_body
    # print("1.1")
    # text_str = ' '.join(t)
    # print("1.2")
    # tokens = Tokenizer(num_words = 300).fit_on_texts(text_str)
    # print("1.3")
    # words = [word.lower() for word in tokens if word.isalpha()]
    # print("1.4")
    # stop_words = set(stopwords.words('english'))
    # print("1.5")
    # words = [word for word in words if not word in stop_words]
    # print("1.6")
    # model = Word2Vec(words)
    # print("1.7")
    # print("2")

    # ----------------------------- HEADLINE -----------------------------
    # FIT THE HEADLINES TO A TOKENIZER AND CREATE THE X VECTORS
    tokenized_headlines = []
    for headline in corpus_headline:
        tokenized_headlines.append(text_to_word_sequence(headline))

    # Fit to a tokenizer
    tokenizer = Tokenizer(num_words=vocabulary_size)
    headlines_fit_on_texts = []
    for h in tokenized_headlines:
        headlines_fit_on_texts.append(' '.join(h[:maximum_headline_length]))
    tokenizer.fit_on_texts(headlines_fit_on_texts)

    # Create x input and apply padding/truncating
    X_text_to_sequences = []
    for h in tokenized_headlines:
        X_text_to_sequences.append(' '.join(h[:maximum_headline_length]))
    X = tokenizer.texts_to_sequences(X_text_to_sequences)
    X_headline = pad_sequences(X, maxlen=maximum_headline_length, padding='post', truncating='post')


    # ----------------------------- BODY -----------------------------
    # FIT THE BODIES TO A TOKENIZER AND CREATE THE X VECTORS
    tokenized_body = []
    for b in corpus_body:
        tokenized_body.append(text_to_word_sequence(b))

    # Fit to a tokenizer
    tokenizer_body = Tokenizer(num_words=vocabulary_size)
    bodies_fit_on_texts = []
    for h in tokenized_body:
        bodies_fit_on_texts.append(' '.join(h[:maxmum_body_length]))
    tokenizer_body.fit_on_texts(bodies_fit_on_texts)

    # Create x input and apply padding/truncating
    X_body_text_to_sequences = []
    for h in tokenized_body:
        X_body_text_to_sequences.append(' '.join(h[:maxmum_body_length]))
    X_bodies = tokenizer_body.texts_to_sequences(X_body_text_to_sequences)
    X_bodies = pad_sequences(X_bodies, maxlen=maxmum_body_length, padding='post', truncating='post')

    # add all the features together to form the X array
    X = np.c_[X_headline, X_bodies]
    
    # # ----------------------------- EMBEDDING MATRIX -----------------------------
    # # Create and fill the embedding matrix from the headline and body tokenziers
    # embedding_matrix = np.zeros((vocabulary_size, num_embedding_matrix_vectors))

    # num_errors = 0
    # for word,i in tokenizer.word_index.items():
    #     try:
    #         embedding_matrix[i] = model.wv[word]
    #         num_success = num_success + 1
    #     except:
    #         num_errors = num_errors + 1
    
    # num_errors = 0
    # for word,i in tokenizer_body.word_index.items():
    #     try:
    #         embedding_matrix[i] = model.wv[word]
    #         num_success = num_success + 1
    #     except:
    #         num_errors = num_errors + 1
    return X, corpus_stances

# Function to convert stances into numbers
def convert_predictions(predictions):
    predicted = []
    for t in predictions:
        max_value = 0
        max_index = 0
        for prob_index in range(len(t)):
            if t[prob_index] > max_value:
                max_value = t[prob_index]
                max_index = prob_index
        predicted.append(LABELS[int(max_index)])
    return predicted

# Function to return a specified models given a model type
def models(name):
    if name == 'cnn':
        # Model #4
        model = Sequential()        
        model.add(Embedding(5000,40))
        model.add(Conv1D(5,5,activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dense(4, activation='softmax'))
    elif name == 'baseline':
        model = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
    elif name == 'basic_embedding_nn':
        model = Sequential()
        model.add(Embedding(5000,40))
        model.add(Dense(4, activation='softmax'))
    elif name == 'basic_lstm':
        model = Sequential()
        model.add(Embedding(5000,40))
        model.add(LSTM(100))
        model.add(Dense(4, activation='softmax'))
    elif name == 'simple_rnn':
        model = Sequential()
        model.add(Embedding(5000,40))
        model.add(SimpleRNN(100))
        model.add(Dense(4, activation='softmax'))
    elif name == 'basic_dense_nn':
        model = Sequential()
        model.add(Dense(66,38))
        model.add(Dense(4, activation='softmax'))
    elif name == 'bidirectional_lstm':
        model = Sequential()
        model.add(Embedding(1000,40,input_length=44))
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Bidirectional(LSTM(100)))
        model.add(Dense(4, activation='softmax'))
    elif name == 'tfidf':
        model = Sequential()        
        model.add(Dense(1000, activation='relu')) # making embedding layer
        model.add(Dropout(0.2))
        model.add(Dense(4, activation='softmax'))
    elif name == 'embedding_matrix':
        model.add(Embedding(5000,40))
        # model.add(Embedding(input_dim=5000, output_dim=100, weights = [embedding_matrix], trainable=True, name='word_embedding_layer'))
        model.add(Conv1D(5,5,activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dense(4, activation='softmax'))
    return model

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition, bodyID_competition, headline_competition = generate_features_tfidf(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()
    bodyIDs = dict()
    headlines = dict()
    
    # Load/Precompute all features now
    X_holdout,y_holdout, bodyID_holdout, headline_holdout = generate_features_tfidf(hold_out_stances,d,"holdout")

    for fold in fold_stances:
        Xs[fold],ys[fold],bodyIDs[fold],headlines[fold] = generate_features_tfidf(fold_stances[fold],d,str(fold))

    best_score = 0
    best_fold = None

    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        model = models('tfidf')

        # Compile model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train,y_train,batch_size=2000,epochs=3)

        predictions = model.predict(X_test)
        predicted = convert_predictions(predictions)
        actual = [LABELS[int(a)] for a in y_test]
        
        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = model

    #Run on Holdout set and report the final score on the holdout set
    predictions = best_fold.predict(X_holdout)
    predicted = convert_predictions(predictions)
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    predictions = best_fold.predict(X_competition)
    predicted = convert_predictions(predictions)
    actual = [LABELS[int(a)] for a in y_competition]

    df = pd.DataFrame({
        'Headline':headline_competition,
        'Body ID':bodyID_competition,
        'Stance':predicted
    })

    df.to_csv('answer.csv', index=False, encoding='utf-8')

    print("Scores on the test set")
    report_score(actual,predicted)

    stop = timeit.default_timer()

    print('Time: ', stop - start)  