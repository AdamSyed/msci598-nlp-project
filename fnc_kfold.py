import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version
import pandas as pd

def generate_features(stances,dataset,name):
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

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y,n,h

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition, bodyID_competition, headline_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()
    bodyIDs = dict()
    headlines = dict()
    
    # Load/Precompute all features now
    X_holdout,y_holdout, bodyID_holdout, headline_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold],bodyIDs[fold],headlines[fold] = generate_features(fold_stances[fold],d,str(fold))

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

        c = 0
        c2 = 0
        for j in X_test:
            c = c + 1
            for x in j:
                c2 = c2 + 1
        print(c)
        print(c2)
        # model = Sequential()
        # model.add(Dense(12, input_dim=,activation='relu'))
        # model.add(Dense(8, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy',optimizer-'adam',metrics=['accuracy'])
        # model.fit(X,y,epochs=150,batch_size=10)

    #     clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
    #     clf.fit(X_train, y_train)

    #     predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
    #     actual = [LABELS[int(a)] for a in y_test]

    #     fold_score, _ = score_submission(actual, predicted)
    #     max_fold_score, _ = score_submission(actual, actual)

    #     score = fold_score/max_fold_score

    #     print("Score for fold "+ str(fold) + " was - " + str(score))
    #     if score > best_score:
    #         best_score = score
    #         best_fold = clf


    # #Run on Holdout set and report the final score on the holdout set
    # predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    # actual = [LABELS[int(a)] for a in y_holdout]

    # print("Scores on the dev set")
    # report_score(actual,predicted)
    # print("")
    # print("")

    # #Run on competition dataset
    # predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    # actual = [LABELS[int(a)] for a in y_competition]

    # df = pd.DataFrame({
    #     'Headline':headline_competition,
    #     'Body ID':bodyID_competition,
    #     'Stance':predicted
    # })

    # df.to_csv('answer.csv', index=False, encoding='utf-8')

    # print("Scores on the test set")
    # report_score(actual,predicted)