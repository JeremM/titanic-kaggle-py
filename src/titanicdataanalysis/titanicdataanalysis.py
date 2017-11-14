import pandas as pd
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def myfunc():
    print("Hello module!")


def loadingData(dataPath='../../data/', logLevel="DEBUG"):

    # this block is logger initialisation
    #TODO: put it in a function at next use
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s')
    stream_handler = logging.StreamHandler()
    logger.setLevel(logLevel)
    logger.addHandler(stream_handler)

    logger.info('Loading and preparing Titanic data')
    logger.debug('pas bon')

    train_df = pd.read_csv(dataPath+"train.csv")
    print(train_df.head())

    # train_df = pd.read_csv(open("C:/Users/Jérémie/IdeaProjects/titanic-kaggle-py/data/train.csv", 'r'))
    rt = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=3, random_state=0)

    columns = ["Fare", "Pclass","SibSp","Parch"]

    labels = train_df["Survived"].values
    features = train_df[list(columns)].values

    cross_val_result = cross_val_score(rt, features, labels, cv=5, n_jobs=-1)
    score_mean = cross_val_result.mean()
    score_max = cross_val_result.max()

    print("{0} -> RF mean: {1}".format(columns, score_mean))
    print("{0} -> RF_max: {1}".format(columns, score_max))

    return 1


if __name__ == '__main__':
    loadingData()
