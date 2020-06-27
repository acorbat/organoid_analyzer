import pathlib
from collections import Counter
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from serialize import dump, load


class Normalizer(object):

    def __init__(self):
        self.norm_dict = {}

    def normalize(self, df):
        for col, minmax in self.norm_dict.items():
            df[col + '_normalized'] = (df[col] - minmax['min']) / minmax['max']

    def find_normalization(self, df, col):
        if isinstance(col, str):
            self.norm_dict[col] = {'min': min(df[col].values),
                                   'max': max(df[col].values)}
        else:
            for this_col in col:
                self.find_normalization(df, this_col)

    def save(self, path):
        """Save actual state of normalizer to json.
        Parameters
        ----------
        path : path
            path to file where normalizer is to be saved
        """
        dump(self.norm_dict, path)

    def load(self, path):
        """Load parameters from json file to normalizer.
        Parameters
        ----------
        path : path
            path to file where normalizer is to be saved
        """
        data = load(path)

        self.norm_dict = data


class Classifier(object):

    def __init__(self):
        self.clf = None
        self.cols = [None]

    def train(self, df, balance=True):
        count = Counter(df.c.values)
        print(count)

        if balance:
            mx = np.min(list(count.values()))

            selected = []
            for label in count.keys():
                ndxs = df[df['c'] == label].index.tolist()
                np.random.shuffle(ndxs)
                selected.extend(ndxs[:mx])

            df = df.loc[selected]

            count = Counter(df.c.values)
            print(count)

        X = df[self.cols]
        y = df.c.values

        self.clf.fit(X, y)

    def classify(self, df):
        X = df[self.cols]

        c_predict = self.clf.predict(X)

        return c_predict

    def save(self, path):
        path = pathlib.Path(path)
        pickle.dump(self.clf, open(str(path), 'wb'))
        dump(self.cols, str(path.with_suffix('.json')))

    def load(self, path):
        path = pathlib.Path(path)
        self.clf = pickle.load(open(str(path), 'rb'))
        self.cols = load(str(path.with_suffix('.json')))
