import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import match
from dataloader import Dataloader


class TextFeatureExtraction:
    def __init__(self, args):
        self.word_model = args.word_model
        self.vocabulary_volume = args.vocabulary_volume

        if args.word_model == 'bow':
            self.model = CountVectorizer(max_features=args.vocabulary_volume,
                                         ngram_range=(1, args.n_gram))
        elif args.word_model == 'tf-idf':
            self.model = TfidfVectorizer(max_features=args.vocabulary_volume,
                                         ngram_range=(1, args.n_gram))
        print("Generated a vectorizer of {} with {} vocabulary".format(args.word_model, args.vocabulary_volume))


    def fit(self, corpus):
        self.model.fit(corpus)
        # print(self.vocabulary_demo())

    def transform(self, raw):
        return self.model.transform(raw)

    def fit_transform(self, corpus):
        return self.model.fit_transform(corpus)

    def vocabulary_demo(self):
        return self.model.vocabulary_.keys()

    def reset(self):
        if self.word_model == 'bow':
            self.model = CountVectorizer(max_features=self.vocabulary_volume)
        elif self.word_model == 'tf-idf':
            Exception()


def calc_prob_h(n, tot):
    return np.log(n) - np.log(tot)


class NBC:
    def __init__(self, args):
        # self.weight = np.zeros(0,0)
        self.text_feature = TextFeatureExtraction(args)
        self.alpha = args.alpha
        self.prior_h = None
        self.feature_size = None
        self.likelihood = None  # likelihood(h, D) = P(D|h)

        if args.weight is not None:
            self.load_weight(args.weight)

        self.save = args.save_weight

    def train(self, dataset):
        # dataset: [(raw, label)]
        self.text_feature.fit([i[0] for i in dataset])

        self.prior_h = np.zeros(2)
        self.feature_size = self.text_feature.model.max_features
        self.likelihood = np.zeros((2,self.feature_size))   # likelihood(h, D) = P(D|h)

        for idx, c in enumerate(['ham', 'spam']):
            # Prior Probability of h
            raw_c = [i for i in dataset if i[1] == c]
            cnt = len(raw_c)
            self.prior_h[idx] = calc_prob_h(cnt, len(dataset))

            # Likelihood
            feature = self.feature_extraction([i[0] for i in raw_c])  # (batch_size, feature_size)
            cnt = feature.sum(axis=0)   # (feature_size)
            tot = np.sum(feature)   # TODO: consider what to do when there's more feature than word_vector
            for i, f in enumerate(cnt):
                self.likelihood[idx][i] = self.calc_likelihood(f, tot)

        if self.save is not None:
            self.save_weight(self.save)


    def save_weight(self, filename):
        print("Save model weight to: ", filename)
        with open(filename, 'wb+') as f:
            pickle.dump(self.prior_h, f)
            pickle.dump(self.likelihood, f)
            pickle.dump(self.text_feature, f)

    def load_weight(self, filename):
        with open(filename, 'rb') as f:
            self.prior_h = pickle.load(f)
            self.likelihood = pickle.load(f)
            self.text_feature = pickle.load(f)

    def eval(self, raw):
        feature = self.feature_extraction(raw)
        result = np.zeros(len(raw))

        for i in range(len(raw)):
            estimated = [np.sum(feature[i] * self.likelihood[c]) + self.prior_h[c] for c in [0,1]]
            result[i] = np.argmax(estimated)

        return result

    def feature_extraction(self, raw):
        text_feature = self.text_feature.transform(raw).toarray()
        # TODO: match feature

        feature = np.concatenate([text_feature], 1)
        return feature

    def calc_likelihood(self, cnt, tot):
        # cnt: #{x | x = xk, y(x) = c}, tot: #{x | y(x) = c}
        return np.log(cnt + self.alpha) - np.log(tot + self.alpha * 2)

    def demo(self, raw):
        # raw: single raw text
        feature = self.feature_extraction([raw])
        estimated = np.argmax([np.sum(feature[0] * self.likelihood[c]) + self.prior_h[c] for c in [0,1]])
        if estimated:
            return "spam"
        else:
            return "ham"

    def reset(self):
        self.prior_h = None
        self.likelihood = None
        # self.text_feature.reset()