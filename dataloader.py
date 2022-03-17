import os
import random
import glob


def path_to_idx(path):
    return '_'.join(path.split(os.sep)[-2:])


class Dataloader:
    def __init__(self, args):
        self.k_fold = args.k_fold
        self.data_path = args.data_path
        self.label_path = args.label_path
        self.random_seed = args.seed
        self.label = dict()
        self.raw_data = dict()
        self.raw_idx = []
        self.train_idx = []
        self.test_idx = []

        if not self.random_seed:
            self.random_seed = int(random.random())

        self.load_data()
        self.load_label()
        self.shuffle()

        self.fold = list(range(0, len(self), int(len(self) / self.k_fold)))
        self.fold[-1] = len(self)
        print("Loaded {} data".format(len(self)))
        print("Random seed: {}".format(self.random_seed))

    def idx_to_path(self, idx: str):
        return os.path.join(self.data_path, *idx.split('_'))

    def load_data(self):
        # data/000/000, data/000/001, ... data/126/021
        for file in glob.glob(self.data_path + '/*/*'):
            idx = path_to_idx(file)
            self.raw_idx.append(idx)
            try:
                self.raw_data[idx] = open(file).read()
            except UnicodeDecodeError:
                self.raw_data[idx] = open(file, encoding='iso-8859-1').read()

    def load_label(self):
        with open(self.label_path) as f:
            for line in f.readlines():
                [label, idx] = line.split()
                self.label[path_to_idx(idx)] = label

    def get_data(self, idx):
        return self.raw_data[idx]

    def get_label(self, idx):
        return self.label[idx]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [(self.raw_data[i], self.label[i]) for i in idx]
        else:
            return self.raw_data[idx], self.label[idx]

    def __len__(self):
        return len(self.raw_idx)

    def shuffle(self):
        random.shuffle(self.raw_idx)

    def get_fold_idx(self, idx):
        assert(0 <= idx < self.k_fold)
        return self.raw_idx[self.fold[idx]: self.fold[idx+1]]

    def get_train_fold(self, idx):
        train_idx = []
        for i in range(0, self.k_fold):
            if not i == idx:
                train_idx += self.get_fold_idx(i)
        return train_idx


    def get_test_fold(self, idx):
        test_idx = self.get_fold_idx(idx)
        return test_idx

    def get_ham(self):
        return [i for i in self.raw_idx if self.label[i] == 'ham']

    def get_spam(self):
        return [i for i in self.raw_idx if self.label[i] == 'spam']