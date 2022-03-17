import argparse
import yaml

from dataloader import Dataloader
from model import NBC
from params import get_args

def main():
    args = get_args()
    dataloader = Dataloader(args)
    nbc = NBC(args)

    for i in range(args.k_fold):

        if args.phase == 'train' or args.phase == 'full':
            train_idx = dataloader.get_train_fold(i)
            dataset = dataloader[train_idx]
            nbc.train(dataset)
            print("Train of fold {} done".format(i))

        if args.phase == 'test' or args.phase == 'full':
            test_idx = dataloader.get_test_fold(i)
            dataset = dataloader[test_idx]
            result = nbc.eval([i[0] for i in dataset])
            true_label = [(0 if l[1] == 'ham' else 1) for l in dataset]

            cnt = 0
            for idx in range(len(true_label)):
                if result[idx] == true_label[idx]:
                    cnt += 1

            print("Accuracy: {}, ({}/{})".format(cnt/len(test_idx), cnt, len(test_idx)))

        nbc.reset()

if __name__ == "__main__":
    main()
