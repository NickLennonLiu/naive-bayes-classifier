import os
import sys
from datetime import datetime
import numpy as np
from dataloader import Dataloader
from eval import output
from model import NBC
from params import get_args


def touch_file(filename):
    folder, file = os.path.split(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def main(args):
    dataloader = Dataloader(args)
    nbc = NBC(args)

    save_file = sys.stdout
    if args.save_result is not None:
        touch_file(args.save_result)
        save_file = open(args.save_result, 'w')

    # Use 5-fold to train and test the model
    macro_result = np.array([])
    macro_target = np.array([])
    micro_evaluation = []

    for i in range(args.k_fold):
        if args.phase == 'train' or args.phase == 'full':
            start = datetime.now()
            train_idx = dataloader.get_train_fold(i)
            dataset = dataloader[train_idx]
            nbc.train(dataset)
            estimated_time = (datetime.now() - start).seconds
            print("fold {} estimated time {}(s)".format(i, estimated_time))

        if args.phase == 'test' or args.phase == 'full':
            test_idx = dataloader.get_test_fold(i)
            dataset = dataloader[test_idx]

            result = nbc.eval([i[0] for i in dataset])
            true_label = np.array([(0 if l[1] == 'ham' else 1) for l in dataset])

            macro_result = np.concatenate([macro_result, result])
            macro_target = np.concatenate([macro_target, true_label])
            micro_evaluation.append(output(result, true_label, args.beta, save_file))

        if args.phase != 'test':
            nbc.reset()

    if args.phase != 'train':   # either test or full phase would want to store the result
        # Macro result
        # print("Macro: ", file=save_file)
        output(macro_result, macro_target, args.beta, save_file)

        # Micro result
        # print("Micro: ", file=save_file)
        print("{:.4} {:.4} {:.4} {:.4}".
              format(*np.average(np.array(micro_evaluation), axis=0), args.beta), file=save_file)

    if save_file != sys.stdout:
        save_file.close()

if __name__ == "__main__":
    _args = get_args()
    main(_args)
