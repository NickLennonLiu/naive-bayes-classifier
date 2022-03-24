import argparse
import random

import yaml

def get_default_args():
    parser = get_parser()
    p = parser.parse_args([])

    if p.config is not None:
        # load config file
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        # update parser from config file
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key
        parser.set_defaults(**default_arg)
    return parser.parse_args([])

def get_parser():
    parser = argparse.ArgumentParser(description='Naive Bayes Classification')
    parser.add_argument('--config', default='./configs/baseline.yaml')
    parser.add_argument('--data_path')
    parser.add_argument('--label_path')
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--save_weight', type=str, default=None)
    parser.add_argument('--phase', type=str, default='full', choices=['train', 'test', 'full'])
    parser.add_argument('--save_result', type=str, default=None)


    # Analysis
    parser.add_argument('--sample', type=float, default=1.0)

    # Evaluation
    parser.add_argument('--beta', type=float, default=1)

    # Naive Bayes Classifier
    parser.add_argument('--alpha', type=float, default=1)

    # text feature extraction
    parser.add_argument('--word_model', type=str, default='bow', choices=['bow', 'tf-idf'])
    parser.add_argument('--n_gram', type=int, default=1)
    parser.add_argument('--vocabulary_volume', type=int, default=1000)

    # Extra Features
    parser.add_argument('--content_feature', type=bool, default=False)

    return parser


def get_args():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        # load config file
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        # update parser from config file
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key
        parser.set_defaults(**default_arg)
    return parser.parse_args()