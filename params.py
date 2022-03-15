import argparse
import yaml


def get_args():
    parser = argparse.ArgumentParser(description='Naive Bayes Classification')
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument('--data_path')
    parser.add_argument('--label_path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k_fold', type=int, default=5)

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