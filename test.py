import os
import main
import params
import matplotlib.pyplot as plt
import numpy as np

def plot_result(dirname, var, title, metric, key=None):
    plt.close()
    result_files = sorted([i for i in os.listdir(dirname) if i.split('.')[-1] == 'txt'], key=key)
    size = len(result_files)
    result = np.zeros((8, size))
    xname = []
    for idx, filename in enumerate(result_files):
        with open(os.path.join(dirname, filename)) as f:
            temp = f.read().split()[-8:]
            # print(temp)
            result[:, idx] = temp
            xname.append(os.path.splitext(filename)[0].split('_')[-1])

    metrics = [['accuracy', 'precision', 'recall', 'F1'][i] for i in metric]
    for idx in metric:
        plt.plot(xname, result[idx])
    plt.legend(metrics)
    # plt.title(title)
    plt.xlabel(var)
    plt.savefig(os.path.join(dirname, title + '.png'))

def test_alpha(args):
    alpha_list = [0, 1]
    for alpha in alpha_list:
        args.alpha = alpha
        args.save_result = "./workdir/alpha/baseline_alpha_{}.txt".format(alpha)
        # main.main(args)
    plot_result('./workdir/alpha', 'alpha', 'result', [0])

def test_vocabulary_size(args):
    size_list = [100, 500, 1000, 2000, 5000, 10000]
    for size in size_list:
        args.vocabulary_volume = size
        args.save_result = "./workdir/vol/baseline_vol_{}.txt".format(size)
        # main.main(args)
    plot_result('./workdir/vol', 'vol_size', 'result', [0], key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

def test_n_gram(args):
    n_gram = [1, 2, 3]
    for n in n_gram:
        args.n_gram = n
        args.save_result = "./workdir/ngram/baseline_ngram_{}.txt".format(n)
        # main.main(args)
    plot_result('./workdir/ngram', 'n_gram', 'result', [0])


def test_sample(args):
    sample_rate = [0.05, 0.5, 1]
    for sr in sample_rate:
        args.sample = sr
        args.save_result = "./workdir/sample/baseline_sample_{}.txt".format(sr)
        # main.main(args)
    plot_result('./workdir/sample', 'sample_rate', 'result', [0])

def test_sample_tfidf(args):
    sample_rate = [0.05, 0.5, 1]
    for sr in sample_rate:
        args.word_model = 'tf-idf'
        args.sample = sr
        args.save_result = "./workdir/tfidf_sample/tfidf_sample_{}.txt".format(sr)
        # main.main(args)
    plot_result('./workdir/tfidf_sample', 'sample_rate', 'result', [0])

def test_random_seed(args):
    seed_list = [1, 2, 3, 4]
    for seed in seed_list:
        args.seed = seed
        args.save_result = "./workdir/base_seed_{}.txt".format(seed)
        main.main(args)

def baseline(args):
    args.save_result = "./workdir/baseline.txt"
    main.main(args)

if __name__ == '__main__':
    _args = params.get_args()
    # plot_result("./workdir", 'asdf', 'asdf')
    # baseline(_args)
    test_alpha(_args)
    test_vocabulary_size(_args)
    test_n_gram(_args)
    test_sample(_args)
    test_sample_tfidf(_args)
    # test_random_seed(_args)
    # plot_result('./workdir/sample', 'sample_rate', 'Sample rate on Baseline', [0])