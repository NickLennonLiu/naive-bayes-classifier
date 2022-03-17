import main
import params

def test_alpha(args):
    alpha_list = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    for alpha in alpha_list:
        args.alpha = alpha
        args.save_result = "./workdir/baseline_alpha_{}.txt".format(alpha)
        main.main(args)

def test_vocabulary_size(args):
    size_list = [100, 500, 1000, 2000, 5000, 10000]
    for size in size_list:
        args.vocabulary_volume = size
        args.save_result = "./workdir/baseline_vol_{}.txt".format(size)
        main.main(args)

def test_n_gram(args):
    n_gram = [1, 2, 3]
    for n in n_gram:
        args.n_gram = n
        args.save_result = "./workdir/baseline_ngram_{}.txt".format(n)
        main.main(args)




if __name__ == '__main__':
    _args = params.get_args()
    # test_alpha(_args)
    # test_vocabulary_size(_args)
    test_n_gram(_args)