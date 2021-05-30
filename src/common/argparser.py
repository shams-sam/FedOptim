import argparse
from common.utils import booltype, Struct


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--clf', type=str, required=False)
    parser.add_argument('--optim', type=str, required=False, default='sgd')
    parser.add_argument('--scheduler', type=booltype,
                        required=False, default=False)
    # Type of algorithm used: Stochastic, PCA, K-Grad, TopK, etc.
    parser.add_argument('--paradigm', type=str, nargs='+',
                        required=False, default=[])
    parser.add_argument('--p-args', type=str, nargs='+', required=False)
    parser.add_argument('--ncomponent', type=int, required=False)
    parser.add_argument('--rp-eps', type=float, required=False)
    parser.add_argument('--pca-var', type=float, required=False)
    parser.add_argument('--sdir-full', type=booltype, required=False)
    parser.add_argument('--kgrads', type=int, required=False)
    parser.add_argument('--topk', type=float, required=False)
    parser.add_argument('--atomo-r', type=int, required=False)
    parser.add_argument('--dga-bs', type=int, required=False)
    parser.add_argument('--num-dga', type=int, required=False)

    # number of grads to be accumulated before gradient decomposition starts
    parser.add_argument('--residual', type=booltype, required=False)
    parser.add_argument('--error-tol', type=float, required=False)
    parser.add_argument('--num-workers', type=int, required=False)
    parser.add_argument('--batch-size', type=int, required=False, default=0)
    parser.add_argument('--test-batch-size', type=int,
                        required=False, default=0)

    # Std Dev of Gaussian noise added to test datasets
    parser.add_argument('--noise', type=float, required=False)
    parser.add_argument('--start-epoch', type=int, required=False, default=1)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--tau', type=int, required=False, default=2)
    parser.add_argument('--loss-type', type=str, required=False, default='ce')
    parser.add_argument('--lr', type=float, required=False)
    parser.add_argument('--nesterov', type=booltype,
                        required=False, default=False)
    parser.add_argument('--momentum', type=float, required=False, default=0)
    parser.add_argument('--decay', type=float, required=False, default=1e-5)

    parser.add_argument('--no-cuda', type=booltype,
                        required=False, default=False)
    parser.add_argument('--device-id', type=int, nargs='+', required=False)
    parser.add_argument('--seed', type=int, required=False, default=1)

    parser.add_argument('--stratify', type=booltype,
                        required=False, default=True)
    # If data should be distributed uniformly among nodes
    parser.add_argument('--uniform-data', type=booltype,
                        required=False, default=True)
    # Shuffle data between epochs among nodes
    # to shuffle would be an easier case as the batch would be dynamic
    # closer to the centralized sgd case
    parser.add_argument('--shuffle-data', type=booltype,
                        required=False, default=True)
    # number of class labels per node
    parser.add_argument('--non-iid', type=int, required=False)
    # if the samples should be repeated among nodes or be a partition
    parser.add_argument('--repeat', type=float, required=False)

    parser.add_argument('--dry-run', type=booltype,
                        required=False, default=True)
    parser.add_argument('--early-stopping', type=booltype,
                        required=False, default=False)
    parser.add_argument('--patience', type=int, required=False, default=2)
    parser.add_argument('--log-intv', type=int, required=False, default=1)
    parser.add_argument('--save-model', type=booltype,
                        required=False, default=False)
    parser.add_argument('--load-model', type=str,
                        required=False, default=False)

    args = vars(parser.parse_args())
    args = Struct(**args)

    return args
