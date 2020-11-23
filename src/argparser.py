import argparse
from utils import booltype, Struct


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--clf', type=str, required=True)
    parser.add_argument('--paradigm', type=str, required=True)
    parser.add_argument('--num-workers', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=False, default=0)
    parser.add_argument('--test-batch-size', type=int,
                        required=False, default=0)
    parser.add_argument('--noise', type=booltype, required=False)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--nesterov', type=booltype,
                        required=False, default=False)
    parser.add_argument('--decay', type=float, required=False, default=1e-5)
    parser.add_argument('--conj-dev', type=str, required=False)
    parser.add_argument('--no-cuda', type=booltype,
                        required=False, default=False)
    parser.add_argument('--device-id', type=int,
                        required=True)
    parser.add_argument('--seed', type=int, required=False, default=1)
    parser.add_argument('--log-interval', type=int, required=False, default=1)
    parser.add_argument('--save-model', type=booltype,
                        required=False, default=True)
    parser.add_argument('--stratify', type=booltype,
                        required=False, default=True)
    parser.add_argument('--uniform-data', type=booltype,
                        required=False, default=True)
    parser.add_argument('--shuffle-data', type=booltype,
                        required=False, default=True)
    parser.add_argument('--non-iid', type=int, required=True)
    parser.add_argument('--repeat', type=int, required=True)
    parser.add_argument('--dry-run', type=booltype,
                        required=False, default=True)

    args = vars(parser.parse_args())
    args = Struct(**args)

    return args
