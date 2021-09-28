from common.argparser import argparser
from common.arguments import Arguments
from common.utils import get_paths
from common.approximation import get_dga_sdirs
import pickle as pkl

args = Arguments(argparser())
paths = get_paths(args)
print('Loading data: {}'.format(paths.data_path))
X_trains, _, y_trains, _, meta = pkl.load(
    open(paths.data_path, 'rb'))

sdirs = get_dga_sdirs(args, X_trains, y_trains)

print('Saving:', paths.dga_path)
if not args.dry_run:
    pkl.dump(sdirs, open(paths.dga_path, 'wb'))
