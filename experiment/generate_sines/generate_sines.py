import utils
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--no', type=int)
parser.add_argument('--seq_len', type=int)
parser.add_argument('--dim', type=int)
parser.add_argument('--freq_range', nargs='+', type=float)
parser.add_argument('--amp_range', nargs='+', type=float)
parser.add_argument('--phase_range', nargs='+', type=float)
args = parser.parse_args()

# run experiment
titles = ['jakob_gen', 'timegan_gen_1', 'timegan_gen_2', 'timegan_gen_3', 'timegan_gen_4']
generators = [utils.jakob_gen, utils.timegan_gen_1, utils.timegan_gen_2, utils.timegan_gen_3, utils.timegan_gen_4]
for gen, title in zip(generators, titles):
    X = gen(args.no, args.seq_len, args.dim, args.freq_range, args.amp_range, args.phase_range)
    X_train, X_test = train_test_split(X, train_size=0.8, random_state=1)

    # save data to timeVAE and TSGBench
    np.savez('../timeVAE/datasets/' + title + '_train.npz', X_train)
    np.savez('../timeVAE/datasets/' + title + '_test.npz', X_test)
    np.savez('../TSGBench/data/' + title + '_test.npz', X_test)

    # plot sines
    utils.plot_sines(X, title)

print('finished')
