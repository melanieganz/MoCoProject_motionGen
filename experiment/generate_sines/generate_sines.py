import utils
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int)
parser.add_argument('--seq_len', type=int)
parser.add_argument('--dim', type=int)
parser.add_argument('--freq_range', nargs='+', type=float)
parser.add_argument('--amp_range', nargs='+', type=float)
parser.add_argument('--phase_range', nargs='+', type=float)
args = parser.parse_args()

# run experiment
titles = ['jakob_gen', 'timegan_gen', 'timegan_gen_1', 'timegan_gen_2']

for gen, title in zip(generators, titles):

	# Old generator from Jakob
	if titles == "jakob_gen":
		utils.jakob_gen(
			n_samples=args.n_samples,
			freq_range=args.seq_len,
			dim=args.dim,
			freq_range=args.freq_range,
			amp_range=args.amp_range,
			phase_range=args.phase_range
		)

	# Original
	if titles == "utils.timegan_gen":
		utils.jakob_gen(
			n_samples=args.n_samples,
			x=None,
			freq_range=args.seq_len,
			dim=args.dim,
			freq_range=args.freq_range,
			amp_range=None,
			phase_range=args.phase_range
		)

	x = np.linspace(0, 2*np.pi, args.seq_len)

	if titles == "utils.timegan_gen_1":
		utils.jakob_gen(
			n_samples=args.n_samples,
			x=x,
			freq_range=args.seq_len,
			dim=args.dim,
			freq_range=args.freq_range,
			amp_range=None,
			phase_range=args.phase_range
		)

	if titles == "utils.timegan_gen_2":
		utils.jakob_gen(
			n_samples=args.n_samples,
			x=x,
			freq_range=args.seq_len,
			dim=args.dim,
			freq_range=args.freq_range,
			amp_range=args.amp_range,
			phase_range=args.phase_range
		)

    X_train, X_test = train_test_split(X, train_size=0.8, random_state=1)

    # save data to timeVAE and TSGBench
    np.savez('../timeVAE/datasets/' + title + '_train.npz', X_train)
    np.savez('../timeVAE/datasets/' + title + '_test.npz', X_test)
    np.savez('../TSGBench/data/' + title + '_test.npz', X_test)

    # plot sines
    utils.plot_sines(X, title)

print('finished')
