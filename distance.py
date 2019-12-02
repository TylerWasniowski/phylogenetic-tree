from Bio import pairwise2
import multiprocessing as mp
import numpy as np
from time import time


# Computes distance matrix from sequences in parallel
# If cpus is not provided, will try to use all available
def compute(filename, procs=None, verbose=False):
    if procs is None:
        procs = mp.cpu_count()

    with open(filename, "r") as file:
        sequences = list(filter(lambda seq: not seq.startswith(">"), file.read().splitlines()))

        with mp.Pool(procs) as pool:
            # Pair up each sequence with each other sequence for pairwise alignment
            sequences_pairs = []

            for i in range(len(sequences)):
                for j in range(i + 1, len(sequences)):
                    sequences_pairs.append((sequences[i], sequences[j]))

            # Calls pairwise alignment in parallel for each pair of sequences
            start = time()
            scores = [pool.apply_async(alignment_score, args=(pair,)) for pair in sequences_pairs]
            scores = [score.get() for score in scores]
            end = time()

            dist_matrix = np.zeros((len(sequences), len(sequences)))
            scores_iter = iter(scores)
            for i in range(len(sequences)):
                # Set dist of A->A to infinity
                dist_matrix[i][i] = np.inf
                for j in range(i + 1, len(sequences)):
                    dist_matrix[i][j] = max(len(sequences[i]), len(sequences[j])) - next(scores_iter)
                    # Dist of A to B is same as dist of B to A
                    dist_matrix[j][i] = dist_matrix[i][j]

            if verbose:
                print(dist_matrix)
                print("Scores:", scores)
                print("Time:", end - start, "seconds")

            return dist_matrix


def alignment_score(sequence_pair):
    return pairwise2.align.globalxx(*sequence_pair, score_only=True)


