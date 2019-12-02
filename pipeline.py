import argparse

import distance
import tree


def draw(filename, procs=None, shrink=False, verbose=False):
    dist_matrix = distance.compute(filename, procs=procs, verbose=verbose)
    tree.draw(tree.compute(dist_matrix, verbose=verbose), shrink=shrink)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", help="File of sequences", type=str)
    parser.add_argument("--procs",
                        "-p",
                        help="Number of processes to use for pairwise alignment",
                        type=int,
                        default=None)
    parser.add_argument("--shrink", "-s", help="Shrink graph to fit", type=bool, default=False)
    parser.add_argument("--verbose", "-v", help="Print extra info", type=bool, default=False)
    args = parser.parse_args()

    draw(args.filename, procs=args.procs, shrink=args.shrink, verbose=args.verbose)
