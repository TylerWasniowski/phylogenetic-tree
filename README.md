# phylogenetic-tree
Constructs a phylogenetic tree from DNA sequences.

# Installing
Download and install [python3.6+](https://www.python.org/downloads/)

Once you have python downloaded, make sure the pip binary (shipped with python) is in your path environment variable

Run `pip install numpy biopython opencv-python`

Then run `git clone https://github.com/TylerWasniowski/phylogenetic-tree.git`

# Running
## Full pipeline (from sequence to tree)

Run `cd phylogenetic-tree`

Then use `python pipeline.py --filename <FILENAME>`

## From distance matrix to tree
In python:

`import tree`

`dist_matrix = ...`

`tree.draw(tree.compute(dist_matrix))`

# Options
Here's a list of the pipeline.py optional command-line arguments:

`--procs`, `-p`: Number of processes to use for pairwise alignment

`--shrink`, `-s`: Shrink graph to fit

`--verbose`, `-v`: Print extra info
