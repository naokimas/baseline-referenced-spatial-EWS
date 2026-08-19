"""Explain, rather than merely crash, when input data this repository does not ship is missing.

Two kinds of input are deliberately absent from the repository (see "Data availability" in README.md):
the 34 empirical adjacency matrices, which belong to their original owners and are not ours to
redistribute, and the Stage-1/Stage-2 simulation output, which is regenerable but far too large to
distribute. On a fresh clone the scripts that consume them would otherwise stop at a bare
FileNotFoundError on some arbitrary first file, which says nothing about why it is missing. These two
helpers turn that into one sentence naming the cause and the fix.
"""
import os, sys
import scipy.sparse as sp

SYNTHETIC = {"barabasialbert", "erdosrenyi", "gkk", "hk100", "lattice", "smallworld"}


def require_dir(path, produced_by):
    """Stop unless `path` is a non-empty directory, naming the script that fills it."""
    if os.path.isdir(path) and os.listdir(path):
        return
    rel = os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    sys.exit(f"\nMissing input: {rel}/ is absent or empty.\n"
             f"It is not distributed with this repository (regenerable, but several hundred MB).\n"
             f"Create it by running: python3 {produced_by}\n"
             f"See the 'Data availability' section of README.md.\n")


def load_network(netdir, net):
    """Load data/networks/<net>.npz, or stop with an explanation if it is an absent empirical network."""
    path = os.path.join(netdir, net + ".npz")
    if os.path.exists(path):
        return sp.load_npz(path)
    if net in SYNTHETIC:
        sys.exit(f"\nMissing network file: data/networks/{net}.npz (a synthetic network, "
                 f"which should be present in this repository).\n")
    sys.exit(f"\nMissing network file: data/networks/{net}.npz\n"
             f"'{net}' is one of the 34 empirical networks. Those data belong to their original\n"
             f"owners and are not redistributed here, so you must download it yourself.\n"
             f"See data/networks/README.md for where to obtain it and how to save it.\n")
