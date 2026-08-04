#!/usr/bin/env python
"""Apply data transforms to a GNN HDF5 file and write a new GNN HDF5 file."""

import argparse
import sys
import traceback

import h5py
import tqdm
from torch_geometric.transforms import Compose

from pynuml.data import NuGraphData
from nugraph.util import (EventLabels, FeatureExtension, HierarchicalEdges,
                           PositionFeatures, PruneGraph)
from nugraph.models.nugraph2.transform import Transform as NG2Transform
from nugraph.models.nugraph3.transform import Transform as NG3Transform

# Registry mapping CLI flag name → (class, needs_planes)
# Add future transforms here; no other changes to the script are needed.
TRANSFORMS = {
    "ng3":        (NG3Transform,       True),
    "ng2":        (NG2Transform,       True),
    "hierarchical-edges": (HierarchicalEdges, True),
    "position-features":  (PositionFeatures,  True),
    "feature-extension":  (FeatureExtension,  True),
    "event-labels":       (EventLabels,       False),
    "prune-graph":        (PruneGraph,        True),
}

METADATA_KEYS = [
    "planes", "semantic_classes", "event_classes", "gen",
    "samples/train", "samples/validation", "samples/test",
    "datasize/train",
]


def configure():
    parser = argparse.ArgumentParser(
        description="Apply transforms to a GNN HDF5 file.")
    parser.add_argument("--infile",  "-i", type=str, required=True,
                        help="Input GNN HDF5 file")
    parser.add_argument("--outfile", "-o", type=str, required=True,
                        help="Output GNN HDF5 file")
    for flag in TRANSFORMS:
        parser.add_argument(f"--{flag}", action="store_true",
                            help=f"Apply {flag} transform")
    return parser.parse_args()


def event_id_str(data: NuGraphData) -> str:
    """Return a human-readable run/subrun/event string from a graph's metadata."""
    try:
        md = data["metadata"]
        return (f"run={md.run.item()} subrun={md.subrun.item()} "
                f"event={md.event.item()}")
    except Exception:
        return "<metadata unavailable>"


def main(args):

    selected = [flag for flag in TRANSFORMS if getattr(args, flag.replace("-", "_"))]
    if not selected:
        print("No transforms selected. Use --help to see available options.")
        sys.exit(1)

    # read file-level metadata from the input
    with h5py.File(args.infile, "r") as fin:
        try:
            planes = fin["planes"].asstr()[()].tolist()
        except KeyError:
            print("ERROR: 'planes' metadata not found in input file.")
            sys.exit(1)

        samples = list(fin["dataset"].keys())

    print(f"Input:   {args.infile}  ({len(samples)} graphs)")
    print(f"Output:  {args.outfile}")
    print(f"Planes:  {planes}")
    print(f"Transforms: {', '.join(selected)}")

    # build composed transform
    transform_list = []
    for flag in selected:
        cls, needs_planes = TRANSFORMS[flag]
        transform_list.append(cls(planes) if needs_planes else cls())
    transform = Compose(transform_list)

    # open both files; output created fresh (fails if it already exists)
    with h5py.File(args.infile, "r") as fin, h5py.File(args.outfile, "x") as fout:

        # copy file-level metadata
        for key in METADATA_KEYS:
            if key in fin:
                fout[key] = fin[key][()]

        # iterate over all graphs
        n_ok = 0
        n_err = 0
        for name in tqdm.tqdm(samples, desc="Transforming"):
            data = NuGraphData.load(fin[f"dataset/{name}"])
            try:
                data = transform(data)
                data.save(fout, f"dataset/{name}")
                n_ok += 1
            except Exception as exc:
                n_err += 1
                print(f"\nERROR processing graph '{name}' "
                      f"[{event_id_str(data)}]: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    print(f"\nDone: {n_ok} graphs written, {n_err} errors.")
    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main(configure())
