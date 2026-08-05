#!/usr/bin/env python
"""Apply data transforms to a GNN HDF5 file and write a new GNN HDF5 file."""

import argparse
import math
import multiprocessing as mp
import os
import sys
import h5py
import tqdm
from torch_geometric.transforms import Compose

from pynuml.data import NuGraphData
from nugraph.util import (EventLabels, FeatureExtension, HierarchicalEdges,
                           PositionFeatures, PruneGraph)
from nugraph.models.nugraph2.transform import Transform as NG2Transform
from nugraph.models.nugraph3.transform import Transform as NG3Transform

# Registry mapping CLI flag name → (class, needs_planes).
# ORDER MATTERS: transforms are applied in the order listed here, regardless of
# the order flags appear on the command line. The current ordering reflects these
# dependencies:
#   - prune-graph operates on the planar format (data[p, 'plane', p]) and must
#     run before ng3/ng2/hierarchical-edges, which merge or reformat plane stores
#   - position-features and feature-extension expect the node feature tensor to
#     already be in its final planar or hit format, so they run after format transforms
#   - event-labels has no dependencies and is placed last as a lightweight fixup
# Add future transforms here in the correct position; no other changes needed.
TRANSFORMS = {
    "prune-graph":        (PruneGraph,        True),
    "hierarchical-edges": (HierarchicalEdges, True),
    "ng3":                (NG3Transform,      True),
    "ng2":                (NG2Transform,      True),
    "position-features":  (PositionFeatures,  True),
    "feature-extension":  (FeatureExtension,  True),
    "event-labels":       (EventLabels,       False),
}



def _build_transform(selected: list[str], planes: list[str]) -> Compose:
    transform_list = []
    for flag in selected:
        cls, needs_planes = TRANSFORMS[flag]
        transform_list.append(cls(planes) if needs_planes else cls())
    return Compose(transform_list)


def event_id_str(data: NuGraphData) -> str:
    """Return a human-readable run/subrun/event string from a graph's metadata."""
    try:
        md = data["metadata"]
        return (f"run={md.run.item()} subrun={md.subrun.item()} "
                f"event={md.event.item()}")
    except Exception:
        return "<metadata unavailable>"


def _process_chunk(args: tuple) -> tuple:
    """Process a chunk of samples and write results to a temp HDF5 file.

    Runs entirely inside a worker process — no tensor data is returned to the
    main process, eliminating pickle overhead.
    """
    import torch
    infile, tmp_path, names, planes, selected = args
    # prevent PyTorch's internal thread pool from competing across workers
    torch.set_num_threads(1)
    transform = _build_transform(selected, planes)
    n_ok = 0

    with h5py.File(infile, "r") as fin, h5py.File(tmp_path, "w") as fout:
        for name in names:
            data = NuGraphData.load(fin[f"dataset/{name}"])
            try:
                data = transform(data)
                data.save(fout, f"dataset/{name}")
                n_ok += 1
            except Exception as exc:
                raise RuntimeError(
                    f"Error processing graph '{name}' [{event_id_str(data)}]"
                ) from exc

    return tmp_path, n_ok


def configure():
    parser = argparse.ArgumentParser(
        description="Apply transforms to a GNN HDF5 file.")
    parser.add_argument("--infile",  "-i", type=str, required=True,
                        help="Input GNN HDF5 file")
    parser.add_argument("--outfile", "-o", type=str, required=True,
                        help="Output GNN HDF5 file")
    parser.add_argument("--num-workers", "-j", type=int, default=4,
                        help="Number of parallel worker processes (0 for sequential)")
    for flag in TRANSFORMS:
        parser.add_argument(f"--{flag}", action="store_true",
                            help=f"Apply {flag} transform")
    return parser.parse_args()


def main(args):

    selected = [flag for flag in TRANSFORMS if getattr(args, flag.replace("-", "_"))]
    if not selected:
        print("No transforms selected. Use --help to see available options.")
        sys.exit(1)

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
    print(f"Workers: {args.num_workers if args.num_workers > 0 else 'sequential'}")
    print("Transforms (applied in this order):")
    for i, flag in enumerate(selected):
        print(f"  {i+1}. {flag}")

    # write metadata before forking (HDF5 handles are not fork-safe)
    with h5py.File(args.outfile, "x") as fout:
        with h5py.File(args.infile, "r") as fin:
            # copy all top-level items except 'dataset' (written incrementally later)
            for key in fin.keys():
                if key != "dataset":
                    fin.copy(key, fout)
            # copy root attributes
            for attr, val in fin.attrs.items():
                fout.attrs[attr] = val
        # pre-create the dataset group so entries are inserted into an existing
        # index rather than triggering group creation on the first copy
        fout.create_group("dataset", track_order=False)

    n_ok = 0

    if args.num_workers == 0:
        # sequential: single chunk, no temp files, cleaner stack traces
        _, n_ok = _process_chunk(
            (args.infile, args.outfile + ".tmp.h5", samples, planes, selected))
        with h5py.File(args.outfile, "a") as fout, \
             h5py.File(args.outfile + ".tmp.h5", "r") as ftmp:
            for name in tqdm.tqdm(samples, desc="Writing"):
                ftmp.copy(f"dataset/{name}", fout, f"dataset/{name}")
        os.remove(args.outfile + ".tmp.h5")

    else:
        # divide samples into chunks; target ~4x num_workers tasks so the
        # progress bar updates regularly without excessive file overhead
        n_tasks = max(args.num_workers * 4, 8)
        chunk_size = max(1, math.ceil(len(samples) / n_tasks))
        chunks = [samples[i:i + chunk_size]
                  for i in range(0, len(samples), chunk_size)]

        chunk_args = [
            (args.infile, f"{args.outfile}.tmp.{i}.h5", chunk, planes, selected)
            for i, chunk in enumerate(chunks)
        ]

        tmp_files = [a[1] for a in chunk_args]

        try:
            with mp.Pool(args.num_workers) as pool:
                it = pool.imap_unordered(_process_chunk, chunk_args)
                with h5py.File(args.outfile, "a") as fout:
                    for tmp_path, n in tqdm.tqdm(
                            it, total=len(chunks), desc="Transforming",
                            unit="chunk"):
                        with h5py.File(tmp_path, "r") as ftmp:
                            for name in ftmp.get("dataset", {}).keys():
                                ftmp.copy(f"dataset/{name}", fout,
                                          f"dataset/{name}")
                        os.remove(tmp_path)
                        n_ok += n
        except Exception:
            # clean up any temp files left behind by a pool crash
            for p in tmp_files:
                if os.path.exists(p):
                    os.remove(p)
            raise

    print(f"\nDone: {n_ok} graphs written.")


if __name__ == "__main__":
    main(configure())
