#!/usr/bin/env python
import argparse

from nugraph.data import H5DataModule

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='HDF5 file to compute spacepoint normalization for')
    return parser.parse_args()

if __name__ == '__main__':
    args = configure()
    H5DataModule.generate_norm(args.file)
