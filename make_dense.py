import argparse
import vigra
import numpy as np
import os

from coordinates_from_json import coordinates_from_json
from dense_reconstruction import dense_reconstruction

# TODO need to check file formats with zhihao
# coul also use tif, if he doesn't like hdf5
def process_command_line():
    parser = argparse.ArgumentParser(
            description='Input for dense reconstruction from skeletons')

    parser.add_argument('prob_path', type=str, help = 'Path to probability maps')
    parser.add_argument('prob_key',  type=str, help = 'Key for probability maps')

    parser.add_argument('skeleton_path', type=str, help = 'Path to the json with the skeleton data')

    parser.add_argument('rf_path', type=str, help = 'Path to save Random Forest')

    parser.add_argument('bounding_box', type=tuple, help = 'Bounding box of the probability maps in relation to the coordinate format in the json file')

    args = parser.parse_args()

    return args


def main():
    args = process_command_line()

    bounding_box = args.bounding_box
    assert len(bounding_box) == 6, "Bounding box needs 6 coordinates!"

    # get the skeleton coordinates first
    skeleton_coordinates = coordinates_from_json(args.skeleton_path, bounding_box)

    # perform dense reconstruction for all skeleton coordinates we have
    # TODO should this turn out too much overhead, we could also query
    # for only the ids of interest
    dense_skeletons = dense_reconstruction(args.prob_path, args.prob_key,
            args.skeleton_path, args.rf_path)

    # TODO save this for optical check and figure out how to further process the result


if __name__ == '__main__':
    main()
