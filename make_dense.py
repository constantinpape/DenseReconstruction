import argparse
import vigra
import numpy as np
import os

from coordinates_from_json import coordinates_from_json
from dense_reconstruction import dense_reconstruction

def process_command_line():
    parser = argparse.ArgumentParser(
            description='Input for dense reconstruction from skeletons')

    parser.add_argument('prob_path', type=str, help = 'Path to probability maps')
    parser.add_argument('prob_key',  type=str, help = 'Key for probability maps')

    parser.add_argument('skeleton_path', type=str, help = 'Path to the json with the skeleton data')

    parser.add_argument('rf_path', type=str, help = 'Path to save Random Forest')

    parser.add_argument('bounding_box_file', type=str,
            help = 'File with bounding box of the probability maps in relation to the coordinate format in the json file, expects order x_min, x_max, y_min, y_max, z_min, z_max')

    parser.add_argument('--output_folder', default = None,
            help = "Folder for saving result")

    args = parser.parse_args()

    return args


def main():
    args = process_command_line()

    with open(args.bounding_box_file, 'r') as f:
        bb_str = f.readline()
        bounding_box = bb_str.split(' ')
        bounding_box = [int(x) for x in bounding_box]

    assert len(bounding_box) == 6, "Bounding box needs 6 coordinates!"
    print "Bounding box:", bounding_box

    print "Reading in skeletons from", args.skeleton_path
    # get the skeleton coordinates first
    skeleton_coordinates = coordinates_from_json(args.skeleton_path, bounding_box)

    # perform dense reconstruction for all skeleton coordinates we have
    # TODO should this turn out too much overhead, we could also query
    # for only the ids of interest
    print "Projecting skeletons to dense segments"
    dense_skeletons = dense_reconstruction(args.prob_path, args.prob_key,
            skeleton_coordinates, args.rf_path)

    # TODO save this for optical check and figure out how to further process the result
    if args.output_folder != None:
        print "Saving dense segments to", args.output_folder
        if not os.path.exists(args.output_folder):
            os.mkdir(args.output_folder)
        fname = os.path.join(args.output_folder, "dense_skeletons_z=")
        vigra.impex.writeVolume(dense_skeletons, fname, '.tif')

    # for debugging
    #from volumina_viewer import volumina_n_layer
    #probs = vigra.readHDF5(args.prob_path, args.prob_key)
    #volumina_n_layer([probs, dense_skeletons.astype(np.uint32)])


if __name__ == '__main__':
    main()
