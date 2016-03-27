import argparse
import vigra
import numpy as np
import os
import re
import cPickle as pickle
from scipy import sparse

from coordinates_from_json import coordinates_from_json
from dense_reconstruction import dense_reconstruction_slice

def segment_statistics(prob_path, skeleton_coordinates, rf_path, shape, debug_folder):

    # load RF
    with open(rf_path, 'r') as f:
        rf = pickle.load(f)

    if debug_folder != None:
        if not os.path.exists(debug_folder):
            os.mkdir(debug_folder)

    sparse_skeletons = []
    for z in xrange(shape[2]):
        sparse_skeletons.append( sparse.dok_matrix( (shape[0], shape[1]),
            dtype = np.uint32 ) )
    for skel_id in skeleton_coordinates:
        for c in skeleton_coordinates[skel_id]:
            sparse_skeletons[ int(c[2]) ][int(c[0]), int(c[1])] = int( skel_id )


    # arrays to save the stats
    stats = {}
    for skel_id in skeleton_coordinates:
        stats[int(skel_id)] = np.zeros( (shape[2], 5) )

    # get the filenames
    files = os.listdir(prob_path)
    assert len(files) == shape[2], "Number of files must be equal to the number of slices"
    numbers = []
    filepaths = []
    # sort paths according to z slices
    for f in files:
        # This assumes, that we have only the number counting the z slice in the filename
        number = re.findall(r'\d+', f)
        assert len(number) == 1, "Filename should only contain the number indicating the z-slice"
        numbers.append( int(number[0]) )
        filepaths.append( os.path.join(prob_path,f) )
    indices = np.argsort(numbers)
    filepaths = np.array(filepaths)
    filepaths = filepaths[indices]

    # don't need this anymore
    del numbers
    del files
    del indices

    statistics =  [ "Count", "RegionRadii"]

    # iterate over z slices, reconstruct and extract statistics
    for z in xrange(shape[2]):
        print "Processing slice", z, "/", shape[2]
        probs_slice_path = filepaths[z]
        skeletons_volume = sparse_skeletons[z].toarray()
        seg_z = dense_reconstruction_slice(probs_slice_path, skeletons_volume, rf)

        if debug_folder != None:
            save_debug = os.path.join(debug_folder, "debug_" + str(z) + ".tif")
            vigra.impex.writeImage(seg_z, save_debug)

        # extract statistics
        extractor = vigra.analysis.extractRegionFeatures(
                np.zeros( (shape[0], shape[1]) ).astype(np.float32),
                seg_z.astype(np.uint32),
                features = statistics )

        for skel_id in np.unique(skeletons_volume):
            if skel_id == 0:
                continue
            stats[skel_id][z,0] = z
            # Area
            stats[skel_id][z,1] = extractor["Count"][skel_id]
            # x-radius
            stats[skel_id][z,2] = extractor["RegionRadii"][skel_id][0]
            # y-radius
            stats[skel_id][z,3] = extractor["RegionRadii"][skel_id][1]
            # combined-radius TODO is this the right way to do it?
            stats[skel_id][z,4] = ( extractor["RegionRadii"][skel_id][0] + extractor["RegionRadii"][skel_id][1] ) / 2.

    return stats


def process_command_line():
    parser = argparse.ArgumentParser(
            description='Input for extracting region statistics from skeletons')

    parser.add_argument('prob_folder', type=str, help = 'Path to the folder containing the probability maps')

    parser.add_argument('skeleton_path', type=str, help = 'Path to the json with the skeleton data')

    parser.add_argument('rf_path', type=str, help = 'Path to save Random Forest')

    parser.add_argument('bounding_box_file', type=str,
            help = 'File with bounding box of the probability maps in relation to the coordinate format in the json file, expects order x_min, x_max, y_min, y_max, z_min, z_max')

    parser.add_argument('output_folder',  help = "Folder for saving result")

    parser.add_argument('--debug_folder', default = None,
            help = "Folder for saving intermediary results for debugging")

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

    shape = (bounding_box[1] - bounding_box[0],
            bounding_box[3] - bounding_box[2],
            bounding_box[5] - bounding_box[4] )

    print "Resulting shape:", shape

    print "Reading in skeletons from", args.skeleton_path
    # get the skeleton coordinates first
    skeleton_coordinates = coordinates_from_json(args.skeleton_path, bounding_box)

    # perform dense reconstruction for all skeleton coordinates we have
    # TODO should this turn out too much overhead, we could also query
    # for only the ids of interest
    print "Extracting segment statistics from skeletons"
    segment_stats = segment_statistics(args.prob_folder,
            skeleton_coordinates, args.rf_path, shape,
            args.debug_folder)

    # save results
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    for skel_id in segment_stats:
        save_file = os.path.join(args.output_folder, str(skel_id) + ".csv")
        np.savetxt(save_file, segment_stats[skel_id], delimiter = ",")




if __name__ == '__main__':
    main()
