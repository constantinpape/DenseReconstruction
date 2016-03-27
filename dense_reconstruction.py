import vigra
import re
import numpy as np
import os
import cPickle as pickle
from scipy import sparse

#
# watershed on distance transform
# this is mostly copied from our implementation
# which is not public right now (we plan on making it public soon, then I will replace this)
#

def remove_wrongly_sized_connected_components(a, min_size,
        max_size=None,
        in_place=False, bin_out=False):
    """
    Copied from lazyflow.operators.opFilterLabels.py
    Originally adapted from http://github.com/jni/ray/blob/develop/ray/morpho.py
    (MIT License)
    """
    original_dtype = a.dtype

    if not in_place:
        a = a.copy()
    if min_size == 0 and (max_size is None or max_size > np.prod(a.shape)): # shortcut for efficiency
        if (bin_out):
            np.place(a,a,1)
        return a

    try:
        component_sizes = np.bincount( a.ravel() )
    except TypeError:
        # On 32-bit systems, must explicitly convert from uint32 to int
        # (This fix is just for VM testing.)
        component_sizes = np.bincount( np.asarray(a.ravel(), dtype=int) )
    bad_sizes = component_sizes < min_size
    if max_size is not None:
        np.logical_or( bad_sizes, component_sizes > max_size, out=bad_sizes )

    bad_locations = bad_sizes[a]
    a[bad_locations] = 0
    if (bin_out):
        # Replace non-zero values with 1
        np.place(a,a,1)
    return np.array(a, dtype=original_dtype)


# get the signed distance transform of pmap
def getSignedDt(pmap, pmin, minMembraneSize):

    # get the thresholded pmap
    binary_membranes = np.zeros_like(pmap, dtype=np.uint8)
    binary_membranes[pmap >= pmin] = 1

    # delete small CCs
    labeled = vigra.analysis.labelImageWithBackground(binary_membranes)
    remove_wrongly_sized_connected_components(labeled, minMembraneSize, in_place=True)

    # use cleaned binary image as mask
    big_membranes_only = np.zeros_like(binary_membranes, dtype = np.float32)
    big_membranes_only[labeled > 0] = 1.

    # perform signed dt on mask
    distance_to_membrane    = vigra.filters.distanceTransform2D(big_membranes_only)
    distance_to_nonmembrane = vigra.filters.distanceTransform2D(big_membranes_only, background=False)
    distance_to_nonmembrane[distance_to_nonmembrane>0] -= 1
    dtSigned = distance_to_membrane - distance_to_nonmembrane
    dtSigned[:] *= -1
    dtSigned[:] -= dtSigned.min()

    return dtSigned


# seeds on minima of the distance trafo
def get_wsdt_seeds(dist_trafo, sigma):
    trafo_smoothed = vigra.filters.gaussianSmoothing(
            dist_trafo, sigma)

    seedsVolume = vigra.analysis.localMinima(
            trafo_smoothed,
            neighborhood=8,
            allowPlateaus=True,
            allowAtBorder=True)
    seedsLabeled = vigra.analysis.labelImageWithBackground(seedsVolume)
    return seedsLabeled


# watershed on distance trafo
def watersheds_dt(probs_2d, thresh, sigma):
        dist_trafo = getSignedDt(probs_2d, thresh, 50)
        seeds_wsdt = get_wsdt_seeds(dist_trafo, sigma)
        seg_wsdt = vigra.analysis.watershedsNew(dist_trafo, seeds = seeds_wsdt)[0]
        seg_wsdt = vigra.analysis.labelImage(seg_wsdt)
        return seg_wsdt


def dense_reconstruction_slice(probs_slice_path, skeletons_volume, rf):
    probs = vigra.readImage(probs_slice_path)
    probs = np.squeeze(probs)
    probs = np.array(probs) # get rid of axistags...
    # need to invert for watersheds
    probs = 1. - probs

    # Threshold and sigma hardcoded to values suited for the google pmaps
    seg_wsdt = watersheds_dt(probs, 0.15, 2.6)

    # this may happen for the black slices...
    if np.unique(seg_wsdt).shape[0] == 1:
        return np.zeros_like(probs, dtype = np.uint32)

    rag = vigra.graphs.regionAdjacencyGraph(
            vigra.graphs.gridGraph(seg_wsdt.shape[0:2]), seg_wsdt )

    # +1 because we dont have a zero in overseg
    merge_nodes = np.zeros(rag.nodeNum+1, dtype = np.uint32)

    gridGraphEdgeIndicator = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph,
            probs)
    edge_feats = rag.accumulateEdgeStatistics(gridGraphEdgeIndicator)
    edge_feats = np.nan_to_num(edge_feats)
    edge_probs = rf.predict_proba(edge_feats)[:,1]
    probs_thresh = 0.5

    for skel_id in np.unique(skeletons_volume):
        if skel_id == 0:
            continue
        skel_c = np.where(skeletons_volume == skel_id)
        for i in xrange(skel_c[0].size):
            seg_id = seg_wsdt[ skel_c[0][i], skel_c[1][i] ]
            merge_nodes[seg_id] = skel_id
            root = rag.nodeFromId( long(seg_id) )
            nodes_for_merge = [root]
            already_merged  = [root]
            while nodes_for_merge:
                u = nodes_for_merge.pop()
                for v in rag.neighbourNodeIter(u):
                    edge = rag.findEdge(u,v)
                    #print edge_mean_probs[edge.id]
                    if edge_probs[edge.id] <= probs_thresh and not v.id in already_merged:
                        merge_nodes[v.id] = skel_id
                        nodes_for_merge.append(v)
                        already_merged.append(v.id)
    return rag.projectLabelsToBaseGraph(merge_nodes)


# dense reconstruction using random forest for agglomeration decisions
def dense_reconstruction(prob_path, skeleton_coordinates, rf_path, shape):

    seg = np.zeros(shape, dtype = np.uint32)

    with open(rf_path, 'r') as f:
        rf = pickle.load(f)

    sparse_skeletons = []
    for z in xrange(shape[2]):
        sparse_skeletons.append( sparse.dok_matrix( (shape[0], shape[1]),
            dtype = np.uint32 ) )
    for skel_id in skeleton_coordinates:
        for c in skeleton_coordinates[skel_id]:
            sparse_skeletons[ int(c[2]) ][int(c[0]), int(c[1])] = skel_id

    # get the filenames
    files = os.listdir(prob_path)
    assert len(files) == shape[2]
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

    # iterate over z slices and reconstruct
    for z in xrange(seg.shape[2]):
        print "Processing slice", z, "/", seg.shape[2]
        probs_slice_path = filepaths[z]
        seg[:,:,z] = dense_reconstruction_slice(probs_slice_path, sparse_skeletons[z].toarray(), rf)

    return seg
