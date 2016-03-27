# Dense Reconstruction of neurons from skeletons

Repo for constructing dense neuron segments from CATMAID skeletons, 
given probability maps.

To run this, you need: 

- vigra (you need the current master!)
- sklearn
- scipy

Input data:

- membrane probability maps for the area of interest as tif files
- json file with skeleton coordinates
- bounding box coordinates to map from skeleton coordinates to probability maps
    -  these have to be stored in the first line of a file, in the order: x_min, x_max, y_min, y_max, z_min, z_max
- pretrained random forest for oversegmentation edges (I will provide this)

There are two scripts, which implement different functionality:

* make_dense: Constructs the dense segmentation for all skeletons and saves it.
* get_statistcs: Performs the reconstruction and extracts relevant statistics over the segments.

## Usage:

For make_dense:

```bash
python make_dense path_to_probability_maps path_to_exported_skeletons
path_to_rf bounding_box.txt folder_for_results
```
The results will be stored as tif slices.


For get_statistics:

```bash
python get_statistcs path_to_probability_maps path_to_exported_skeletons
path_to_rf bounding_box.txt folder_for_results --debug_folder folder_for_images
```

The results will be stored as csv (one csv for each skeleton),
containing the following statistics (in units of pixel) for every slice:

* Area
* x-Radius
* y-Radius
* Radius

Values for a skeleton id, which is not present in a given slice, are zero.

The argument debug_folder is optional, if given, the reconstructions will be saved in the given folder.

## Possible Issues

- You need proper cutouts of probability maps
- So far, I haven't come around to look at the interpolation of virtual nodes, so there
are some segments missing, but I will do this once I work on learning from skeletons.
- Reconstruction might fail for some cells, but for regular KCs it should work ok
    - Also I can improve this with more features for the Random Forest or a suited agglomeration procedure.
