# Dense Reconstruction of neurons from skeletons

Reconstruct dense neuron segmentation from skeletons.

To run this, you need: 

- vigra
- sklearn

Input data:

- membrane probability maps for the area of interest
- json file with skeleton coordinates
- bounding box coordinates to map from skeleton coordinates to probability maps
    - format: (x_min, x_max, y_min, y_max, z_min, z_max)
- pretrained random forest for oversegmentation edges (I can provide this on request)

Usage:

python make_dense path_to_probability_maps key_to_probability_maps path_to_exported_skeletons
path_to_rf bounding_box
