# Dense Reconstruction of neurons from skeletons

Reconstruct dense neuron segmentation from skeletons.

To run this, you need: 

- vigra
- sklearn

Input data:

- membrane probability maps for the area of interest
- json file with skeleton coordinates
- bounding box coordinates to map from skeleton coordinates to probability maps
- pretrained random forest for oversegmentation edges (I can provide this on request)

Usage:

python make_dense 
