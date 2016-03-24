# Dense Reconstruction of neurons from skeletons

Reconstruct dense neuron segmentation from skeletons.

To run this, you need: 

- vigra
- sklearn
- scipy

Input data:

- membrane probability maps for the area of interest as tif files
- json file with skeleton coordinates
- bounding box coordinates to map from skeleton coordinates to probability maps
    -  these have to be stored in the first line of a file, in the order: x_min, x_max, y_min, y_max, z_min, z_max
- pretrained random forest for oversegmentation edges (I will provide this)

Usage:

```python
python make_dense path_to_probability_maps path_to_exported_skeletons
path_to_rf bounding_box.txt
```

For storing the results as tifs, you can give an optional folder:

```python
python make_dense path_to_probability_maps path_to_exported_skeletons
path_to_rf bounding_box.txt --output_folder folder_for_results
```

## Statistics

Resulting statistics:
Have to discuss what you need


## Possible Issues

- You need proper cutouts of probability maps
- Scaling for large volumes (3k x 3k x 200 takes ~ 1/2 hour on my laptop)
- Reconstruction might fail for some cells, but for regular KCs it should work ok
    - Also I can improve this with more features for the Random Forest or a suited agglomeration procedure.
