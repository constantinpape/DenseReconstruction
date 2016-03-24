import vigra
import numpy as np
import os
import json


# TODO interpolation for virtual nodes
def coordinates_from_json(skeleton_path, bounding_box):

    data = json.load( open(skeleton_path) )

    skeletons = {}

    bounding_size = (bounding_box[1] - bounding_box[0],
            bounding_box[3] - bounding_box[2],
            bounding_box[5] - bounding_box[4])

    n_nodes = 0
    for skeleton_id in data["skeletons"]:
        #print skeleton_id
        for treenode_id in data["skeletons"][skeleton_id]["treenodes"]:
            if "include" in data["skeletons"][skeleton_id]["treenodes"][treenode_id]:
                if "error" in data["skeletons"][skeleton_id]:
                    continue
                [x,y,z] = data["skeletons"][skeleton_id]["treenodes"][treenode_id]["location"]
                x = (x // 4)  - bounding_box[0]
                y = (y // 4)  - bounding_box[2]
                z = (z // 35) - bounding_box[4]
                if x >= 0 and y >= 0 and z >= 0 and x < bounding_size[0] and y < bounding_size[1] and z < bounding_size[2]:
                    if skeleton_id in skeletons:
                        skeletons[skeleton_id].append([x, y, z])
                    else:
                        skeletons[skeleton_id] = [ [ x, y, z] ]
                    n_nodes += 1

    return skeletons
