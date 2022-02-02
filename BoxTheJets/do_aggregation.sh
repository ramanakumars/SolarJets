#!/bin/bash

# first get the extracts
panoptes_aggregation extract box-the-jets-classifications.csv Extractor_config_workflow_19650_V4.52.yaml -o box_the_jets

# squash the frames
python3 squash_frames.py

# then do the reductions
panoptes_aggregation reduce point_extractor_by_frame_box_the_jets_squashed.csv \
    Reducer_config_workflow_19650_V4.52_point_extractor_by_frame.yaml -o box_the_jets
panoptes_aggregation reduce shape_extractor_rotateRectangle_box_the_jets_squashed.csv\
    Reducer_config_workflow_19650_V4.52_shape_extractor_rotateRectangle.yaml -o box_the_jets