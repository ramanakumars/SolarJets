#!/bin/bash

# first get the extracts
panoptes_aggregation extract box-the-jets-classifications.csv Extractor_config_workflow_19650_V4.52.yaml -o test

# squash the frames
python3 squash_frames.py

# then do the reductions
panoptes_aggregation reduce point_extractor_by_frame_test_squashed.csv \
    Reducer_config_workflow_19650_V4.52_point_extractor_by_frame.yaml -o box_the_jets_merge_tasks
panoptes_aggregation reduce shape_extractor_rotateRectangle_test_squashed.csv\
    Reducer_config_workflow_19650_V4.52_shape_extractor_rotateRectangle.yaml -o box_the_jets_merge_tasks