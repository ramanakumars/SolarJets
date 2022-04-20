#!/bin/bash

usage() { 
	echo "Usage: $0 [-c <int>]"; 
	echo "     -c : Number of processors to use for reductions [default=1]" 1>&2; 
	exit 1; 
}

NUM_PROCS=1;

while getopts ":c:" options; do 
	case "${options}" in 
		c) 
			NUM_PROCS=${OPTARG}
			;;
		*) 
			usage
			;;
	esac
done

mkdir -p {extracts,reductions}/

# first get the extracts
cd extracts/;
panoptes_aggregation extract ../box-the-jets-classifications.csv ../configs/Extractor_config_workflow_19650_V4.52.yaml -o box_the_jets

# squash the frames
cd ..;
python3 scripts/normalize_subject_size.py
python3 scripts/squash_frames.py

# then do the reductions
cd reductions/
panoptes_aggregation reduce ../extracts/point_extractor_by_frame_box_the_jets_scaled_squashed.csv \
    ../configs/Reducer_config_workflow_19650_V4.52_point_extractor_by_frame.yaml -o box_the_jets
#panoptes_aggregation reduce ../extracts/point_extractor_by_frame_box_the_jets_squashed_merged.csv \
#    ../configs/Reducer_config_workflow_19650_V4.52_point_extractor_by_frame.yaml -o box_the_jets_merged

# Using the old euclidian metric for clustering
#panoptes_aggregation reduce ../extracts/shape_extractor_rotateRectangle_box_the_jets_squashed.csv\
#    ../configs/Reducer_config_workflow_19650_V4.52_shape_extractor_rotateRectangle.yaml -o box_the_jets

# Using the new jaccard metric for clustering
panoptes_aggregation reduce ../extracts/shape_extractor_rotateRectangle_box_the_jets_scaled_squashed.csv \
    ../configs/Reducer_config_workflow_19650_V4.52_shape_extractor_rotateRectangle_IoU.yaml \
	-o box_the_jets -c ${NUM_PROCS}

panoptes_aggregation reduce ../extracts/question_extractor_box_the_jets.csv\
    ../configs/Reducer_config_workflow_19650_V4.52_question_extractor.yaml -o box_the_jets
