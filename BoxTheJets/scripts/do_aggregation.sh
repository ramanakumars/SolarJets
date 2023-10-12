usage() { 
	echo "Usage: $0 [-c <int>]"; 
	echo "     -c Number of processors to use for reductions [default=1]" 1>&2; 
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
panoptes_aggregation extract ../box-the-jets-classifications.csv\
	../configs/Extractor_config_workflow_21225_V50.59.yaml -o box_the_jets

# squash the frames
cd ..;
python3 scripts/squash_frames.py

# then do the reductions
cd reductions/
panoptes_aggregation reduce ../extracts/shape_extractor_temporalPoint_box_the_jets_merged.csv \
    ../configs/Reducer_config_workflow_21225_V50.59_pointExtractor_temporalPoint.yaml -o box_the_jets -c ${NUM_PROCS}
panoptes_aggregation reduce ../extracts/shape_extractor_temporalRotateRectangle_box_the_jets_merged.csv \
   ../configs/Reducer_config_workflow_21225_V50.59_shapeExtractor_temporalRotateRectangle.yaml -o box_the_jets -c ${NUM_PROCS}

# get and create the subject metadata
cd ..;
panoptes project download -t subjects 11265 ../solar-jet-hunter-subjects.csv
python3 scripts/create_subject_metadata.py
