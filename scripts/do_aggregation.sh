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

mkdir -p { extracts , reductions }

# first do the JetOrNot workflow
cd extracts &&
panoptes_aggregation extract ../JetOrNot/jet-or-not-classifications.csv\
	../configs/Extractor_config_workflow_25059_V2.15.yaml -o jet_or_not;

cd ../reductions &&
panoptes_aggregation reduce ../extracts/question_extractor_jet_or_not.csv \
   ../configs/Reducer_config_workflow_25059_V2.15_question_extractor.yaml -o jet_or_not -c ${NUM_PROCS}

# get and create the subject metadata
cd .. &&
panoptes project download -t subjects 11265 ../solar-jet-hunter-subjects.csv &&
python3 scripts/create_subject_metadata.py

# first get the extracts
cd extracts &&
panoptes_aggregation extract ../BoxTheJets/box-the-jets-classifications.csv\
	../configs/Extractor_config_workflow_21225_V50.59.yaml -o box_the_jets

# squash the frames
cd .. &&
python3 scripts/squash_frames.py

# then do the reductions
cd reductions/ &&
panoptes_aggregation reduce ../extracts/shape_extractor_temporalPoint_box_the_jets_merged.csv \
    ../configs/Reducer_config_workflow_21225_V50.59_pointExtractor_temporalPoint.yaml -o box_the_jets -c ${NUM_PROCS} &&
panoptes_aggregation reduce ../extracts/shape_extractor_temporalRotateRectangle_box_the_jets_merged.csv \
   ../configs/Reducer_config_workflow_21225_V50.59_shapeExtractor_temporalRotateRectangle.yaml -o box_the_jets -c ${NUM_PROCS}


# get the unique jets
python3 scripts/get_unique_jets.py && python3 scripts/cluster_jets_by_sol.py
