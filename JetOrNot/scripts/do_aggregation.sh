#!/bin/bash

mkdir -p {extracts,reductions}/

# first get the extracts
cd extracts/;
panoptes_aggregation extract ../jet-or-not-classifications.csv ../configs/Extractor_config_workflow_18563_V5.19.yaml -o jet_or_not

# remove the beta classifications
python3 ../scripts/trim_beta_classifications.py question_extractor_jet_or_not.csv

# then do the reductions
cd ../reductions/
panoptes_aggregation reduce ../extracts/question_extractor_trimmed.csv \
	../configs/Reducer_config_workflow_18563_V5.19_question_extractor.yaml -o jet_or_not

# finally create the CSV files for the subject agreement
cd ../
python3 scripts/make_T0_csvfiles.py
