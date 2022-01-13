Jet or Not workflow
===================

This features the set of scripts to aggregate the workflow (a binary task), which asks whether a given subject contains a jet. 

## Getting the data
To do the aggregation, you will need the [panoptes aggregation app](https://github.com/zooniverse/aggregation-for-caesar/) installed for offline use as well as the raw classifications. See [here](https://aggregation-caesar.zooniverse.org/README.html) on how to install Panoptes for offline use.

### Preparing the aggregation pipeline
To get the extracts, you will need the workflow export as well as the classification exports from the Zooniverse project builder. Open [zooniverse.org/lab](https://www.zooniverse.org/lab), select the Solar Jet Hunter project and go to the Data Exports tab. Click on "Request new classification export". You will receive and email when these are ready. The workflow export does not generally need to be regenerated, unless the workflows have been changed, so we can just download the existing one. Save both these files in a directory.

### Getting the extracts
Open a terminal in the directory where both those CSVs are and make sure that `panoptes_aggregation` is installed. Then, we will create the extractor and reducer configurations for the panoptes aggregation module by running:
```bash
panoptes_aggregation panoptes_aggregation config solar-jet-hunter-workflows.csv 18563
```

18563 is the workflow ID of the Jet or Not workflow, so this creates the configuration for extracting and reducing the data for this workflow. This will generate three files: `Extractor_config_workflow_18563_V[xx].[xx].yaml`,  `Reducer_config_workflow_18563_V[xx].[xx]_question_extractor.yaml` and  `Task_labels_workflow_18563_V[xx].[xx].yaml` where the two `[xx]` are version numbers for the workflow. In my case, they are 5.19.

Now, we can generate the extracts from the classification data by doing:
```bash
panoptes_aggregation extract solar-jet-hunter-classifications.csv Extractor_config_workflow_18563_V5.19.yaml -o jet_or_not
```

This will generate the `question_extract_jet_or_not.csv` which contains all the extracts from the classification data. Note that for this workflow, this also includes the beta and charlie classifications, so we need to trim those. To do this run,
```bash
python3 trim_beta_classifications.py
```

This will produce a `question_extractor_trimmed.csv` which only contains classifications after Dec 7, 2021. This is what we will use for reductions

### Generating the reduced data
Now, let's generate the reducted data. To do this, run:
```bash
panoptes_aggregation reduce question_extractor_trimmed.csv Reducer_config_workflow_18563_V5.19_question_extractor.yaml -o jet_or_not 
```

This generates the `question_reducer_jet_or_not.csv` file which contains the per-subject reduced data, which shows how many volunteers selected each answer for each subject. 