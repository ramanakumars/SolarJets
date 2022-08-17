Jet or Not workflow
===================

This features the set of scripts to aggregate the workflow (a binary task), which asks whether a given subject contains a jet. 

## Getting the data
To do the aggregation, you will need the [panoptes aggregation app](https://github.com/zooniverse/aggregation-for-caesar/) installed for offline use as well as the raw classifications. See [here](https://aggregation-caesar.zooniverse.org/README.html) on how to install the aggregation tool for offline use, or follow the [installation procedure](https://github.com/ramanakumars/SolarJets/blob/main/README.md/) for the main repo to install everything.

### Preparing the aggregation pipeline
To get the extracts, you will need the workflow export as well as the classification exports from the Zooniverse project builder. Open [zooniverse.org/lab](https://www.zooniverse.org/lab), select the Solar Jet Hunter project and go to the Data Exports tab. 

The workflow export does not generally need to be regenerated, unless the workflows have been changed, so we can just download the existing one by clicking on the "download your data export" next to "Request new workflow export". Save this file in the main directory.

For the classifications export, click on "Request new workflow classification export" and select the "Jet or Not" workflow and click "Export". You will receive and email when these are ready. Save this file in the `JetOrNot/` directory.

Alternatively, you can download the new classifications directly. Install the `panoptes-cli` module with pip:
```bash
pip install panoptescli
```

Configure your Zooniverse account by entering your username and password:
```bash
panoptes configure
```

Now, in the `JetOrNot/` folder, you can run the following command to generate and download the classification data for this workflow:
```bash
panoptes workflow download-classifications -g 18563 jet-or-not-classifications.csv
```

### Doing the extraction and reduction of the classification data
Open a terminal in the `JetOrNot` directory and make sure that `panoptes_aggregation` is installed (check [https://github.com/ramanakumars/SolarJets/blob/main/README.md](this README) for more details). Then, we will create the extractor and reducer configurations for the panoptes aggregation module by running:

```bash
panoptes_aggregation config ../solar-jet-hunter-workflows.csv 18563
```

18563 is the workflow ID for this workflow. This will generate 3 files: the extractor config, one reducer configs and the task labels. Move these files into the `configs/` folder (create the directory if it doesn't exist).

Now, to run the aggregation, call the `do_aggregration.sh` script from the `JetOrNot/` folder:
```bash
scripts/do_aggregation.sh
```

which will do the following:

1. Run the extract on the `jet-or-not-classifications.csv`

2. Trim the extracts to remove the beta test classifications

3. Run the reducer so as to get the volunteer responses for each subject

## Running the analysis
To run the analysis, you will need the [Panoptes Python Client](https://github.com/zooniverse/panoptes-python-client) to interface with the Panoptes backend to retrieve subject metadata. This is installed alongside the other dependencies if you followed the [installation procedure](https://github.com/ramanakumars/SolarJets/blob/main/README.md) in the main repo page. 

Check the `jetornot.ipynb` for the full analysis or `jet_time_distribution.ipynb` for getting the temporal distribution of jets. 


## Under the hood of `do_aggregation.sh`

These are the individual components of the aggregation script:

### Getting the extracts
Now, we can generate the extracts from the classification data by doing (in the `extracts/` directory):
```bash
panoptes_aggregation extract ../jet-or-not-classifications.csv ../configs/Extractor_config_workflow_18563_V5.19.yaml -o jet_or_not
```

### Trimming the beta classifications
This will generate the `question_extractor_jet_or_not.csv` which contains all the extracts from the classification data. Note that for this workflow, this also includes the beta and charlie classifications, so we need to trim those. To do this run,
```bash
python3 ../scripts/trim_beta_classifications.py question_extractor_jet_or_not.csv
```

This will produce a `question_extractor_trimmed.csv` which only contains classifications after Dec 7, 2021. This is what we will use for reductions

### Generating the reduced data
Now, let's generate the reducted data. To do this, run (from the `reductions/` directory):
```bash
panoptes_aggregation reduce ../extracts/question_extractor_trimmed.csv ../configs/Reducer_config_workflow_18563_V5.19_question_extractor.yaml -o jet_or_not 
```

This generates the `question_reducer_jet_or_not.csv` file which contains the per-subject reduced data, which shows how many volunteers selected each answer for each subject. 

