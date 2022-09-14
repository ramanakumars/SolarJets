# Box The Jets workflow

This features the set of scripts to aggregate the 'Box the Jets' workflow, 
which asks volunteers to annotate the start and end of jets and also draw a box around them.

### Preparing the aggregation pipeline
To get the extracts, you will need the workflow export as well as the classification exports from the Zooniverse project builder. Open [zooniverse.org/lab](https://www.zooniverse.org/lab), select the Solar Jet Hunter project and go to the Data Exports tab. Click on "Request new workflow classification export" and select the "Box the Jets" workflow. You will receive and email when these are ready. The workflow export does not generally need to be regenerated, unless the workflows have been changed, so we can just download the existing one. Save both these files in a directory. In this repo the workflow files are in the main folder, and the classifications will be saved in this folder. 

### Aggregating the data
Open a terminal in the directory where both those CSVs are and make sure that `panoptes_aggregation` is installed. Then, we will create the extractor and reducer configurations for the panoptes aggregation module by running:
```bash
panoptes_aggregation config ../solar-jet-hunter-workflows.csv 19650
```

19650 is the workflow ID for this workflow. This will generate 5 files: the extractor config, three reducer configs (one for point, one for the box and one for the question) and the task labels. We need to modify the reducer files, so that they match the following:

`Reducer_config_workflow_19650_V4.52_shape_extractor_rotateRectangle.yaml`:
```yaml
reducer_config:
    shape_reducer_hdbscan:
        shape: rotateRectangle
        min_cluster_size: 2
        min_samples: 2
        allow_single_cluster: True
        metric_type: IoU
```


`Reducer_config_workflow_19650_V4.52_point_extractor_by_frame.yaml`:
```yaml
reducer_config:
    point_reducer_hdbscan: 
        min_cluster_size: 2
        min_samples: 2
        allow_single_cluster: True
```

Move these files into the `configs/` folder. 

This is to use the HDBSCAN algorithm rather than the default, and also allow the clustering pipeline to generate one cluster when the data is tightly packed. Now, to run the aggregation, call the `do_aggregration.sh` script from the `BoxTheJets/` folder:

```bash
scripts/do_aggregation.sh
```

which will do the following

1. Run the extraction on the `box-the-jets-classification.csv` file
2. Squash the frames so that all the data is in one frame, and also create a separate datafile where the tasks (Jet 1 and 2) are merged together.
3. Run the reducer which will cluster the data 

Each set of files will be moved into their respective folder (extract files in `extracts/` and the HDBSCAN 
reduced cluster data in `reductions/`)
