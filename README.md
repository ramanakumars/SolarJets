# SolarJets
Tools and scripts for working with the SolarJets zooniverse project

## Requirements
To install the required python modules, run the following in main repo folder:
```bash
python3 -m pip install -r requirements.txt
```

## Usage


``` bash
# Start in the JetOrNot directory
cd JetOrNot
``` 

# Run the aggregation on the JetOrNot workflow results
scripts/do_aggregation.sh 

GO into README Jet or not

#Check the results by looking at ipynb notebooks

#Look at the observation times of jet subject and non-jet subjects
jet_time_distribution.ipynb
#Look at the agreement and the number of votes of the subjects
jetornot.ipynb
# Plotting agreement of the answers given per subjects over time from the Jet or Not workflow. Sorted by SOL/ HEK event
Plotting_agreement_T0.ipynb

# return to main directory
cd ..

# Run the aggregation on the BoxTheJets workflow results
scripts/do_aggregation.sh 

#Check the results
