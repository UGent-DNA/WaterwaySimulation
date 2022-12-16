# WaterwaySimulator

Data analysis and simulation of vessels based on AIS data on the waterway connecting the harbour of Antwerp with the Albert canal
Project in collaboration with Port of Antwerp-Bruges.

### Get started

1. Add the data files from [data_link](https://doi.org/10.17605/OSF.IO/5KDMZ) in the resources folder
2. Set up the configuration file to create pkl files (transform_csv_to_pkl=True) and run experiments (experiments = "all")
3. Run sample/main.py
4. Disable creating .pkl files (to avoid doing double work: transform_csv_to_pkl=False)
5. Set up configuration file to analyse the results (experiments = "experiment_analysis")
6. Enjoy your tables and figures in the output directory

### Code structure

The configuration.py file makes it possible to make small changes to the run configuration: which experiments to perform, what data to include, which file destinations to use...
After changing configurations.py, run the program by running sample/main.py
