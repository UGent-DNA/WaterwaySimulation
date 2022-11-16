import os
import random

# Configuration and definitions file for this project.

########################################################################################################################
# Prepare data. This should only be done once! And before you try to run experiments ;)
# Transform .txt files to a pickled dataclass instances
# This step needs to be repeated if you change any of the base classes defined in sample/data_processing/data_classes.py
resource_dir = "ship_movements"
transform_csv_to_pkl = True

########################################################################################################################
# What experiment do you want to run?
# If experiments = 'all', run all experiments. If 'visual' create an MP4.
# If 'data_analysis' run a full suite analysis on original data.
# If 'experiment_analysis' run analysis on experiment results => You first have to run experiments!.
experiments = "all"

# If you want to add additional data, make sure to change the _get_trips method in sample/data_processing/read_data.py
# Defines what part of the data is used for the experiments or visualisation. Select category:
# 1. all days, 2. training days, 3. testing days)
select_trip_days = [(month, day) for month in [3, 4] for day in range(32)]
# select_trip_days = [(month, day) for month in [3, 4] for day in range(32) if not (month == 4 and 18 <= day)]
# select_trip_days = [(month, day) for month in [4] for day in range(18, 25)]
save_location = "experiments_trial"  # where to save experiment data / read exp data for analysis

########################################################################################################################
# Run full experiment suite: run each combination (auto-excludes non-relevant combinations)
n_iterations = 200  # Number of iterations for each scenario + model.
type_of_day_list = ["midweek", "weekend"]
trips_per_day_list = [200, 250, 300]  # Number of trips (300 is peak, 250 is busy), weekend auto-switches to 150.
one_way_area_end_list = ["AK5"]  # 'AK3' (short single direction SD) 'AK5' (long SD) or 'base' (no SD)
seaship_stats = [0, 2]  # Use 0 for no seaship, 1-3 for different seaship schedules. See simulation/model.py
models = ["RegOpt", "Reg15x15", "Reg30x15", "Reg45x15", "Dynamic"]  # See simulation/model.py

########################################################################################################################
# Run single experiment
type_of_day = type_of_day_list[0]
ntrips = trips_per_day_list[2]
one_way_area_end = one_way_area_end_list[0]
seaship_status = seaship_stats[0]
model_type = models[1]

########################################################################################################################
# Run visualise [use scenario from single]
hour_range = range(25)  # Which hours of the day to visualise (takes about 10 min per hour).
draw_convoy = False  # If true, draw detected convoys (can slow visualisation down)

########################################################################################################################
# Run experiment_analysis: generate figures and tables based on experiments
generate_figures_queue_unresolved = True
generate_figures_time_to_queue = True
table_extension = "latex"  # Options are 'latex' and 'markdown'. Add additional types in sample/general/table_writer.py


########################################################################################################################
# Specify file directories based on project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is the Project Root
RESOURCE_PATH = os.path.join(ROOT_DIR, 'resources')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'output')

########################################################################################################################
# Fix random for experiment duplication
random.seed(14967516667)


def change_seed(seed: int):
    random.seed(seed)


########################################################################################################################
# Rectangle coordinate specifications
is_east = 4.4017
is_west = 4.40807
is_north = 51.2451
is_south = 51.24192
AK5_south = 51.2330
LOCATIONS = {
    "Albertdok": (51.2503, 4.4033, is_north, 4.4098),
    "Amerikadok": (is_north, 4.3971, 51.2425, is_east),
    "Ac": (51.2425, 4.4006, 51.2416, is_east),  # Amerikadok_corner
    "Royerssluis_S": (51.2416, 4.39906, 51.23946, is_east),  # Royerssluis_Schelde
    "Royerssluis_i": (is_south, is_east, 51.24096, 4.40508),  # Royerssluis_intersectie
    "Suezdok": (is_south, 4.40508, 51.23979, is_west),
    "Straatsburgdok_east": (51.24364, 4.41279, is_south, 4.41973),
    "Straatsburgdok_west": (is_south, 4.41786, 51.24106, 4.42026),
    "Asiadok": (51.24000, 4.41786, 51.23816, 4.41974),
    "Lobroekdok": (AK5_south, 4.4306, 51.2289, 4.4411),
    "Albertkanaal": (51.2359, 4.43982, AK5_south, 4.447),
    "Fork_narrow": (51.2400, 4.4231, 51.2376, 4.4273),
    "Intersection": (is_north, is_east, is_south, is_west),
    "AK1": (51.2435, is_west, 51.241, 4.41279),
    "AK2": (is_south, 4.41279, 51.24000, 4.41786),
    "AK3": (51.24106, 4.41786, 51.24000, 4.41974),
    "AK4": (51.24106, 4.41974, 51.23816, 4.4231),
    "AK5": (51.23873, 4.4273, AK5_south, 4.43982)}

loc_eng = {"Albertdok": "Albert dock", "Amerikadok": "America dock", "Albertkanaal": "Albert Canal",
           "Diverse": "Varia", "Stadshaven": "City dock", "Royerssluis_S": "Royers Lock S",
           "Royerssluis_i": "Royers Lock i", "Suezdok": "Suez dock", "Straatsburgdok_east": "Straatsburg dock east",
           "Straatsburgdok_west": "Straatsburg dock west", "Asiadok": "Asia dock", "Lobroekdok": "Lobroek dock"}

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
