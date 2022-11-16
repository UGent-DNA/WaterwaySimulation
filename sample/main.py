import time
from typing import List

import configurations as conf
from sample.analysis.animate_fleet import AnimateFleet
from sample.analysis.capacity_analysis import get_throughput_fork
from sample.analysis.distribution_by_day import get_ship_distributions, weekday_analysis
from sample.analysis.intersection_analysis import make_plots
from sample.data_processing.data_classes import VesselTrip
from sample.data_processing.read_data import get_filtered_trips, read_vessel_data, pickle_days
from sample.data_processing.pseudonymizer import data_to_csv
from sample.general.convoy_detection import find_convoy, trips_in_convoy
from sample.general.util_ships import detect_intersection_as_turning_bowl
from sample.simulation.experiment_analysis import run_queue_over_time, run_tests_time
from sample.simulation.simulation_controller import SimulationRunner


def animate_fleet():
    """ Create mp4 with moving ships. """
    vessel_trips_o, name = SimulationRunner("vis").run_sim_single()
    # vessel_trips = [vt for vt in vessel_trips_o if vt.trip[0].time.hour in conf.hour_range]

    hour_range = conf.hour_range
    vessel_trips = [vt for vt in vessel_trips_o if vt.trip[0].time.hour in hour_range]
    print("Number of vesseltrips:", len(vessel_trips))

    precision = 10
    save_name = f"{name}_hour{hour_range[0]}_{hour_range[-1]}_prec{precision}"
    AnimateFleet(vessel_trips, precision, save_name)


def get_location_rectangles():
    """ Get map with location rectangles. """
    rectangle_creator = AnimateFleet([], 10, "")
    rectangle_creator.get_location_rectangles()


def run_analysis_for_figures(vessel_trips: List[VesselTrip]):
    """ Create a lot of very interesting figures based on the data. """
    # Get daily distributions and weekly averages.
    for ship_count in ["number", "time"]:
        ship_distrib = get_ship_distributions(True, ship_count)
        weekday_analysis(ship_distrib, ship_count)

    # Create plots with intersection time. Including plots with convoys.
    make_plots(vessel_trips)


def print_data_analysis_elements(vessel_trips: List[VesselTrip]):
    """ Print several data analysis elements to std::out. """
    # List all ships in convoy.
    convoys = find_convoy(vessel_trips)
    trips_in_convoys = trips_in_convoy(convoys)
    for vt in sorted(trips_in_convoys, key=lambda vt_: vt_.trip[0].time):
        print(vt.trip[0].time, vt.get_path(), vt.vessel)

    # Print large vessels that use the intersection as turning bowls.
    detect_intersection_as_turning_bowl(vessel_trips)

    # Print the saturation of the network at peak times.
    get_throughput_fork(vessel_trips)


if __name__ == "__main__":
    t_start = time.time()

    if conf.transform_csv_to_pkl:
        pickle_days()

    if conf.experiments == "single":
        sr = SimulationRunner()
        sr.run_sim_single()
    elif conf.experiments == "visual":
        animate_fleet()
    elif conf.experiments == "all":
        sr = SimulationRunner()
        sr.run_all_experiments()
    elif conf.experiments == "data_analysis":
        vts = get_filtered_trips(conf.select_trip_days)
        get_location_rectangles()
        run_analysis_for_figures(vts)
        print_data_analysis_elements(vts)
    elif conf.experiments == "experiment_analysis":
        queues_by_time = run_queue_over_time()
        run_tests_time(queues_by_time)
    elif conf.experiments == "iterative_improvements":
        sr = SimulationRunner()
        # sr.run_iterative_improvements()
        sr.run_differential_evolution()
    else:
        raise ValueError()

    print()
    print("Experiments done, total time:", time.time() - t_start)
