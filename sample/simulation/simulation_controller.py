import os
import random
import shutil
import time
from typing import Set, List, Tuple

import dill
import networkx as nx
import simpy
from scipy.optimize import differential_evolution

import configurations as conf
from configurations import OUTPUT_PATH
from sample.data_processing.data_classes import VesselTrip
from sample.data_processing.read_data import get_filtered_trips
from sample.simulation.experiment_analysis import get_results
from sample.simulation.model import Model
from sample.simulation.simulator import SimulationDataStructures, Simulation
from sample.simulation.tripstate import Location, TripState


def verify_log_files(directory):
    """ Make sure that you do not overwrite an existing directory; if so requests a change. """
    base_path = os.path.join(OUTPUT_PATH, directory, "logs")
    try:
        os.makedirs(base_path, exist_ok=False)
    except OSError:
        print(f"Clear contents of {base_path}? (Y/N)")
        answer = input().strip().lower()
        if answer == "y":
            shutil.rmtree(base_path)
            print("Directory cleared successfully, start experiments.")
        else:
            print("Use new location for logging? (Location name / N)")
            answer = input().strip()
            if answer.lower() == "n":
                raise OSError(f"Directory {base_path} should be clear to log experiments.")
            else:
                base_path = os.path.join(OUTPUT_PATH, answer, "logs")
                print("Writing to new location, start experiments.")

        os.makedirs(base_path, exist_ok=True)
    return base_path


def select_experiment_trips(trips, vessels_per_day) -> List[VesselTrip]:
    tripstates = []
    random.shuffle(trips)

    chosen_mmsi = set()

    trips_evaluated = 0
    for trip in trips:
        if len(tripstates) >= vessels_per_day:
            break
        trips_evaluated += 1

        if trip.vessel.v_id in chosen_mmsi:
            continue

        path = trip.get_path()
        if len({loc for loc, dur in path}) < len(path):
            print("Path returns to location", path)
            continue

        ts = TripState(trip)
        if ts.get_direction() == "both":
            continue

        tripstates.append(trip)
        chosen_mmsi.add(trip.vessel.v_id)

    tripstates = sorted(tripstates, key=lambda t: t.trip[0].time)
    return tripstates


def prep_data(days_of_week, trips_original) -> Tuple[int, List[VesselTrip], float]:
    link_counter = {}
    for t in trips_original:
        path_locations = [p[0] for p in t.get_path()]
        if {'Pause', 'Not_found'}.intersection(set(path_locations)):
            continue
        for p, q in zip(path_locations, path_locations[1:]):
            p, q = sorted((p, q))
            link_counter[(p, q)] = link_counter.get((p, q), 0) + 1
    network = nx.Graph()
    for k, v in sorted(link_counter.items(), key=lambda k_v: -k_v[1]):
        network.add_edge(k[0], k[1], weight=-v)
    tree = nx.minimum_spanning_tree(network, )
    trips_in_tree = []
    for t in trips_original:
        path_locations = [p[0] for p in t.get_path()]
        if {"Straatsburgdok_west", "Straatsburgdok_east", 'Asiadok', 'Suezdok', 'Lobroekdok'}.intersection(
                set(path_locations)) and len(path_locations) < 4:
            continue
        if len(set(path_locations)) != len(path_locations):
            continue
        valid = True
        for p, q in zip(path_locations, path_locations[1:]):
            p, q = sorted((p, q))
            if (p, q) not in tree.edges:
                valid = False
        if valid:
            trips_in_tree.append(t)

    print(tree)
    for e in tree.edges():
        print(e)

    ratio_removed = len(trips_original) / len(trips_in_tree)
    print(f"Cleaning links to enforce tree-structure. "
          f"{len(trips_in_tree)} remaining from {len(trips_original)} (ratio: {ratio_removed:5.2f})")

    trips_at_day = [trip for trip in trips_in_tree if trip.trip[0].time.weekday() in days_of_week]
    number_of_days = len({trip.trip[0].time.dayofyear for trip in trips_in_tree})
    for trip in trips_at_day:
        for vs in trip.trip:
            vs.time = vs.time.replace(year=2022, month=3, day=1)
    trips_single_day = sorted(trips_at_day, key=lambda t_d: t_d.trip[0].time)

    return number_of_days, trips_single_day, ratio_removed


def run_iterative_improvements_df_without_self(x, block_time, trips_exp_list, type_of_day, trips_per_day, seaship_stat):
    return sr.run_iterative_improvements_df(x, block_time, trips_exp_list, type_of_day, trips_per_day, seaship_stat)


def write_results(i, log_path, sim, sorted_schedule_final, signal_schedule):
    """ Dump results to a pickled file, allowing quick access to extract information. """
    dill.dump(sorted_schedule_final,
              open(os.path.join(log_path, f"vessel_trips_{i}.pkl"), "wb"))
    dill.dump(signal_schedule,
              open(os.path.join(log_path, f"signal_schedule_{i}.pkl"), "wb"))
    path = os.path.join(log_path, f"{i}.txt")
    with open(path, "w+") as f:
        for loc, b_is in sim.blocked_intervals.items():
            for interval in b_is:
                direction = f"_{loc[1]}" if len(loc[1]) > 1 else ""
                f.write(f"{loc[0]}{direction} {interval[0]} {interval[1]}\n")


class SimulationRunner:
    """ Controller for the simulation. Set up experiments with different scenarios through this class. """

    def __init__(self, directory=conf.save_location):
        self.one_way_canal: Location = Location("AK1", 0, "canal")
        self.one_way_harbor: Location = Location("AK5", 0, "harbor")
        self.one_way_area: Set[str] = {"AK1", "AK2", "AK3", "AK4", "Fork_narrow", "AK5"}
        self.sim_structs = SimulationDataStructures()

        if conf.experiments == "all" or "iterative_improvements":
            self.base_path = verify_log_files(directory)
        else:
            self.base_path = directory

        self.trips_original: List[VesselTrip] = sorted(get_filtered_trips(conf.select_trip_days),
                                                       key=lambda t: t.trip[0].time)
        self.sim_structs.build_fixed_structures(self.trips_original)
        self.seaship_status = conf.seaship_status

    def run_all_experiments(self):
        """ Runs several scenarios {conf.n_iterations} times and writes results to files. """

        n_iterations = conf.n_iterations

        for type_of_day in conf.type_of_day_list:
            days_of_week = {0, 1, 2, 3, 4} if type_of_day == "midweek" else {5, 6}

            number_of_days, trips_single_day, ratio_removed = prep_data(days_of_week, self.trips_original)
            for trips_per_day in conf.trips_per_day_list if type_of_day == "midweek" else [150]:
                for one_way_area_end in conf.one_way_area_end_list:
                    if one_way_area_end == "AK3":
                        self.one_way_area = {"AK1", "AK2", "AK3"}
                        self.one_way_harbor = Location("AK3", 0, "harbor")
                    elif one_way_area_end == "AK5":
                        self.one_way_harbor: Location = Location("AK5", 0, "harbor")
                        self.one_way_area: Set[str] = {"AK1", "AK2", "AK3", "AK4", "Fork_narrow", "AK5"}
                    else:
                        self.one_way_harbor = ""
                        self.one_way_area = {}

                    if type_of_day != "midweek" or one_way_area_end == "base":
                        seaship_stats = [0]
                    else:
                        seaship_stats = conf.seaship_stats
                    for seaship_stat in seaship_stats:
                        self.seaship_status = seaship_stat

                        trips_exp_list = [select_experiment_trips(trips_single_day, trips_per_day)
                                          for _ in range(n_iterations)]

                        model_list = conf.models  # Model.MODELS.keys()
                        if one_way_area_end == "base":
                            model_list = [list(model_list)[0]]
                        for model_type in model_list:
                            name = f"sim_{type_of_day}_{trips_per_day}_{one_way_area_end}_S{seaship_stat}_{model_type}"
                            print(name)
                            log_path = os.path.join(self.base_path, name)
                            os.mkdir(log_path)

                            for i in range(n_iterations):
                                # Create an environment and start the setup process
                                if model_type == "Dynamic":
                                    sim = self._run_single_experiment_iterative(trips_exp_list[i], 0,
                                                                                schedule_type="dynamic")
                                elif model_type == "RegOpt":
                                    try:
                                        p = self.get_best_params(type_of_day, trips_per_day, one_way_area_end,
                                                                 seaship_stat)
                                    except ValueError:
                                        break
                                    sim = self._run_single_experiment_iterative(trips_exp_list[i], *p,
                                                                                schedule_type="regular")
                                else:
                                    strip = model_type.strip().split("x")
                                    open_time = int(strip[0][3:]) * 60
                                    open_time_off = int(strip[1]) * 60
                                    sim = self._run_single_experiment_iterative(trips_exp_list[i], open_time,
                                                                                open_time_off,
                                                                                schedule_type="regular")

                                sorted_schedule_final = sorted(sim.final_schedule, key=lambda t: t.trip[0].time)
                                write_results(i, log_path, sim, sorted_schedule_final, sim.get_schedule_from_signals())

    def run_iterative_improvements(self):
        n_iterations = conf.n_iterations
        type_of_day = "midweek"
        one_way_area_end = "AK5"
        trips_per_day = 300

        days_of_week = {0, 1, 2, 3, 4} if type_of_day == "midweek" else {5, 6}
        self.set_one_way_settings(one_way_area_end)

        number_of_days, trips_single_day, ratio_removed = prep_data(days_of_week, self.trips_original)

        for seaship_stat in conf.seaship_stats:
            self.seaship_status = seaship_stat

            trips_exp_list = [select_experiment_trips(trips_single_day, trips_per_day)
                              for _ in range(n_iterations)]

            # model_list = conf.models  # Model.MODELS.keys()
            # if one_way_area_end == "base":
            #     model_list = [list(model_list)[0]]

            for open_time in range(15 * 60, 46 * 60, 60):
                for block_time in range(15 * 60, 46 * 60, 60):
                    name = f"sim_i_{type_of_day}_{trips_per_day}_{one_way_area_end}_S{seaship_stat}_{open_time}_{block_time}"
                    print(name)
                    log_path = os.path.join(self.base_path, name)
                    os.makedirs(log_path, exist_ok=True)

                    vtrips = {}
                    schedules = {}
                    queue_intervals = {}
                    for i in range(n_iterations):
                        # Create an environment and start the setup process
                        sim = self._run_single_experiment_iterative(trips_exp_list[i], open_time)

                        sorted_schedule_final = sorted(sim.final_schedule, key=lambda t: t.trip[0].time)
                        write_results(i, log_path, sim, sorted_schedule_final, sim.get_schedule_from_signals())

                        vtrips[i] = sorted_schedule_final
                        schedules[i] = sim.get_schedule_from_signals()
                        queue_intervals[i] = sim.blocked_intervals
                    sail_time, ql = get_results(name, queue_intervals, vtrips, schedules)
                    print(f"* {sail_time + 150 * ql:6.0f}: {sail_time:6.0f}-{ql:5.2f}")

    def set_one_way_settings(self, one_way_area_end):
        if one_way_area_end == "AK3":
            self.one_way_area = {"AK1", "AK2", "AK3"}
            self.one_way_harbor = Location("AK3", 0, "harbor")
        elif one_way_area_end == "AK5":
            self.one_way_harbor: Location = Location("AK5", 0, "harbor")
            self.one_way_area: Set[str] = {"AK1", "AK2", "AK3", "AK4", "Fork_narrow", "AK5"}

    def run_iterative_improvements_df(self, x, block_time, trips_exp_list, type_of_day, trips_per_day, seaship_stat):
        open_time = x[0] * 60
        open_time_off = x[1] * 60
        off_peak_morning = x[2] * 15
        off_peak_evening = x[3] * 15
        one_way_area_end = "AK5"

        self.set_one_way_settings(one_way_area_end)

        self.seaship_status = seaship_stat
        name = f"sim_i_{type_of_day}_{trips_per_day}_{one_way_area_end}_S{seaship_stat}_{open_time:.0f}_" \
               f"{block_time:.0f}_offp{open_time_off:.0f}_{off_peak_morning:.0f}_{off_peak_evening:.0f}"
        log_path = os.path.join(self.base_path, name)
        os.makedirs(log_path, exist_ok=True)

        vtrips = {}
        schedules = {}
        queue_intervals = {}
        for i in range(conf.n_iterations):
            # Create an environment and start the setup process
            sim = self._run_single_experiment_iterative(trips_exp_list[i], open_time, open_time_off,
                                                        off_peak_morning, off_peak_evening)
            sorted_schedule_final = sorted(sim.final_schedule, key=lambda t: t.trip[0].time)
            write_results(i, log_path, sim, sorted_schedule_final, sim.get_schedule_from_signals())

            vtrips[i] = sorted_schedule_final
            schedules[i] = sim.get_schedule_from_signals()
            queue_intervals[i] = sim.blocked_intervals
        sail_time, ql = get_results(name, queue_intervals, vtrips, schedules)
        print(f"*{name:60s} {sail_time + 150 * ql:6.0f}: {sail_time:6.0f}-{ql:5.2f}")
        return sail_time + 150 * ql

    def run_differential_evolution(self):
        # type_of_day = "midweek"
        # trips_per_day = 300
        # seaship_status = 2

        for type_of_day in ["midweek", "weekend"]:
            days_of_week = {0, 1, 2, 3, 4} if type_of_day == "midweek" else {5, 6}
            number_of_days, trips_single_day, ratio_removed = prep_data(days_of_week, self.trips_original)

            for trips_per_day in [200, 250, 300] if type_of_day == "midweek" else [150]:
                trips_exp_list = [select_experiment_trips(trips_single_day, trips_per_day) for _ in
                                  range(conf.n_iterations)]

                for seaship_status in [0, 2] if type_of_day == "midweek" else [0]:
                    tstart = time.time()

                    # First two every 1m, last two every 15 minutes
                    bounds = [(5, 45), (0, 30), (2 * 4, 8 * 4), (16 * 4, 24 * 4)]
                    result = differential_evolution(run_iterative_improvements_df_without_self,
                                                    args=(30 * 60, trips_exp_list, type_of_day, trips_per_day,
                                                          seaship_status),
                                                    bounds=bounds,
                                                    x0=(30, 15, 24, 84),
                                                    strategy='rand1bin',
                                                    seed=1,
                                                    popsize=20,
                                                    recombination=0.6938,
                                                    mutation=0.9314,
                                                    maxiter=100,
                                                    updating='immediate',
                                                    workers=5,
                                                    init='halton',
                                                    disp=True,
                                                    integrality=[True] * 4)

                    with open(os.path.join(OUTPUT_PATH, "schedule_params_best.txt"), 'a+') as f:
                        res = ' '.join([str(int(rx)) for rx in result.x])
                        scenario = f"sim_i_{type_of_day}_{trips_per_day}_AK5_S{seaship_status}"
                        f.write(
                            f"{scenario} {res} {time.time() - tstart:.0f} {result.fun:.0f} {result.nfev} {result.nit} {result.success}\n")

                    print(f"{res} - V{result.fun:.0f} F{result.nfev} I{result.nit} {result.message}")

    def _run_single_experiment_iterative(self, trips_experiment, open_time, open_time_off=15 * 60,
                                         off_peak_morning=6 * 60, off_peak_evening=21 * 60, schedule_type='regular'):
        env = simpy.Environment()
        self.sim_structs.model = Model(env, self.one_way_canal,
                                       self.one_way_harbor, self.one_way_area, self.seaship_status)
        self.sim_structs.model.block_time = 30 * 60 if "AK5" in self.one_way_area else 15 * 60
        self.sim_structs.model.open_time = open_time
        self.sim_structs.model.open_time_off = open_time_off
        self.sim_structs.model.off_peak_morning = off_peak_morning
        self.sim_structs.model.off_peak_evening = off_peak_evening
        self.sim_structs.model.schedule_type = schedule_type
        sim = Simulation(env, self.one_way_area, self.sim_structs, False)
        env.process(sim.setup(trips_experiment))
        # Execute!
        env.run(until=1234567890123)
        return sim

    def get_best_params(self, type_of_day: str, ntrips: int, one_way: str, seaship_status: int) -> Tuple[
        int, int, int, int]:
        with open(os.path.join(OUTPUT_PATH, "schedule_params_best.txt"), 'r') as f:
            f.readline()
            for line in f:
                words = line.strip().split(" ")
                scenario = words[0].split("_")
                if scenario[2] == type_of_day and int(scenario[3]) == ntrips and scenario[4] == one_way \
                        and int(scenario[5][1]) == seaship_status:
                    return int(words[1]) * 60, int(words[2]) * 60, int(words[3]) * 15, int(words[4]) * 15
            raise ValueError("Scenario not optimized")

    # def _run_single_experiment(self, model_type, trips_experiment, enable_visual=False):
    #     env = simpy.Environment()
    #     self.sim_structs.model = Model(env, model_type, self.one_way_canal,
    #                                    self.one_way_harbor, self.one_way_area, self.seaship_status)
    #     self.sim_structs.model.block_time = 30 * 60 if "AK5" in self.one_way_area else 15 * 60
    #
    #     sim = Simulation(env, self.one_way_area, self.sim_structs, enable_visual)
    #     env.process(sim.setup(trips_experiment))
    #     # Execute!
    #     env.run(until=1234567890123)
    #     return sim

    def run_sim_single(self) -> Tuple[List[VesselTrip], str]:
        """ Run a single scenario once and return the resulting schedule. """

        day = conf.type_of_day
        days_of_week = {0, 1, 2, 3, 4} if day == "midweek" else {5, 6}
        number_of_days, trips_single_day, ratio_removed = prep_data(days_of_week, self.trips_original)

        trips_per_day = conf.ntrips if day == "midweek" else 150
        one_way_area_end = conf.one_way_area_end
        if one_way_area_end == "AK3":
            self.one_way_area = {"AK1", "AK2", "AK3"}
            self.one_way_harbor = Location("AK3", 0, "harbor")
        elif one_way_area_end == "AK5":
            self.one_way_area: Set[str] = {"AK1", "AK2", "AK3", "AK4", "Fork_narrow", "AK5"}
            self.one_way_harbor: Location = Location("AK5", 0, "harbor")
        else:
            self.one_way_harbor = ""
            self.one_way_area = {}
            self.seaship_status = 3

        trips_exp_list = select_experiment_trips(trips_single_day, trips_per_day)

        model_type = conf.model_type

        if conf.experiments == "single":
            enable_visuals = False
        elif conf.experiments == "visual":
            enable_visuals = True
        else:
            raise ValueError()
        sim = self._run_single_experiment_iterative(trips_exp_list, enable_visuals)
        sorted_schedule_final = sorted(sim.final_schedule, key=lambda t: t.trip[0].time)

        name = f"sim_{day}_{trips_per_day}_{conf.one_way_area_end}_S{self.seaship_status}_{model_type}"
        log_path = os.path.join(self.base_path, "single", name)

        write_results("single", log_path, sim, sorted_schedule_final, sim.get_schedule_from_signals())

        return sorted_schedule_final, name


if __name__ == "__main__":
    start = time.time()
    sr = SimulationRunner()
    sr.run_differential_evolution()
    print(time.time() - start)
