import copy
import heapq
import math
import random
from typing import List, Dict, Tuple, Set

import pandas as pd
import simpy

from sample.data_processing.data_classes import VesselTrip, VesselState, Vessel
from sample.general.interval_manipulation import evaluate_overlap, Interval
from sample.general.util_general import average, std
from sample.simulation.model import Model
from sample.simulation.prep_visualisation import create_vis_handbook, find_closest_match
from sample.simulation.tripstate import rectangle_paths, rectangle_endpoints, TripState, Location

loc_d = Tuple[str, str]
random.seed(4242)


class SimulationDataStructures:
    # Variable structures (change during the simulation)
    model: Model

    # Fixed structures: Generate once, use for all simulations.
    vis_handbook: Dict[Tuple[str, str, str], List[Tuple[float, List[VesselState]]]] = {}
    loc_triple_distance_dict: Dict[Tuple[str, str, str], int] = {}

    def build_fixed_structures(self, trips: List[VesselTrip]):
        self.vis_handbook, self.loc_triple_distance_dict = create_vis_handbook(trips)


class OneWayResource(simpy.Resource):
    """ Resource extension to enforce single direction. Makes it impossible for 2 ships to cross in the simulation."""

    def __init__(self, env: simpy.Environment, capacity):
        super().__init__(env, capacity)
        self.must_be_free_resources: List[simpy.Resource] = []
        self.env = env

    def set_must_be_free_resources(self, resources: List[simpy.Resource]):
        self.must_be_free_resources = resources

    def _do_put(self, event) -> None:
        way_is_open = True
        for res in self.must_be_free_resources:
            if len(res.users) > 0:
                way_is_open = False
        if way_is_open and len(self.users) < self.capacity:
            self.users.append(event)
            event.usage_since = self._env.now
            event.succeed()


def print_distances(one_way_area, sim_structs):
    """ Print the distances between the location rectangles in the one way area. """
    locs = ["Intersection", "AK1", "AK2", "AK3", "AK4", "Fork_narrow", "AK5", "Albertkanaal"]
    dist_sum = 0
    for i in range(len(one_way_area), 0, -1):
        dist = sim_structs.loc_triple_distance_dict[locs[i - 1], locs[i], locs[i + 1]]
        print("Dist: ", locs[i - 1], locs[i], locs[i + 1], dist)
        dist_sum += dist
    print("Sum dist = ", dist_sum)


class Simulation(object):
    def __init__(self, env, one_way_area: Set[str], sim_structs: SimulationDataStructures,
                 enable_visual_trajectories: bool = False):
        self.env = env
        self.locations: Dict[loc_d, simpy.Resource] = {}

        self.total_time_in_simulation: int = 0
        self.one_way_area: Set[str] = one_way_area
        self.enable_visual_trajectories = enable_visual_trajectories
        self.sim_structs: SimulationDataStructures = sim_structs

        self.signal_dict: Dict[Interval, str] = {}
        self.last_signal = None

        print_distances_bool = False
        if print_distances_bool:
            print_distances(one_way_area, sim_structs)
        self.active_trips: List[TripState] = []
        self.blocked_intervals: Dict[loc_d, List[Tuple[int, int]]] = {}
        self.final_schedule: List[VesselTrip] = []

        self.latest_ship_movement: int = 0
        self.location_q: Dict[Tuple[str, str], List[Tuple[int, float, TripState]]] = {(owa, d): [] for owa in
                                                                                      one_way_area for d in
                                                                                      ["canal", "harbor"]}
        capacity = 100

        handled_locs = set()
        self.locations[("Intersection", "")] = OneWayResource(env, capacity)
        handled_locs.add("Intersection")

        for loc in rectangle_endpoints:
            if loc in handled_locs:
                continue
            self.locations[(loc, "")] = simpy.Resource(env, capacity)
            handled_locs.add(loc)

        for loc_list in rectangle_paths:
            for loc in loc_list:
                if loc in handled_locs:
                    continue
                for direction in "harbor", "canal":
                    if loc in self.one_way_area:
                        res = simpy.Resource(env, capacity)
                        self.locations[(loc, direction)] = res
                    else:
                        self.locations[(loc, direction)] = simpy.Resource(env, capacity)

        self.locations[("Not_found", "")] = simpy.Resource(env, 100)
        self.locations[("Pause", "")] = simpy.Resource(env, 100)

    def setup(self, trips: List[VesselTrip]):
        prev_start_time = 0
        start_of_day = pd.to_datetime("03/01/2022")
        for trip in trips:
            start_time = seconds_from_start_of_day(start_of_day, trip.trip[0].time)

            yield self.env.timeout(start_time - prev_start_time)
            self.env.process(VesselSimulation(self, trip).sail())
            prev_start_time = start_time
        return self

    def sail_through(self, loc_time):
        yield self.env.timeout(loc_time)

    def get_schedule_from_signals(self) -> Dict[Interval, str]:
        """ Retrace schedule based on signals from queuing ships. Necessary for the dynamic model. """
        schedule: Dict[Interval, str] = {}
        csignal = ((0, 0), "canal")
        sorted_iv = sorted(self.signal_dict.keys())
        for i, iv in enumerate(sorted_iv):
            signal = self.signal_dict[iv]
            if signal == csignal[1]:
                csignal = ((csignal[0][0], max(csignal[0][1], iv[1])), signal)
            else:
                schedule[(csignal[0][0], min(csignal[0][1], iv[0]))] = csignal[1]
                csignal = (iv, signal)
        schedule[csignal[0]] = csignal[1]
        return schedule


class VesselSimulation:
    def __init__(self, sim, vesseltrip: VesselTrip):
        self.sim: Simulation = sim
        self.tripstate: TripState = TripState(vesseltrip, self.sim.env.now)
        self.vessel: Vessel = vesseltrip.vessel

        self.final_states: List[VesselState] = []
        self.final_path = []

        self.start = self.sim.env.now
        self.start_datetime = pd.Timestamp(self.start, unit='s')

    def sail(self):
        self.sim.active_trips.append(self.tripstate)

        for index, location in enumerate(self.tripstate.path_directed):
            if location is None:
                break

            total_time_blocked = 0

            if location.loc in self.sim.one_way_area and self.tripstate.get_prev_loc().loc not in self.sim.one_way_area:
                total_time_blocked += yield from self.queue_at_location(location)

            with self.sim.locations[location.as_tuple()].request() as req:
                yield req

                sail_time_new = self.calculate_sail_time(location)
                self.record_trajectory_location(location, sail_time_new, total_time_blocked)

                yield self.sim.env.process(self.sim.sail_through(sail_time_new))

            yield from self.sail_to_next_location(index)

        self.sim.active_trips.remove(self.tripstate)
        self.sim.total_time_in_simulation += (self.sim.env.now - self.start)

        self.record_trajectory_full()

    def sail_to_next_location(self, index):
        if index == len(self.tripstate.path_directed) - 1:
            self.tripstate.go_to_next_loc(self.sim.env.now)
        yield from update_queues(self.sim)
        if len(self.sim.active_trips) > 500:
            for at in self.sim.active_trips:
                at.print()
            raise RuntimeError("Too many active trips at once: check for an infinite loop.")
        if index < len(self.tripstate.path_directed) - 1:
            self.tripstate.go_to_next_loc(self.sim.env.now)

    def calculate_sail_time(self, location):
        own_duration = location.sail_time
        # Restrict speed in one-way-area
        if location.loc in self.sim.one_way_area:
            dist = self.sim.sim_structs.loc_triple_distance_dict[self.tripstate.get_loc_triple()]
            duration_at_speed = int(dist / self.tripstate.max_speed_in_m_per_s)

            # Take sail_duration and multiply with random factor up to 10% up. DOES NOT USE LOCAL TIME
            own_duration = int(duration_at_speed * (1 + (random.random()) / 10))
        # Make sure that 2 ships do not pass each other and maintain minimal separation
        other_trips_in_same_dir = [at for at in get_trips_at_loc(self.sim.active_trips, location) if
                                   at.vt.vessel.v_id != self.vessel.v_id and self.tripstate.same_direction(
                                       at)]
        other_ship_durations_at_loc = [at.get_remaining_time_at_loc(self.sim.env.now) for at in
                                       other_trips_in_same_dir]
        other_vessel_durations = max(other_ship_durations_at_loc, default=-self.tripstate.separation_time)
        sail_time_new = max(other_vessel_durations + self.tripstate.separation_time, own_duration)
        self.tripstate.replace_sail_time(location, sail_time_new)
        return sail_time_new

    def queue_at_location(self, location: Location) -> int:
        """ Make the vessel queue until it is released. It waits for a signal and until previous ships are released.

        :param location: The location where the ship just arrived
        :return: The time the ship had to wait "in this function" to be released.
        """
        start_at_loc = self.sim.env.now
        self.tripstate.go_to_prev_loc()
        direction = location.direction
        time_blocked = 0

        heapq.heappush(self.sim.location_q[location.as_tuple()], (start_at_loc, random.random(), self.tripstate))

        block_time = 1
        while block_time > 0:
            block_time = 0

            ls = self.sim.last_signal
            if ls is None:
                signal = self.sim.sim_structs.model.get_signal(self.sim.active_trips)
                self.sim.last_signal = (signal[0], self.sim.env.now, signal[1])
            else:
                if ls[1] + ls[2] > self.sim.env.now:
                    signal = (ls[0], (ls[1] + ls[2]) - self.sim.env.now)
                else:
                    signal = self.sim.sim_structs.model.get_signal(self.sim.active_trips)
                    self.sim.last_signal = (signal[0], self.sim.env.now, signal[1])

            self.sim.signal_dict[(self.sim.env.now, signal[1] + self.sim.env.now)] = signal[0]

            if signal[0] == direction or signal[0] == "free":
                diff_last_ship = self.sim.env.now - self.sim.latest_ship_movement
                if diff_last_ship < self.tripstate.separation_time:
                    block_time = self.tripstate.separation_time - diff_last_ship
                elif self.tripstate != self.sim.location_q[location.as_tuple()][0][2]:
                    block_time = self.tripstate.separation_time
            else:
                block_time = signal[1]

            if block_time != 0:
                time_blocked += block_time
                yield self.sim.env.timeout(block_time)

        self.sim.latest_ship_movement = self.sim.env.now
        heapq.heappop(self.sim.location_q[location.as_tuple()])

        self.tripstate.go_to_next_loc(self.sim.env.now)

        if self.sim.env.now != start_at_loc:
            if location.as_tuple() not in self.sim.blocked_intervals:
                self.sim.blocked_intervals[location.as_tuple()] = []
            self.sim.blocked_intervals[location.as_tuple()].append((start_at_loc, self.sim.env.now))

        return time_blocked

    def record_trajectory_full(self):
        if self.final_states:

            # smooth trajectories
            smoothed = []
            for i, vs in enumerate(self.final_states):
                indices = [-2, -1, 0, 1, 2]
                s_lat = self.smoothed_point(i, indices, 'lat')
                s_lon = self.smoothed_point(i, indices, 'lon')
                s_course = self.smoothed_point(i, indices, 'course')
                smoothed.append(VesselState(vs.v_id, vs.time, s_lat, s_lon, vs.speed, s_course))

            trip = VesselTrip(self.vessel, smoothed)
            self.sim.final_schedule.append(trip)

        if not self.sim.enable_visual_trajectories and self.final_path:
            # Add path and first_iter and last vessel state to indicate time correctly.
            vs_start = VesselState(self.vessel.v_id, self.start_datetime)  # .replace(year=2022, month=3, day=1))
            vs_end = VesselState(self.vessel.v_id, pd.Timestamp(self.sim.env.now, unit='s'))  # .replace(year=2022, month=3, day=1))
            trip = VesselTrip(self.vessel, [vs_start, vs_end])
            trip.path = self.final_path

            self.sim.final_schedule.append(trip)

    def smoothed_point(self, i, indices, element):
        return average([self.final_states[i + index].__getattribute__(element) for index in indices if
                        0 <= i + index < len(self.final_states)])

    def record_trajectory_location(self, location, sail_time_new, total_time_blocked):
        # Record result
        if not self.sim.enable_visual_trajectories:
            self.final_path.append((location.loc, sail_time_new, total_time_blocked))
        else:
            vs_list = find_closest_match(self.tripstate.get_loc_triple(), sail_time_new,
                                         self.sim.sim_structs.vis_handbook,
                                         pd.Timestamp(self.sim.env.now, unit='s'), self.vessel.v_id)
            if vs_list:
                # Add ships states also when they stand still
                if self.final_states:
                    last_known = self.final_states[-1]
                else:
                    last_known = copy.deepcopy(vs_list[0])
                    last_known.time = self.start_datetime

                diff = (vs_list[0].time - last_known.time).total_seconds()
                if diff > 6:
                    for secs in range(0, math.floor(diff), 3):
                        self.final_states.append(
                            VesselState(self.vessel.v_id, last_known.time + pd.DateOffset(seconds=secs),
                                        last_known.lat, last_known.lon))
                for vs in vs_list:
                    self.final_states.append(vs)


def get_trips_at_loc(active_trips: List[TripState], loc: Location) -> List[TripState]:
    return [at for at in active_trips if at.get_loc_current() == loc]


def update_queues(sim):
    for _, rec in sim.locations.items():
        if len(rec.put_queue) >= 1:
            with rec.request() as req:
                yield req


def seconds_from_start_of_day(start_of_day: pd.Timestamp, current_time: pd.Timestamp):
    return int((current_time - start_of_day).total_seconds())


def get_original_stats(days_of_week, trips_original, name):
    overlap_sail = []
    trips_by_day: Dict[int, List[VesselTrip]] = {}
    for trips in trips_original:
        if trips.trip[0].time.day_of_week in days_of_week:
            d = trips.trip[0].time.dayofyear
            if d not in trips_by_day:
                trips_by_day[d] = []
            trips_by_day[d].append(trips)
    for d, t_list in trips_by_day.items():
        overlap_sail.append(evaluate_overlap(t_list, f"original_{name}_{d}_{t_list[0].trip[0].time.day_of_week}"))
    print(f"{average([o_s[0] for o_s in overlap_sail]):6.0f} [{std([o_s[0] for o_s in overlap_sail]):7.0f}] -"
          f"{average([o_s[1] for o_s in overlap_sail]):8.0f} [{std([o_s[1] for o_s in overlap_sail]):7.0f}]")
