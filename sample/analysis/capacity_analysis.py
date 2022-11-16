from dataclasses import dataclass
from typing import List

from sample.data_processing.data_classes import VesselTrip
from sample.general.interval_manipulation import merge_intervals, total_interval_time, find_overlap, Interval


def get_direction(path):
    """ Get the direction of a path that goes to Fork-narrow. """
    for i, (loc, dur) in enumerate(path):
        if loc == "Fork_narrow":
            if i > 0:
                prev_loc = path[i - 1][0]
                if "AK4" == prev_loc:
                    return "canal"
                elif "AK5" == prev_loc:
                    return "harbor"
            elif i < len(path) - 1:
                next_loc = path[i + 1][0]
                if "AK4" == next_loc:
                    return "harbor"
                elif "AK5" == next_loc:
                    return "canal"
    return "no"


def get_time_at_location(vt: VesselTrip, loc_name: str) -> Interval:
    """ Return the start and end timestamp (POSIX) when this trip in the indicated location. """
    cum_dur = 0
    valid = False
    fork_dur = 0
    for loc, dur in vt.get_path():
        if loc == loc_name:
            valid = True
            fork_dur = dur
            break
        cum_dur += dur

    if not valid:
        return 0, 0

    start_time = vt.trip[0].time.timestamp()
    fork_start = start_time + cum_dur
    fork_end = start_time + cum_dur + fork_dur

    return fork_start, fork_end


@dataclass
class ForkOverlap:
    vessel_first: VesselTrip
    vessel_second: VesselTrip
    overlap_duration: float
    overlap_start: float
    same_direction: bool


def get_throughput_fork(vts: List[VesselTrip]):
    """ Calculate the throughput in several areas by evaluating the time there is at least one ship present. """
    vts = [vt for vt in vts if
           7 <= vt.trip[0].time.hour and vt.trip[-1].time.hour <= 17 and vt.trip[0].time.dayofweek < 5]
    vts = sorted(vts, key=lambda t: t.trip[0].time)

    vesseltrips_to_canal = []
    vesseltrips_to_harbor = []
    for vt in vts:
        path = vt.get_path()
        if get_direction(path) == "canal":
            vesseltrips_to_canal.append(vt)
        elif get_direction(path) == "harbor":
            vesseltrips_to_harbor.append(vt)

    fork_to_canal = merge_intervals([get_time_at_location(vt, "Fork_narrow") for vt in vesseltrips_to_canal])
    fork_to_harbor = merge_intervals([get_time_at_location(vt, "Fork_narrow") for vt in vesseltrips_to_harbor])

    ak5_times = merge_intervals([get_time_at_location(vt, "AK5") for vt in vesseltrips_to_canal])
    intersection_times = merge_intervals([get_time_at_location(vt, "Intersection") for vt in vesseltrips_to_canal])

    total_time = 3600 * 10 * len({vt.trip[0].time.dayofyear for vt in vts})

    print(len(vesseltrips_to_harbor), len(vesseltrips_to_canal), len(fork_to_canal), len(fork_to_harbor))
    print(len(find_overlap(fork_to_canal, fork_to_harbor)))

    vals = [total_interval_time(fork_to_canal),
            total_interval_time(fork_to_harbor),
            total_interval_time(find_overlap(fork_to_canal, fork_to_harbor)),
            total_interval_time(merge_intervals(fork_to_canal + fork_to_harbor)),
            total_interval_time(ak5_times),
            total_interval_time(intersection_times)]
    descriptions = ["Vork - richting kanaal", "Vork - richting Haven", "Vork kruisende schepen", "Vork totaal",
                    "AK5", "Kruispunt"]
    print(vals)
    print(total_time)
    for v, d in zip(vals, descriptions):
        print(f"{d:25s} & {v:9d} & {100 * v / total_time:6.1f}\\% \\\\")
