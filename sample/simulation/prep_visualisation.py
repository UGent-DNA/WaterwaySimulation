import math
from typing import List, Dict, Tuple

import pandas as pd

from sample.data_processing.data_classes import VesselTrip, VesselState
from sample.general.util_general import calculate_distance, average
from sample.simulation.tripstate import TripState


# VISUALIZATION
def create_vis_handbook(trips: List[VesselTrip]) -> Tuple[
    Dict[Tuple[str, str, str], List[Tuple[float, List[VesselState]]]], Dict[Tuple[str, str, str], int]]:
    # Dict from location to a list of: a pair of duration + list of coordinates (in VesselState format)
    vis_handbook: Dict[Tuple[str, str, str], List[Tuple[float, List[VesselState]]]] = {}

    for t in trips:
        at = TripState(t)
        time_start = t.trip[0].time
        trip = t.trip
        trip_index = 0
        for _ in t.get_path():
            loc_triple = at.get_loc_triple()

            if loc_triple not in vis_handbook:
                vis_handbook[loc_triple] = []
            time_end = time_start + pd.DateOffset(seconds=at.get_loc_current().sail_time)
            # time_end = pd.to_datetime(int(time_start.timestamp() + dur), unit='s')

            trip_state_list = []

            trip_state = trip[trip_index]
            while trip_state.time < time_end:
                trip_state_list.append(trip_state)
                trip_state = trip[trip_index]
                trip_index += 1
            time_start = time_end

            if trip_state_list:
                dur_effective = (trip_state_list[-1].time - trip_state_list[0].time).total_seconds()
                vis_handbook[loc_triple].append((dur_effective, trip_state_list))

            at.go_to_next_loc(0)

    loc_triple_distance_dict = {}
    for loc_triple, my_list in vis_handbook.items():
        #     # print(loc_triple)
        #     all_d = []
        #     for dur, trip_state_list in my_list:
        #         d1 = 0
        #         for i in range(len(trip_state_list)-1):
        #             d1 += calculate_distance(trip_state_list[i], trip_state_list[i+1])
        #
        #
        #         smoothed_intervals = [trip_state_list[0]]
        #         for i in range(len(trip_state_list)):
        #             coords = []
        #             for j in range(-2, 3, 1):
        #                 if 0 <= i+j < len(trip_state_list):
        #                     coords.append(trip_state_list[i+j])
        #             lats = average([c.lat for c in coords])
        #             lons = average([c.lon for c in coords])
        #             smoothed_intervals.append(VesselState(0, trip_state_list[i].time, lat=lats, lon=lons))
        #         smoothed_intervals.append(trip_state_list[-1])
        #         d2 = 0
        #         for i in range(len(smoothed_intervals)-1):
        #             d2 += calculate_distance(smoothed_intervals[i], smoothed_intervals[i+1])
        #         d = calculate_distance(trip_state_list[0], trip_state_list[-1])
        #         all_d.append(d2)
        # print(f"loc-to-loc: {d1:.0f}, smoothed:{d2:.0f}, direct:{d:.0f}")
        # loc_triple_distance_dict[loc_triple] = average(all_d)
        if my_list:
            loc_triple_distance_dict[loc_triple] = average([calculate_distance(tl[0], tl[-1]) for _, tl in my_list])

    print(loc_triple_distance_dict["Intersection", "AK1", "AK2"])
    print(loc_triple_distance_dict["AK1", "AK2", "AK3"])
    print(loc_triple_distance_dict["AK2", "AK3", "AK4"])

    # print(vis_handbook)
    return vis_handbook, loc_triple_distance_dict


def find_closest_match(loc_triple, dur, vis_handbook, start_time, mmsi) -> List[VesselState]:
    dur_diff = float('inf')
    vis: List[VesselState] = []
    if loc_triple not in vis_handbook:
        return []
    for vs_dur, vs_list in vis_handbook[loc_triple]:
        new_diff = abs(dur - vs_dur)
        if dur_diff > new_diff:
            dur_diff = new_diff
            vis = vs_list
    # print(dur, int(dur_diff), loc_directed)
    if len(vis) > 1:
        total_dur = 0
        res = []
        time_offset = vis[0].time - start_time
        prev_time = vis[0].time - time_offset
        for i, vs in enumerate(vis):
            vs_c = vs.make_copy()
            vs_c.time = vs_c.time - time_offset
            vs_c.v_id = mmsi
            res.append(vs_c)
            # Make sure total duration is not longer than simulated version
            if i > 0:
                total_dur += (vs_c.time - prev_time).total_seconds()
                prev_time = vs_c.time
            if total_dur > dur:
                return res[:-1]
        # print(start_time, " ".join([f"{vs.time.hour:2d}:{vs.time.minute:2d}" for vs in res]))

        # Make sure total duration is as close in duration as possible.
        start_time = res[0].time
        seconds_short = dur - (res[-1].time - start_time).total_seconds()
        for _ in range(math.floor(seconds_short / 3)):
            res_copy = res[-1].make_copy()
            res_copy.time += pd.DateOffset(seconds=3)
            res.append(res_copy)

        if (res[-1].time - res[0].time).total_seconds() > dur:
            return []
        return res
    else:
        return []
