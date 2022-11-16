import heapq
import math
from typing import List, Tuple, Dict, Collection

import intervaltree as it

from sample.simulation.tripstate import TripState, rectangle_paths, rectangle_endpoints, Location

Interval = Tuple[float, float]


def time_to_queue_length(intervals: List[Interval]) -> Dict[int, int]:
    intervals = sorted(intervals)
    start = math.floor(intervals[0][0])
    end = math.ceil(intervals[-1][1])
    res = {}
    interval_ends_heap = []
    for timestep in range(start, end):
        while intervals and intervals[0][0] <= timestep:
            current_i = intervals.pop(0)
            heapq.heappush(interval_ends_heap, current_i[1])
        while interval_ends_heap and interval_ends_heap[0] < timestep:
            heapq.heappop(interval_ends_heap)

        res[timestep] = len(interval_ends_heap)

    return res


# def queue_peaks(time_to_queue_length: Dict[int,int]):
#     timings = sorted(time_to_queue_length.keys())
#     qls = [time_to_queue_length[t] for t in timings]
#     qls_neg = [-time_to_queue_length[t] for t in timings]
#     peak_indices, props = find_peaks(qls, prominence=1)
#     peak_indices_neg, props2 = find_peaks(qls_neg, prominence=1, width=(None, None))
#     return [(timings[i], qls[i]) for i in sorted(list(peak_indices))], \
#            [(timings[i]-int(w/2), qls[i]) for i, w in zip(peak_indices_neg, props2['width_heights'])]


def evaluate_overlap(sorted_schedule_final, write_path):
    sailing_time = {}
    overlap_total = 0
    all_intervals: Dict[Location, List[Tuple[float, float]]] = {}
    for vt in sorted_schedule_final:
        a_vt = TripState(vt)
        time_prev = vt.trip[0].time.timestamp()
        for _, dur in vt.get_path():
            time_c = time_prev + dur
            loc_directed = a_vt.get_loc_current()
            if loc_directed not in all_intervals:
                all_intervals[loc_directed] = []
                sailing_time[loc_directed] = 0
            if time_c - time_prev > 1:
                all_intervals[loc_directed].append((time_prev, time_c))
                sailing_time[loc_directed] += dur
            a_vt.go_to_next_loc(0)
            time_prev = time_c

    all_intervals.pop(Location("", 0), None)

    # total_interval_time_dict = {k: sum(total_interval_time([i]) for i in v) for k, v in all_intervals.items()}

    # with open(write_path, "w+") as f:
    #     for k, v in total_interval_time_dict.items():
    #         f.write(f"{k} {v}\n")

    for k, v in all_intervals.items():
        all_intervals[k] = merge_intervals(v)
    for r_l in rectangle_paths:
        for r in r_l:
            if r in rectangle_endpoints or r == "Intersection":
                continue
            if Location(r, 0, "harbor") in all_intervals and Location(r, 0, "canal") in all_intervals:
                overlap = find_overlap(all_intervals[Location(r, 0, "canal")], all_intervals[Location(r, 0, "harbor")])
                overlap_time = total_interval_time(overlap)
                overlap_total += overlap_time
    print(overlap_total, sum(v for v in sailing_time.values()), write_path.split("/")[-1])
    return overlap_total, sum(v for v in sailing_time.values())


def merge_intervals(interval_list: List[Interval]) -> List[Interval]:
    """ Merge intervals to get total better view on total load.

    See https://stackoverflow.com/questions/43600878/merging-overlapping-intervals for an implementation.
    """
    if not interval_list:
        return interval_list
    i_l = sorted(interval_list, key=lambda t: t[0])
    merged_intervals: List[Interval] = [i_l[0]]
    for tup in i_l:
        tup_prev = merged_intervals[-1]
        if tup_prev[1] < tup[0]:
            merged_intervals.append(tup)
        else:
            merged_intervals[-1] = (tup_prev[0], max(tup[1], tup_prev[1]))
    return merged_intervals


def total_interval_time(intervals: List[Interval]) -> float:
    """ Given a list of intervals, calculate the sum. """
    return int(sum(t[1] - t[0] for t in intervals))


def find_overlap(merged_intervals: List[Interval], merged_intervals2: List[Interval]) -> List[Interval]:
    """ Calculate total overlap between two interval lists. Both lists should be non-overlapping by themselves.

    Since this is non-trivial to do correctly and efficiently, we rely on the pypi open-source library intervaltree.
    """

    t = it.IntervalTree.from_tuples(merged_intervals)
    t2 = it.IntervalTree.from_tuples(merged_intervals2)

    overlap_intervals: List[Interval] = []
    for interval in t:
        # Enumerate all overlapping intervals
        for interval_2 in t2.overlap(interval):
            overlap_intervals.append((max(interval.begin, interval_2.begin), min(interval.end, interval_2.end)))

    # print("Number of overlaps:", len(overlap_intervals))
    return merge_intervals(overlap_intervals)


def find_intervals_intersecting_at_point(intervals: Collection[Interval], point) -> List[Interval]:
    return [(a, b) for i, (a, b) in enumerate(intervals) if a <= point <= b]
