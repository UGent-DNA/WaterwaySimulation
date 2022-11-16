from dataclasses import dataclass
from typing import Dict, Tuple, List

import pandas as pd

from sample.data_processing.data_classes import VesselTrip


def lcs(list1, list2):
    """ Find the longest common subsequence between two lists.

    Example: l1: 1, 2, 3, 4, 5; l2: 1, 3, 4, 2, 6. Then the lcs is 1, 3, 4.
    Source: From https://www.codespeedy.com/find-longest-common-subsequence-in-python/
    """

    a = len(list1)
    b = len(list2)

    # Find length lcs
    string_matrix = [[0 for _ in range(b + 1)] for _ in range(a + 1)]
    for i in range(1, a + 1):
        for j in range(1, b + 1):
            if i == 0 or j == 0:
                string_matrix[i][j] = 0
            elif list1[i - 1] == list2[j - 1]:
                string_matrix[i][j] = 1 + string_matrix[i - 1][j - 1]
            else:
                string_matrix[i][j] = max(string_matrix[i - 1][j], string_matrix[i][j - 1])
    index = string_matrix[a][b]

    # Reconstruct lcs
    res = [""] * index
    i = a
    j = b
    while i > 0 and j > 0:
        if list1[i - 1] == list2[j - 1]:
            res[index - 1] = list1[i - 1]
            i -= 1
            j -= 1
            index -= 1
        elif string_matrix[i - 1][j] > string_matrix[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return res


def similarity_paths(vt1: VesselTrip, vt2: VesselTrip) -> Tuple[int, int, int, int]:
    """ Calculate the similarity between two paths and return where the similarity starts, based on lcs.

    :return: Tuple with:
    * The duration of the similar path in the first vesseltrip (i.e., how long does the convoy exist?)
    * The difference in duration between the two paths (i.e., if vt1 takes 70s and vt2 takes 80s, this is 10 seconds)
    * The delay between the paths at their start (i.e., how many seconds is one ship behind the other?)
    * The moment in the first path where the similarity starts
        (i.e., how many seconds after the ship enters the simulation, does it enter a convoy)
    """
    p1_ = vt1.get_path()
    p1 = [(k, v) for k, v, _ in p1_] if len(p1_[0]) == 3 else p1_
    p2_ = vt2.get_path()
    p2 = [(k, v) for k, v, _ in p2_] if len(p2_[0]) == 3 else p2_

    common_path = lcs([a[0] for a in p1], [a[0] for a in p2])
    if not common_path:
        return 0, 0, 0, 0
    index1 = 0
    index2 = 0
    start_offset1 = 0
    start_offset2 = 0
    for name, duration in p1:
        if name == common_path[0]:
            break
        index1 += 1
        start_offset1 += duration

    for name, duration in p2:
        if name == common_path[0]:
            break
        index2 += 1
        start_offset2 += duration

    total_duration = 0
    duration_diff = 0
    while index1 < len(p1) and index2 < len(p2):
        name1, duration1 = p1[index1]
        name2, duration2 = p2[index2]
        if name1 != name2:
            break

        if abs(duration2 - duration1) > 60:
            break

        total_duration += duration1
        duration_diff += duration2 - duration1

        index1 += 1
        index2 += 1

    start_difference = (vt2.trip[0].time - vt1.trip[0].time).seconds + start_offset2 - start_offset1

    return total_duration, duration_diff, start_difference, start_offset1


@dataclass
class Convoy:
    leader: VesselTrip
    follower: VesselTrip
    start_offset_leader: int
    start_difference: int
    duration: int

    def get_time_to_mmsi(self, precision) -> Dict[pd.Timestamp, Tuple[int, int]]:
        """ For each timestamp that this convoy exists, create a dict-entry. """
        res = {}
        time_start = self.leader.trip[0].time.timestamp() + self.start_offset_leader + self.start_difference
        for vs in self.leader.trip:
            vs_time = vs.time.round(f"{precision}S")
            if time_start <= vs_time.timestamp() <= time_start + self.duration:
                res[vs_time] = (self.leader.vessel.v_id, self.follower.vessel.v_id)
        return res

    def get_convoy_start(self) -> float:
        return self.leader.trip[0].time.timestamp() + self.start_offset_leader


def find_convoy(vessel_trips: List[VesselTrip]) -> List[Convoy]:
    """ Find all the convoys in the given vesseltrip list."""
    vessel_trips_sorted = sorted(vessel_trips, key=lambda v: v.trip[0].time)
    convoys: List[Convoy] = []
    for i, vt in enumerate(vessel_trips_sorted):
        for vt2 in vessel_trips_sorted[i + 1:]:
            if (vt2.trip[0].time - vt.trip[0].time).seconds > 3600:
                # If the there is at least one hour between the start times: SKIP
                break
            total_duration, duration_diff, start_difference, leader_offset = similarity_paths(vt, vt2)
            if 0 < start_difference <= 300 and total_duration > 200 and abs(duration_diff / total_duration) < 0.3:
                convoys.append(Convoy(vt, vt2, leader_offset, start_difference, total_duration))
    return convoys


def trips_in_convoy(convoy_list: List[Convoy]) -> List[VesselTrip]:
    """ Given a set of convoys, find all vesseltrips that make them. """
    res = set()
    for i, c in enumerate(convoy_list):
        res.add(c.leader)
        res.add(c.follower)

    return sorted(res, key=lambda vt: vt.trip[0].time)
