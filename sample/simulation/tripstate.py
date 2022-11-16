import itertools
import math
from dataclasses import dataclass
from typing import Tuple, List

from sample.data_processing.data_classes import VesselTrip

rectangle_path = ["Amerikadok", "Albertdok", "Intersection", "AK1", "AK2",
                  "AK3", "AK4", "Fork_narrow", "AK5", "Albertkanaal"]
rectangle_path2 = ["Royerssluis_S", "Royerssluis_i", "Ac", "Intersection"]

rectangle_paths = [rectangle_path, rectangle_path2]
rectangle_endpoints = {"Straatsburgdok_east", "Straatsburgdok_west", "Asiadok", "Suezdok", "Royersluis_S",
                       "Ac", "Lobroekdok", "Albertdok"}


@dataclass
class Location:
    loc: str
    sail_time: int

    direction: str = ""
    queue_time: int = 0

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Location):
            return self.loc == other.loc and self.direction == other.direction
        return False

    def as_tuple(self):
        return self.loc, self.direction


class TripState:
    """
    Wrapper class for {VesselTrip} that maintains the current location, time spent at this location and aids in the
    detection of direction.
    """
    id_iter = itertools.count()

    def __init__(self, vt: VesselTrip, env_now=0):
        self.vt: VesselTrip = vt
        self._path_index: int = 0

        self.path_directed: List[Location] = []
        self.direction = "neither"
        self.create_directed_path()

        vl = self.vt.vessel.length_max
        length = vl if not (math.isnan(float(vl)) or vl == -1) else 100
        if length >= 135:
            self.max_speed_in_m_per_s = 6 / 3.6
        else:
            self.max_speed_in_m_per_s = 7.5 / 3.6
        # Separation equals ships length * 1.5, plus the ships length
        # because we measure from the stern (back) of the ship
        self.separation_time = int(length * 2.5 / self.max_speed_in_m_per_s)

        self.start_at_loc = env_now
        self.id = next(TripState.id_iter)

    def get_path_index(self):
        return self._path_index

    def get_sail_time_at_current(self):
        if self._path_index >= len(self.path_directed):
            return -1
        return self.get_loc_current().sail_time

    def replace_sail_time(self, location, sail_time_new):
        for loc in self.path_directed:
            if loc.loc == location.loc:
                loc.sail_time = sail_time_new
                return

    def get_loc_current(self) -> Location:
        return self._get_loc_at_index(self._path_index)

    def go_to_next_loc(self, time_now):
        self._path_index += 1
        self.start_at_loc = time_now

    def go_to_prev_loc(self):
        self._path_index -= 1
        # Reduce with fixed amount
        self.start_at_loc = self.start_at_loc - 5000

    def _get_loc_at_index(self, index) -> Location:
        if index >= len(self.path_directed) or index < 0:
            return Location("", 0)
        return self.path_directed[index]

    def print(self):
        path = self.vt.get_path()
        path_s = " -> ".join(
            f'{loc}: {dur}' if i != self._path_index else f'**{loc}: {dur}**' for i, (loc, dur) in enumerate(path))

        loc_directed = self.get_loc_current()
        print(f"{self.vt.vessel.v_id} {self.id:4d} {loc_directed} {self.start_at_loc} {path_s}")

    def get_next_loc(self) -> Location:
        """ Return the next known different location, returns an empty location if none is available. """
        pi = self._path_index + 1
        while pi < len(self.path_directed):
            next_loc = self._get_loc_at_index(pi)
            if next_loc.loc != "Pause" and next_loc.loc != "Not_found" and next_loc != self.get_loc_current():
                return next_loc
            pi += 1
        return Location("", 0)

    def get_prev_loc(self) -> Location:
        """ Return the previous known different location, returns an empty location if none is available. """
        pi = self._path_index - 1
        while pi >= 0:
            prev_loc = self._get_loc_at_index(pi)
            if prev_loc.loc != "Pause" and prev_loc.loc != "Not_found" and prev_loc != self.get_loc_current():
                return prev_loc
            pi -= 1
        return Location("", 0)

    def get_remaining_time_at_loc(self, time_now):
        """ The time this vessel has to sail to reach the next location. """
        return max(0, self.get_sail_time_at_current() - (time_now - self.start_at_loc))

    def same_direction(self, other) -> bool:
        """ Return true if {other} sails in the same direction as {self} at the same location.

        Note that we could use the direction on the locations themselves, but this does not work for e.g.
        the intersection where no direction is defined.
        This is not really a problem, because this method is meant for use in the single-direction part.

         :returns: True if they have the same previous location and/or the same next location.
         """
        if not isinstance(other, TripState):
            raise RuntimeError("Other should be a Tripstate instance.")
        if other.get_loc_current() != self.get_loc_current():
            raise RuntimeError("Only compare ships at the same location.")
        if self.get_next_loc() is not None and other.get_next_loc().loc == self.get_next_loc().loc \
                or self.get_prev_loc() is not None and other.get_prev_loc().loc == self.get_prev_loc().loc:
            return True
        return False

    def get_loc_triple(self) -> Tuple[str, str, str]:
        """ Return previous, current and next location (in string-format) to have a more precious directional key. """
        return self.get_prev_loc().loc, self.get_loc_current().loc, self.get_next_loc().loc

    def create_directed_path(self):
        """ Create the directed path of locations. """
        path = self.vt.get_path()
        self.path_directed = []
        directions = set()
        for i, (loc, dur) in enumerate(path):
            direction = get_direction(loc, path[i - 1][0] if i > 0 else None,
                                      path[i + 1][0] if i < len(path) - 1 else None)
            self.path_directed.append(Location(loc, dur, direction))
            directions.add(direction)

        if "canal" in directions and "harbor" in directions:
            self.direction = "both"
        elif "canal" in directions:
            self.direction = "canal"
        elif "harbor" in directions:
            self.direction = "harbor"
        else:
            self.direction = "neither"

    def get_direction(self):
        return self.direction


def get_queueing(active_trips: List[TripState], time_now: int, loc: Location) -> List[TripState]:
    """ Returns all trips from {active_trips} that are queueing at {location} at time {time_now}. """
    return [at for at in active_trips
            if at.get_remaining_time_at_loc(time_now) == 0 and at.get_next_loc() == loc]


def get_direction(center: str, prev: str, next: str):
    """ For a triple, return the direction of {center} based on the {rectangle_paths}"""
    if center in rectangle_endpoints or center in {"Intersection", "Not_found", "Pause"}:
        return ""

    prevs = set()
    nexts = set()
    for loc_list in rectangle_paths:
        prevs.clear()
        nexts.clear()
        center_detected = False
        for loc in loc_list:
            if center == loc:
                center_detected = True
                continue
            if not center_detected:
                prevs.add(loc)
            else:
                nexts.add(loc)
        if center_detected:
            break

    # I want Amerikadok to be two-way (not an endpoint) and still be detected
    if center == "Amerikadok":
        prevs.add(None)

    if prev in prevs or next in nexts:
        return "canal"
    if next in prevs or prev in nexts:
        return "harbor"
    return ""
