from typing import List, Dict

from sample.general.interval_manipulation import Interval, find_intervals_intersecting_at_point
from sample.simulation.tripstate import TripState, get_queueing


# def create_regular_schedule(open_time: int, block_time: int) -> Dict[Interval, str]:
#     ctime = 0
#     schedule: Dict[Interval: str] = {}
#     signal_duration = [("canal", open_time), ("blocked", block_time),
#                        ("harbor", open_time), ("blocked", block_time)]
#     index = 0
#     while ctime < 25 * 60 * 60:
#         signal, duration = signal_duration[index]
#         schedule[(ctime, ctime + duration)] = signal
#         ctime += duration
#         index = (index + 1) % len(signal_duration)
#     return schedule


class Model:
    def __init__(self, env, one_way_canal, one_way_harbor, one_way_area, seaship_status):
        self.env = env
        self.one_way_canal = one_way_canal
        self.one_way_harbor = one_way_harbor
        self.one_way_area = one_way_area

        self.seaship_status = seaship_status
        self.seaship_schedule: Dict[Interval, str] = {}
        self.start_sea = 0
        if self.seaship_status > 0:
            self.start_sea = self.init_seaship_schedule()

        self.min_canal = 0
        self.min_harbor = 0
        self.active_trips = []
        self.qlss = "open"  # Queue length switching status

        self.schedules = {}
        self.schedule_type = "regular"
        self.queue_trigger_length = 3
        self.open_time = 15 * 60
        self.block_time = 30 * 60
        self.open_time_off = 15 * 60
        self.off_peak_morning = 6 * 60
        self.off_peak_evening = 21 * 60
        # self.random_schedule = self.create_random_schedule()

    def init_seaship_schedule(self):
        start_sea = (8 * 60) * 60 + 15 * 60  # Start first pass at 8h15; takes 30m
        self.seaship_schedule[(start_sea - 30 * 60, start_sea)] = "not_harbor"
        self.seaship_schedule[(start_sea, start_sea + 30 * 60)] = "blocked"
        self.seaship_schedule[(start_sea + 30 * 60, start_sea + 45 * 60)] = "harbor"
        mid_sea = start_sea + 75 * 60  # Second pass (in scenario 1); takes 30m, no extra buffer
        if self.seaship_status == 1:
            self.seaship_schedule[(start_sea + 45 * 60, start_sea + 105 * 60)] = "blocked"
        elif self.seaship_status == 2:
            self.seaship_schedule[(start_sea + 45 * 60, start_sea + 60 * 60)] = "harbor"
            self.seaship_schedule[(start_sea + 60 * 60, start_sea + 120 * 60)] = "blocked"
        elif self.seaship_status == 3:
            self.seaship_schedule[(start_sea + 45 * 60, mid_sea)] = "blocked"
            self.seaship_schedule[(mid_sea, mid_sea + 15 * 60)] = "canal"
            self.seaship_schedule[(mid_sea + 15 * 60, mid_sea + 45 * 60)] = "blocked"
        return start_sea

    def get_signal(self, active_trips: List[TripState]):
        self.min_canal = 0
        self.min_harbor = 0
        self.active_trips = active_trips
        for at in active_trips:
            location = at.get_loc_current()

            if location.loc in self.one_way_area:
                direction = location.direction
                time_out_of_ow = self.expected_time_through_one_way(at)

                if direction == "canal":
                    self.min_canal = max(time_out_of_ow, self.min_canal)
                elif direction == "harbor":
                    self.min_harbor = max(time_out_of_ow, self.min_harbor)

                if self.min_canal > 0 and self.min_harbor > 0:
                    pass
                    # raise ValueError("Ships on collision course :O")

        if self.schedule_type == "dynamic":
            return self.scenario_queue_length_switching()
        else:
            return self.scenario_regular()

        # return self.scenario_regular()#self.MODELS[self.model](self)

    def _signal_at_index(self, ctime, index, signal_duration_peak, signal_duration_off):
        signal, duration = signal_duration_peak[index]
        if ctime < self.off_peak_morning * 60 or ctime > self.off_peak_evening * 60:
            signal, duration = signal_duration_off[index]
        return duration, signal

    def _get_partial_schedule(self, start_time, end_time, start_index, signal_duration_peak, signal_duration_off,
                              reverse_schedule=False):
        schedule: Dict[Interval, str] = {}
        ctime = start_time
        index = start_index

        if not reverse_schedule:
            while ctime < end_time:
                duration, signal = self._signal_at_index(ctime, index, signal_duration_peak, signal_duration_off)

                schedule[(ctime, ctime + duration)] = signal
                ctime += duration
                index = (index + 1) % len(signal_duration_off)
        else:
            while ctime > end_time:
                duration, signal = self._signal_at_index(ctime, index, signal_duration_peak, signal_duration_off)
                schedule[(ctime - duration, ctime)] = signal
                ctime -= duration
                index = (index + 1) % len(signal_duration_off)

        return schedule

    def expected_time_through_one_way(self, at: TripState):
        remaining_time_in_ow = sum(location.sail_time for location in at.path_directed[at.get_path_index() + 1:]
                                   if location.loc in self.one_way_area)

        if at.get_loc_current().loc in self.one_way_area:
            remaining_time_in_ow += at.get_remaining_time_at_loc(self.env.now)

        return remaining_time_in_ow

    def create_regular_schedule_sea(self, open_time: int, block_time: int, start_index=2) -> Dict[Interval, str]:
        schedule: Dict[Interval, str] = {}
        signal_duration_peak = [("canal", open_time), ("blocked", block_time),
                                ("harbor", open_time), ("blocked", block_time)]
        signal_duration_off = [("canal", self.open_time_off), ("blocked", block_time),
                               ("harbor", self.open_time_off), ("blocked", block_time)]
        if self.seaship_status == 0:
            schedule.update(
                self._get_partial_schedule(0, 25 * 60 * 60, start_index, signal_duration_peak, signal_duration_off))
        elif self.seaship_status >= 1:
            schedule.update(
                self._get_partial_schedule(self.start_sea, 0, 0, signal_duration_peak, signal_duration_off, True))

            schedule.update({k: v for k, v in self.seaship_schedule.items() if v in {"harbor", "canal", "blocked"}})

            if self.seaship_status == 1:
                index = 0
                ctime = self.start_sea + 120 * 60
            elif self.seaship_status == 2:
                index = 0
                ctime = self.start_sea + 135 * 60
            else:
                index = 2
                ctime = self.start_sea + 135 * 60

            schedule.update(
                self._get_partial_schedule(ctime, 25 * 60 * 60, index, signal_duration_peak, signal_duration_off))
        return schedule

    def scenario_regular(self):
        schedule_key = (self.open_time, self.block_time)
        if schedule_key not in self.schedules:
            self.schedules[schedule_key] = self.create_regular_schedule_sea(*schedule_key)
        schedule = self.schedules[schedule_key]
        intervals = find_intervals_intersecting_at_point(schedule.keys(), self.env.now)
        if len(intervals) == 0:
            return self.scenario_serve_open()
        interval = intervals[-1]
        signal = schedule[interval]
        time_to_new_signal = max(interval[1] - self.env.now, 1)

        if signal == "canal":
            if self.min_harbor != 0:
                return "blocked", self.min_harbor
            return signal, max(time_to_new_signal, self.min_canal)
        elif signal == "harbor":
            if self.min_canal != 0:
                return "blocked", self.min_canal
            return signal, max(time_to_new_signal, self.min_harbor)
        elif signal == "free":
            return self.scenario_serve_open()
        elif signal == "blocked":
            return signal, time_to_new_signal
        else:
            raise ValueError(f"The signal {signal} is not known!")

    def scenario_serve_open(self):
        if self.min_canal != 0:
            if self.min_harbor != 0:
                raise RuntimeError("BLOKKADE")
            else:
                return "canal", self.min_canal
        elif self.min_harbor != 0:
            return "harbor", self.min_harbor
        else:
            return "free", 0

    #
    # def scenario_fixed_time15(self):
    #     return self.scenario_regular(self.schedules[0])
    #
    # def scenario_fixed_mixed30_15(self):
    #     return self.scenario_regular(self.schedules[1])
    #
    # def scenario_fixed_mixed45_15(self):
    #     return self.scenario_regular(self.schedules[2])

    def scenario_queue_length_switching(self):
        canal_queue = len(get_queueing(self.active_trips, self.env.now, self.one_way_canal))
        harbor_queue = len(get_queueing(self.active_trips, self.env.now, self.one_way_harbor))

        # queue_trigger_length = 3

        if self.seaship_status > 0:
            if self.start_sea - 30 * 60 < self.env.now < self.start_sea + 130 * 60:
                intervals = find_intervals_intersecting_at_point(self.seaship_schedule.keys(), self.env.now)
                if len(intervals) != 0:
                    self.qlss = "open"
                    interval = intervals[-1]
                    signal = self.seaship_schedule[interval]
                    if signal == "not_harbor":
                        if self.min_harbor != 0:
                            return "blocked", self.min_harbor
                        return "canal", max(self.min_canal, interval[1] - self.env.now)
                    return signal, max(interval[1] - self.env.now, 1)
                else:
                    pass

        if self.qlss == "resolve_canal_queue":
            if canal_queue == 0:
                self.qlss = "open"
                # print(self.env.now, "Canal queue empty ", self.min_canal, self.min_harbor)
            else:
                if self.min_harbor != 0:
                    return "blocked", self.min_harbor
                return "canal", max(self.min_canal, 1)
        elif self.qlss == "resolve_harbor_queue":
            if harbor_queue == 0:
                self.qlss = "open"
                # print(self.env.now, "Harbor queue empty ", self.min_canal, self.min_harbor)
            else:
                if self.min_canal != 0:
                    return "blocked", self.min_canal
                return "harbor", max(self.min_harbor, 1)

        if canal_queue < self.queue_trigger_length and harbor_queue < self.queue_trigger_length:
            return self.scenario_serve_open()
        elif canal_queue > harbor_queue:
            self.qlss = "resolve_canal_queue"
            if self.min_harbor != 0:
                return "blocked", self.min_harbor
            return "canal", max(self.min_canal, 1)
        else:
            self.qlss = "resolve_harbor_queue"
            if self.min_canal != 0:
                return "blocked", self.min_canal
            return "harbor", max(self.min_harbor, 1)

    # MODELS = {"Fixed_Time15": scenario_fixed_time15,
    #           #  "Serve_Open": scenario_serve_open,
    #           "Mixed_Time30_15": scenario_fixed_mixed30_15,
    #           "Mixed_Time45_15": scenario_fixed_mixed45_15,
    #           "Queue_length_switching": scenario_queue_length_switching}
