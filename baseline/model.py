import json

from typing import List, Optional, Dict

from dataclasses import dataclass

TOTAL_MINUTES = 1440

SPARE = 0
KOVKA = 1
PROKAT = 4
OTHER = 2


@dataclass(init=True)
class Oven:
    id: int
    start_temp: int
    working_temps: List[int]
    operations: List[str]


@dataclass(init=True)
class Operation:
    name: str
    start_time: int
    end_time: int
    timing: int


@dataclass(init=True)
class ScheduleTask:
    name: str
    temperature: int
    start_time: int
    end_time: int


@dataclass(init=True)
class Series:
    total_time: int
    priority_index: int
    temperature: int
    operations: List[Operation]


class OvenSchedule:
    def __init__(self, oven: Oven):
        self.current_time = 0
        self.params = oven
        self.current_temp = oven.start_temp
        self.minutes_vector = [0] * TOTAL_MINUTES
        self.tasks: List[ScheduleTask] = []

    def __str__(self):
        return str(self.tasks)

    def add_task(self, series: Series, t: int = 0) -> bool:
        self.current_time = t
        sup_time = self.current_time + series.total_time
        time_counter = self.current_time
        delta = 0 if self.current_temp == series.temperature else 120
        if (sup_time + delta) <= TOTAL_MINUTES:
            for op in series.operations:

                if series.temperature != self.current_temp:
                    warmup_task = ScheduleTask(
                        name="smena_temp",
                        temperature=series.temperature,
                        start_time=time_counter,
                        end_time=(time_counter + 120)
                    )
                    for k in range(warmup_task.start_time, warmup_task.end_time + 1):
                        self.minutes_vector[k] = 2
                    self.tasks.append(warmup_task)
                    self.current_temp = warmup_task.temperature
                    time_counter += 120
                    self.current_time = time_counter

                schedule_task = ScheduleTask(
                    name=op.name,
                    temperature=series.temperature,
                    start_time=time_counter,
                    end_time=(time_counter + op.timing)
                )
                self.current_temp = series.temperature
                # l = len(self.tasks)
                l = KOVKA if schedule_task.name in [
                    'kovka', 'prokat'] else OTHER
                for k in range(schedule_task.start_time, schedule_task.end_time + 1):
                    self.minutes_vector[k] = l
                time_counter += op.timing
                self.current_time = sup_time
                self.tasks.append(schedule_task)
            self.current_time = sup_time
            return True
        return False


@dataclass(init=True)
class ForgererGroup:
    ovens_under_control: List[OvenSchedule]


def read(filename: str) -> object:
    with open(filename) as f:
        data = json.load(f)
    return data


def schedule_series(schedule: OvenSchedule, series: Series) -> bool:
    return schedule.add_task(series)


def process_step(ovens: Dict[int, OvenSchedule], series_seq: List[Series], step: str) -> Dict[int, OvenSchedule]:
    for series in series_seq:
        for oven_id, oven_schedule in ovens.items():
            if step == 'important':
                if schedule_series(oven_schedule, series):
                    break


def transform(data: object):

    STEPS = ['important', 'no_change', 'decrease', 'increase']
    series_seq: List[Series] = []
    for idx, series in enumerate(data['series'][:15]):

        _operations = [Operation(**d, start_time=0, end_time=0)
                       for d in series['operations']]

        series_obj = Series(
            total_time=sum(
                [op.timing for op in _operations]
            ),
            priority_index=idx,
            temperature=series['temperature'],
            operations=_operations,
        )
        accum = 0
        for oper in series_obj.operations:
            oper.start_time = accum
            oper.end_time = oper.timing + oper. start_time

            accum = oper.end_time

        series_seq.append(series_obj)

    ovens_seq: List[Oven] = []
    for idx, oven in enumerate(data['ovens'][:15]):
        oven_obj = Oven(**oven, id=idx)
        ovens_seq.append(oven_obj)

    schedule = {}

    for i, idx in enumerate(range(0, len(ovens_seq), 7)):
        schedule[i] = ForgererGroup(
            ovens_under_control=[OvenSchedule(oven) for oven in ovens_seq[idx:idx+7]])

    sorted_series = sorted(
        series_seq, key=lambda s: (-s.total_time, s.priority_index))

    for step in STEPS:
        while sorted_series:
            print(sorted_series)
            series = sorted_series[0]


            oven_exist = False
            forgerer_group: ForgererGroup
            for forgerer_id, forgerer_group in schedule.items():
                for oven in forgerer_group.ovens_under_control:
                    if step == 'important':
                        schedule_series(oven, series)
                        oven_exist = True
                    elif step == 'no_change':
                        if oven.current_temp == series.temperature:
                            oven_exist = True
                    elif step == 'decrease':
                        if oven.current_temp > series.temperature:
                            oven_exist = True
                    elif step == 'increase':
                        if oven.current_temp < series.temperature:
                            oven_exist = True
                    else:
                        pass
            if oven_exist:
                for forgerer_id, forgerer_group in schedule.items():
                    matrix = [
                        oven.minutes_vector for oven in forgerer_group.ovens_under_control]
                    print(matrix)
                    possible_place_id = -1
                    possible_time = ()

                    for t in range(1441):
                        s_forge = 0  # Sum of kovka
                        s_roll = 0

                        filtered_ops_kovka = [
                            s for s in series.operations if s.name == 'kovka']
                        is_suitable = True
                        found_slice = (-1, -1)
                        for oven in forgerer_group.ovens_under_control:
                            for kovka in filtered_ops_kovka:
                                slice = (t + kovka.start_time,
                                         t + kovka.end_time + 1)

                                if slice[0] >= TOTAL_MINUTES or slice[1] >= TOTAL_MINUTES:
                                    is_suitable = False
                                    break

                                for p in range(slice[0], slice[1]):
                                    column_sum = 0
                                    for g in range(len(matrix)):
                                        if matrix[g][p] == KOVKA:
                                            column_sum += 1
                                    if column_sum > 0:
                                        is_suitable = False
                                        break

                                if (sum(oven.minutes_vector[t: series.total_time]) != 0):
                                    is_suitable = False
                                    break
                                if sum(found_slice) == -2 and is_suitable:
                                    found_slice = (oven.params.id, t)
                                if oven.params.id < found_slice[0] and is_suitable:
                                    found_slice = (oven.params.id, t)

                        if is_suitable:
                            possible_time = found_slice
                    t = possible_time
                    print(t)
                    if sum(t) >= 0:
                        print("Adding")
                        print(series)
                        forgerer_group.ovens_under_control[t[0] % 7].add_task(series, t=t[1])
                        sorted_series.pop(0)
    
            for i, group in schedule.items():
                print()
                print(f"Schedule for group: {i}")
                group: ForgererGroup
                for oven in group.ovens_under_control:
                    print(sum(oven.minutes_vector))
                    for task in oven.tasks:
                        print(f"Name: {task.name:10}. Start: {task.start_time:5d}. End: {task.end_time:5d}. T: {task.temperature:5d}")
            o = input()


