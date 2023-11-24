import json

from typing import List, Optional, Dict
from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm

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
        time_temporary = t
        sup_time = time_temporary + series.total_time
        time_counter = self.current_time
        delta = 0 if self.current_temp == series.temperature else 120
        if (sup_time + delta) <= TOTAL_MINUTES:
            self.current_time = time_temporary
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
                # if schedule_task.end_time > TOTAL_MINUTES:
                #     return False
                for k in range(schedule_task.start_time, schedule_task.end_time):
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
    target = ['kovka', 'prokat']
    operation = {
        'name': 'podogrev',
        'timing': 120
    }

    for s_ind in range(len(data['series'])):
        start = None
        length = 0
        for ind in range(len(data['series'][s_ind]['operations']) - 1):
            if data['series'][s_ind]['operations'][ind]['name'] in target and data['series'][s_ind]['operations'][ind+1]['name'] in target:
                if start is None:
                    start = ind + 1
                length += 1
            else:
                pass
        while length != 0:
            data['series'][s_ind]['operations'].insert(start, operation)
            start += 2
            length -= 1

    for idx, series in enumerate(data['series'][:200]):

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
    for idx, oven in enumerate(data['ovens'][:7]):
        oven_obj = Oven(**oven, id=idx)
        ovens_seq.append(oven_obj)

    schedule = {}

    for i, idx in enumerate(range(0, len(ovens_seq), 7)):
        schedule[i] = ForgererGroup(
            ovens_under_control=[OvenSchedule(oven) for oven in ovens_seq[idx:idx+7]])

    sorted_series = sorted(
        series_seq, key=lambda s: (-s.total_time, s.priority_index))

    for step in tqdm(STEPS):
        while sorted_series:

            series = sorted_series[0]

            oven_exist = False
            forgerer_group: ForgererGroup
            for forgerer_id, forgerer_group in schedule.items():
                for oven in forgerer_group.ovens_under_control:
                    if step == 'important':
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
                found = False
                possible_time = (-1, -1)
                for forgerer_id, forgerer_group in schedule.items():
                    matrix = [
                        oven.minutes_vector for oven in forgerer_group.ovens_under_control]

                    for t in range(1441):

                        found = False
                        for oven in forgerer_group.ovens_under_control:
                            is_suitable = True
                            start_time = oven.current_time

                            if start_time + series.total_time <= TOTAL_MINUTES:

                                k = deepcopy(series)
                                for op_idx, op in enumerate(k.operations):
                                    if op_idx > 0:
                                        op.start_time = k.operations[op_idx - 1].end_time
                                        op.end_time = op.start_time + op.timing
                                    if op.name == 'kovka':
                                        success = False
                                        while not success:
                                            slice = (
                                                start_time + op.start_time, start_time + op.end_time + 1)

                                            if slice[0] >= TOTAL_MINUTES or slice[1] >= TOTAL_MINUTES:
                                                success = False
                                                break
                                            ll = 0
                                            for p in range(slice[0], slice[1]):
                                                for g in range(len(matrix)):
                                                    if matrix[g][p] == KOVKA:
                                                        ll += 1

                                            if ll != 0:
                                                op.start_time += 15
                                                op.end_time += 15
                                            elif ll == 0:
                                                success = True
                                                possible_time = (
                                                    oven.params.id, start_time
                                                )

                                        if not success:
                                            is_suitable = False
                                    else:
                                        continue

                                if is_suitable:
                                    series = k
                                    found = True
                                    break
                    if possible_time == (-1, -1) or not forgerer_group.ovens_under_control[possible_time[0] % 7].add_task(series, possible_time[1]):
                        found = False
                    else:
                        found = True
                        sorted_series.pop(0)
                        break
                if not found:
                    sorted_series.pop(0)

    for i, group in schedule.items():
        print()
        print(f"Schedule for group: {i}")
        group: ForgererGroup
        for oven in group.ovens_under_control:
            print(
                f"Oven {oven.params.id}. Start temp: {oven.params.start_temp}")
            for task in oven.tasks:
                print(
                    f"\tName: {task.name:10}. Start: {task.start_time:5d}. End: {task.end_time:5d}. T: {task.temperature:5d}")
    return schedule
