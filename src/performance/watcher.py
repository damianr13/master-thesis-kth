import warnings
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict


class ExecutionSegment(Enum):
    TOKENIZING = 1
    MODEL_FEEDFORWARD = 2


class PerformanceWatcher:
    __instance = None

    current_segment: Optional[ExecutionSegment] = None
    current_segment_start: Optional

    stats: Dict[Enum, timedelta]

    @staticmethod
    def get_instance():
        if PerformanceWatcher.__instance is None:
            PerformanceWatcher.__instance = PerformanceWatcher()

        return PerformanceWatcher.__instance

    def __init__(self):
        if PerformanceWatcher.__instance is None:
            raise Exception("This class is a singleton")

        self.stats = {k: timedelta(0) for k in ExecutionSegment}

        PerformanceWatcher.__instance = self

    def enter_segment(self, seg: ExecutionSegment):
        self.current_segment_start = datetime.now()
        self.current_segment = seg

    def exit_segment(self):
        if self.current_segment is None:
            warnings.warn('Exit signal received without being inside a segment')
            return

        self.stats[self.current_segment] += datetime.now() - self.current_segment_start
        self.current_segment = None
        self.current_segment_start = None

    def print_stats(self):
        for k, v in self.stats.items():
            print(f'Spent {v} in {k}')
