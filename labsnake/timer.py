import logging
from time import perf_counter

logger = logging.getLogger(__name__)


class Timer:
    """ Measures the time consumption of a code block in a context managed fashion. """

    def __init__(self, label: str, logging_period: int = 10) -> None:
        self._label = label
        self._current_time = perf_counter()
        self._time_logs = []
        self._logging_period = logging_period

        self._counter = 0

    def __enter__(self):
        self._counter += 1
        self._current_time = perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_diff = perf_counter() - self._current_time
        self._time_logs.append(time_diff)

        if len(self._time_logs) == self._logging_period:
            mean_consumption = sum(self._time_logs) / len(self._time_logs)
            logger.info(
                f"Average time consumption of {self._label} over {self._logging_period} occurences: {mean_consumption:.3f} ({1 / mean_consumption:.3f} Hz)")
            self._reset()

    def _reset(self):
        self._counter = 0
        self._time_logs = []
