"""
What the service is doing, in Prometheus text format.

Written against the exposition format rather than ``prometheus_client`` so that
metrics cost no dependency: the wire format is stable and this needs counters,
a gauge and one histogram. Swapping in the library later means replacing this
module, not the call sites.

Values live in the process, so they reset on restart and are per-replica —
which is what a Prometheus counter is anyway.
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Iterable

# Seconds. A global search is N+1 LLM calls, so the tail matters more than the
# head and the buckets run out to minutes.
DURATION_BUCKETS: tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300,
)

Labels = tuple[tuple[str, str], ...]


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _number(value: float) -> str:
    """
    Render a bucket bound the way Prometheus expects it.
    """
    return repr(int(value)) if float(value).is_integer() else repr(value)


class _Histogram:
    """
    One histogram series: per-bucket counts plus the sum and total count.
    """

    __slots__ = ("buckets", "total", "count")

    def __init__(self) -> None:
        self.buckets = [0] * len(DURATION_BUCKETS)
        self.total = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for index, bound in enumerate(DURATION_BUCKETS):
            if value <= bound:
                self.buckets[index] += 1
                return


def _render_labels(labels: Labels) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{name}="{_escape(value)}"' for name, value in labels)
    return "{" + inner + "}"


class Metrics:
    """
    A counter/gauge/histogram registry with a Prometheus text rendering.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, dict[Labels, float]] = defaultdict(dict)
        self._gauges: dict[str, dict[Labels, float]] = defaultdict(dict)
        # Bucket counts plus sum and count, not the observations themselves: a
        # long-running service must not accumulate one float per request.
        self._histograms: dict[str, dict[Labels, "_Histogram"]] = defaultdict(dict)
        self._help: dict[str, tuple[str, str]] = {}

    def describe(self, name: str, kind: str, help_text: str) -> None:
        """
        Record the HELP and TYPE lines for a metric.
        """
        self._help[name] = (kind, help_text)

    def increment(self, name: str, labels: Labels = (), amount: float = 1.0) -> None:
        """
        Add to a counter.
        """
        with self._lock:
            series = self._counters[name]
            series[labels] = series.get(labels, 0.0) + amount

    def set(self, name: str, value: float, labels: Labels = ()) -> None:
        """
        Set a gauge.
        """
        with self._lock:
            self._gauges[name][labels] = value

    def observe(self, name: str, value: float, labels: Labels = ()) -> None:
        """
        Record one observation in a histogram.
        """
        with self._lock:
            series = self._histograms[name]
            histogram = series.get(labels)
            if histogram is None:
                histogram = _Histogram()
                series[labels] = histogram
            histogram.observe(value)

    def render(self) -> str:
        """
        Render every metric in the Prometheus text exposition format.
        """
        with self._lock:
            lines: list[str] = []
            for name, counters in sorted(self._counters.items()):
                lines.extend(self._render_simple(name, counters, "counter"))
            for name, gauges in sorted(self._gauges.items()):
                lines.extend(self._render_simple(name, gauges, "gauge"))
            for name, histograms in sorted(self._histograms.items()):
                lines.extend(self._render_histogram(name, histograms))
        return "\n".join(lines) + "\n"

    def _header(self, name: str, default_kind: str) -> list[str]:
        kind, help_text = self._help.get(name, (default_kind, name))
        return [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]

    def _render_simple(
        self, name: str, series: dict[Labels, float], kind: str
    ) -> Iterable[str]:
        yield from self._header(name, kind)
        for labels, value in sorted(series.items()):
            yield f"{name}{_render_labels(labels)} {value}"

    def _render_histogram(
        self, name: str, series: dict[Labels, "_Histogram"]
    ) -> Iterable[str]:
        yield from self._header(name, "histogram")
        for labels, histogram in sorted(series.items()):
            running = 0
            for bucket, count in zip(DURATION_BUCKETS, histogram.buckets):
                running += count
                bucket_labels = labels + (("le", _number(bucket)),)
                yield f"{name}_bucket{_render_labels(bucket_labels)} {running}"
            inf_labels = labels + (("le", "+Inf"),)
            yield f"{name}_bucket{_render_labels(inf_labels)} {histogram.count}"
            yield f"{name}_sum{_render_labels(labels)} {histogram.total}"
            yield f"{name}_count{_render_labels(labels)} {histogram.count}"


metrics = Metrics()

REQUESTS = "ragu_api_requests_total"
DURATION = "ragu_api_request_duration_seconds"
SEARCHES = "ragu_api_searches_total"
GRAPHS = "ragu_api_graphs"
JOBS = "ragu_api_jobs"

metrics.describe(REQUESTS, "counter", "HTTP requests handled, by route and status.")
metrics.describe(DURATION, "histogram", "How long requests take, in seconds.")
metrics.describe(
    SEARCHES, "counter", "Searches by mode and outcome; degraded means a child engine or the reranker did not contribute."
)
metrics.describe(GRAPHS, "gauge", "Configured graphs, by whether they loaded.")
metrics.describe(JOBS, "gauge", "Jobs this process knows about, by state.")


class Timer:
    """
    Context manager that records how long its block took.
    """

    def __init__(self, name: str, labels: Labels = ()):
        self.name = name
        self.labels = labels

    def __enter__(self) -> "Timer":
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        metrics.observe(self.name, time.perf_counter() - self._started, self.labels)
