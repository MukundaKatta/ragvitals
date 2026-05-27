"""PhoenixSink: stream ragvitals drift reports to Arize Phoenix.

Arize Phoenix is an open-source LLM observability platform that ingests
OpenTelemetry spans. Each ragvitals DetectorReport becomes one parent
span ("ragvitals.detector.report") with one child span per dimension,
plus the dimension's drift score and severity as attributes. Phoenix
renders the resulting timeline so on-call engineers can correlate drift
events with their existing LLM call traces in the same UI.

Usage:

    from ragvitals import Detector, PhoenixSink

    det = Detector(
        dimensions=[...],
        sinks=[PhoenixSink(
            endpoint="http://localhost:4317",   # local Phoenix
            project_name="ragvitals",
        )],
    )

For Arize Cloud (managed Phoenix), point ``endpoint`` at the OTLP
collector URL from your Arize space settings and add ``headers={"api-key":
"<arize-api-key>"}``.

Requires:
    pip install ragvitals[phoenix]
which pulls opentelemetry-sdk + opentelemetry-exporter-otlp-proto-grpc.
The hard dependency on OpenTelemetry is opt-in so the core ragvitals
install stays zero-dep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ragvitals.types import DetectorReport, Severity


_SEVERITY_LABEL = {
    Severity.OK: "OK",
    Severity.WARN: "WARN",
    Severity.DEGRADED: "DEGRADED",
}


@dataclass
class PhoenixSink:
    """Emit DetectorReport as OpenTelemetry spans to Arize Phoenix."""

    endpoint: str = "http://localhost:4317"
    project_name: str = "ragvitals"
    headers: Mapping[str, str] | None = None
    service_name: str = "ragvitals.detector"
    _tracer: object | None = field(default=None, init=False, repr=False)
    _provider: object | None = field(default=None, init=False, repr=False)

    def _get_tracer(self):
        if self._tracer is not None:
            return self._tracer
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as e:
            raise ImportError(
                "PhoenixSink requires OpenTelemetry. Install with "
                "`pip install ragvitals[phoenix]`."
            ) from e

        # Phoenix reads ``openinference.project.name`` from the resource and
        # groups spans into the matching project (mirrors how
        # arize-phoenix-otel sets it).
        resource = Resource.create({
            "service.name": self.service_name,
            "openinference.project.name": self.project_name,
        })
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=self.endpoint,
            headers=dict(self.headers) if self.headers else None,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))

        self._provider = provider
        self._tracer = provider.get_tracer("ragvitals", "0.3.0")
        return self._tracer

    def emit(self, report: DetectorReport) -> None:
        tracer = self._get_tracer()
        start_ns = int(report.window_start.timestamp() * 1_000_000_000)
        end_ns = int(report.window_end.timestamp() * 1_000_000_000)

        overall = (
            "DEGRADED" if report.degraded
            else "WARN" if report.warned
            else "OK"
        )

        with tracer.start_as_current_span(
            "ragvitals.detector.report",
            start_time=start_ns,
            attributes={
                "ragvitals.window.dimension_count": len(report.dimensions),
                "ragvitals.window.overall_severity": overall,
                "ragvitals.window.degraded_dimensions": ",".join(report.degraded),
                "ragvitals.window.warned_dimensions": ",".join(report.warned),
            },
        ) as parent:
            for d in report.dimensions:
                attrs: dict[str, object] = {
                    "ragvitals.dimension": d.name,
                    "ragvitals.severity": _SEVERITY_LABEL[d.severity],
                    "ragvitals.sample_size": d.sample_size,
                }
                if d.value is not None:
                    attrs["ragvitals.score"] = float(d.value)
                if d.baseline is not None:
                    attrs["ragvitals.baseline"] = float(d.baseline)
                if d.z_score is not None:
                    attrs["ragvitals.z_score"] = float(d.z_score)
                if d.detail:
                    attrs["ragvitals.detail"] = d.detail
                tracer.start_span(
                    f"ragvitals.dimension.{d.name}",
                    start_time=start_ns,
                    attributes=attrs,
                ).end(end_time=end_ns)
            parent.end(end_time=end_ns)
