from __future__ import annotations

from arc_model_lab.api.schemas.experiments import MetricAggregateOut
from arc_model_lab.domain import ExperimentMetricAggregate


def test_metric_aggregate_out_from_domain_maps_all_fields() -> None:
    aggregate = ExperimentMetricAggregate(metric_name="faithfulness", average_score=0.875, evaluated_count=8)

    response = MetricAggregateOut.from_domain(aggregate)

    assert response.metric_name == "faithfulness"
    assert response.average_score == 0.875
    assert response.evaluated_count == 8
