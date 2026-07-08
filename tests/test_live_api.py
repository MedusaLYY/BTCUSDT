from api import app as api_app


def test_get_live_metrics_returns_live_metrics_payload(monkeypatch):
    calls = []

    def fake_load_live_metrics():
        calls.append("metrics")
        return {
            "settled_count": 2,
            "pending_count": 1,
            "signal_hit_rate": 0.5,
        }

    monkeypatch.setattr(api_app, "load_live_metrics", fake_load_live_metrics)

    response = api_app.get_live_metrics()

    assert calls == ["metrics"]
    assert response["settled_count"] == 2
    assert response["pending_count"] == 1


def test_get_live_predictions_returns_settlement_fields(monkeypatch):
    calls = []

    def fake_load_live_prediction_records(limit: int):
        calls.append(limit)
        return [
            {
                "id": "LIVE-1",
                "openTime": "2026-07-08 10:00",
                "currentPrice": 100.0,
                "signal": "BUY",
                "buyProbability": 0.63,
                "predReturn": 0.003,
                "predHighPrice": 100.3,
                "settlement_status": "SETTLED",
                "actual_future_max_return_30m": 0.004,
                "actual_y_buy": True,
                "classification_hit": True,
                "signal_hit": True,
                "return_abs_error": 0.001,
            }
        ]

    monkeypatch.setattr(
        api_app,
        "load_live_prediction_records",
        fake_load_live_prediction_records,
    )

    response = api_app.get_live_predictions(limit=30)

    assert calls == [30]
    assert response[0]["settlement_status"] == "SETTLED"
    assert response[0]["actual_future_max_return_30m"] == 0.004
    assert response[0]["signal_hit"] is True
