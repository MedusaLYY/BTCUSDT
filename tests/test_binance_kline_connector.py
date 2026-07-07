from data.fetch_binance_klines import (
    market_klines_to_dicts,
    parse_binance_kline,
)


def test_parse_binance_kline_maps_array_to_frontend_contract():
    row = [
        0,
        "63000.1",
        "63100.2",
        "62900.3",
        "63050.4",
        "123.45",
        299999,
        "0",
        42,
        "0",
        "0",
        "0",
    ]

    kline = parse_binance_kline(row)

    assert kline.openTime == "1970-01-01 08:00"
    assert kline.open == 63000.1
    assert kline.high == 63100.2
    assert kline.low == 62900.3
    assert kline.close == 63050.4
    assert kline.volume == 123.45


def test_market_klines_to_dicts_returns_frontend_keys():
    kline = parse_binance_kline([0, "1", "2", "0.5", "1.5", "10"])

    payload = market_klines_to_dicts([kline])

    assert payload == [
        {
            "openTime": "1970-01-01 08:00",
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 10.0,
        }
    ]
