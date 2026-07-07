import { useCallback, useEffect, useMemo, useState } from 'react';
import { ChartToolbar } from '../components/market/ChartToolbar';
import { KlineChart } from '../components/market/KlineChart';
import { LinePriceChart } from '../components/market/LinePriceChart';
import { getKlines, getSignals } from '../services/api';
import type { Kline, SignalRecord } from '../types/market';

export function ChartPage() {
  const [chartType, setChartType] = useState<'kline' | 'line'>('kline');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [klines, setKlines] = useState<Kline[]>([]);
  const [signals, setSignals] = useState<SignalRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();

  const loadChart = useCallback(async () => {
    setLoading(true);
    const [klineResult, signalResult] = await Promise.all([
      getKlines('BTCUSDT', '5m', 500),
      getSignals(),
    ]);

    setKlines((current) => (klineResult.fromMock && current.length > 0 ? current : klineResult.data));
    setSignals((current) => (signalResult.fromMock && current.length > 0 ? current : signalResult.data));
    setError(klineResult.error ?? signalResult.error);
    setLoading(false);
  }, []);

  useEffect(() => {
    void loadChart();
  }, [loadChart]);

  useEffect(() => {
    if (!autoRefresh) return undefined;

    const timer = window.setInterval(() => {
      void loadChart();
    }, 30000);

    return () => window.clearInterval(timer);
  }, [autoRefresh, loadChart]);

  const latestPrice = klines.at(-1)?.close;
  const visibleSignals = useMemo(
    () => signals.filter((item) => item.signal !== 'NO_BUY').slice(-80),
    [signals],
  );

  return (
    <div className="page-stack">
      {error && <div className="status-banner">{error}</div>}

      <section className="panel chart-panel chart-page-panel fade-rise">
        <div className="panel-heading">
          <div>
            <div className="panel-kicker">BTCUSDT 5m</div>
            <h2>{chartType === 'kline' ? 'K线图' : '价格折线图'}</h2>
          </div>
          <div className="chart-heading-actions">
            <div className="latest-price-pill">最新 ${latestPrice?.toLocaleString('en-US', { maximumFractionDigits: 1 }) ?? '--'}</div>
            <ChartToolbar
              chartType={chartType}
              autoRefresh={autoRefresh}
              loading={loading}
              onChartTypeChange={setChartType}
              onAutoRefreshChange={setAutoRefresh}
              onRefresh={loadChart}
            />
          </div>
        </div>

        {klines.length === 0 ? (
          <div className="empty-panel">暂无图表数据。</div>
        ) : chartType === 'kline' ? (
          <KlineChart klines={klines} signals={visibleSignals} latestPrice={latestPrice} />
        ) : (
          <LinePriceChart klines={klines} signals={visibleSignals} latestPrice={latestPrice} />
        )}
      </section>
    </div>
  );
}
