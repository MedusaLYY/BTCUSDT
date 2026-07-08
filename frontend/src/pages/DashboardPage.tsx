import { useCallback, useEffect, useMemo, useState } from 'react';
import { MetricCard } from '../components/backtest/MetricCard';
import { KlineChart } from '../components/market/KlineChart';
import { PriceHeader } from '../components/market/PriceHeader';
import { PredictionPanel } from '../components/signal/PredictionPanel';
import { SignalTable } from '../components/signal/SignalTable';
import { getKlines, getLatestSignal, getLiveMetrics, getLivePredictions } from '../services/api';
import { formatInteger, formatPercent, formatSignedPercent } from '../utils/format';
import type { Kline, LiveMetrics, PredictionSignal, SignalRecord } from '../types/market';

function calculateChange24h(klines: Kline[]) {
  if (klines.length < 289) return 0;

  const latest = klines[klines.length - 1];
  const previous = klines[klines.length - 289];

  return (latest.close - previous.close) / previous.close;
}

function optionalMetric(value?: number | null) {
  return value === null ? undefined : value;
}

export function DashboardPage() {
  const [klines, setKlines] = useState<Kline[]>([]);
  const [latestSignal, setLatestSignal] = useState<PredictionSignal>();
  const [signals, setSignals] = useState<SignalRecord[]>([]);
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [fromMock, setFromMock] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<string>();

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    const [klineResult, signalResult, recordsResult, metricsResult] = await Promise.all([
      getKlines('BTCUSDT', '5m', 500),
      getLatestSignal(),
      getLivePredictions(30),
      getLiveMetrics(),
    ]);

    setKlines((current) => (klineResult.fromMock && current.length > 0 ? current : klineResult.data));
    setLatestSignal((current) => (signalResult.fromMock && current ? current : signalResult.data));
    setSignals((current) => (recordsResult.fromMock && current.length > 0 ? current : recordsResult.data));
    setLiveMetrics((current) => (metricsResult.fromMock && current ? current : metricsResult.data));
    setFromMock(klineResult.fromMock || signalResult.fromMock || recordsResult.fromMock || metricsResult.fromMock);
    setError(klineResult.error ?? signalResult.error ?? recordsResult.error ?? metricsResult.error);
    setUpdatedAt(signalResult.updatedAt);
    setLoading(false);
  }, []);

  useEffect(() => {
    void loadDashboard();

    const timer = window.setInterval(() => {
      void loadDashboard();
    }, 30000);

    return () => window.clearInterval(timer);
  }, [loadDashboard]);

  const change24h = useMemo(() => calculateChange24h(klines), [klines]);
  const latestPrice = latestSignal?.currentPrice ?? klines.at(-1)?.close;
  const recentSignals = useMemo(
    () => signals.slice().sort((a, b) => b.openTime.localeCompare(a.openTime)).slice(0, 30),
    [signals],
  );

  return (
    <div className="page-stack">
      <div className="dashboard-actions">
        <button type="button" className="terminal-button" onClick={loadDashboard} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && <div className="status-banner">{error}</div>}

      <PriceHeader signal={latestSignal} change24h={change24h} updatedAt={updatedAt} fromMock={fromMock} />

      <section className="metric-grid fade-rise-delay">
        <MetricCard label="Settled" value={formatInteger(liveMetrics?.settled_count)} caption="live predictions" />
        <MetricCard label="Pending" value={formatInteger(liveMetrics?.pending_count)} caption="waiting 6 bars" />
        <MetricCard label="Signal hit" value={formatPercent(optionalMetric(liveMetrics?.signal_hit_rate))} tone="positive" />
        <MetricCard label="BUY hit" value={formatPercent(optionalMetric(liveMetrics?.buy_signal_hit_rate))} tone="positive" />
        <MetricCard label="Last 30 hit" value={formatPercent(optionalMetric(liveMetrics?.last_30_signal_hit_rate))} />
        <MetricCard label="Mean error" value={formatSignedPercent(optionalMetric(liveMetrics?.mean_return_error))} />
        <MetricCard label="Mean abs error" value={formatPercent(optionalMetric(liveMetrics?.mean_abs_return_error))} />
      </section>

      <section className="dashboard-grid">
        <div className="panel chart-panel fade-rise-delay">
          <div className="panel-heading">
            <div>
              <div className="panel-kicker">BTCUSDT 5m Kline</div>
              <h2>5-minute chart</h2>
            </div>
            <div className="latest-price-pill">
              Latest ${latestPrice?.toLocaleString('en-US', { maximumFractionDigits: 1 }) ?? '--'}
            </div>
          </div>
          {klines.length > 0 ? (
            <KlineChart klines={klines} signals={recentSignals.slice(0, 10)} latestPrice={latestPrice} />
          ) : (
            <div className="empty-panel">No chart data.</div>
          )}
        </div>

        <PredictionPanel signal={latestSignal} />
      </section>

      <section className="panel fade-rise-delay-2">
        <div className="panel-heading">
          <div>
            <div className="panel-kicker">Live records</div>
            <h2>Latest 30 live predictions</h2>
          </div>
        </div>
        <SignalTable records={recentSignals} compact />
      </section>
    </div>
  );
}
