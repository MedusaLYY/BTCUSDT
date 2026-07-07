import { useCallback, useEffect, useMemo, useState } from 'react';
import { KlineChart } from '../components/market/KlineChart';
import { PriceHeader } from '../components/market/PriceHeader';
import { PredictionPanel } from '../components/signal/PredictionPanel';
import { SignalTable } from '../components/signal/SignalTable';
import { getKlines, getLatestSignal, getSignals } from '../services/api';
import type { Kline, PredictionSignal, SignalRecord } from '../types/market';

function calculateChange24h(klines: Kline[]) {
  if (klines.length < 289) return 0;

  const latest = klines[klines.length - 1];
  const previous = klines[klines.length - 289];

  return (latest.close - previous.close) / previous.close;
}

export function DashboardPage() {
  const [klines, setKlines] = useState<Kline[]>([]);
  const [latestSignal, setLatestSignal] = useState<PredictionSignal>();
  const [signals, setSignals] = useState<SignalRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [fromMock, setFromMock] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<string>();

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    const [klineResult, signalResult, recordsResult] = await Promise.all([
      getKlines('BTCUSDT', '5m', 500),
      getLatestSignal(),
      getSignals(),
    ]);

    setKlines((current) => (klineResult.fromMock && current.length > 0 ? current : klineResult.data));
    setLatestSignal((current) => (signalResult.fromMock && current ? current : signalResult.data));
    setSignals((current) => (recordsResult.fromMock && current.length > 0 ? current : recordsResult.data));
    setFromMock(klineResult.fromMock || signalResult.fromMock || recordsResult.fromMock);
    setError(klineResult.error ?? signalResult.error ?? recordsResult.error);
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
    () => signals.slice().sort((a, b) => b.openTime.localeCompare(a.openTime)).slice(0, 10),
    [signals],
  );

  return (
    <div className="page-stack">
      <div className="dashboard-actions">
        <button type="button" className="terminal-button" onClick={loadDashboard} disabled={loading}>
          {loading ? '刷新中...' : '手动刷新'}
        </button>
      </div>

      {error && <div className="status-banner">{error}</div>}

      <PriceHeader signal={latestSignal} change24h={change24h} updatedAt={updatedAt} fromMock={fromMock} />

      <section className="dashboard-grid">
        <div className="panel chart-panel fade-rise-delay">
          <div className="panel-heading">
            <div>
              <div className="panel-kicker">BTCUSDT 5m K线</div>
              <h2>5分钟K线图</h2>
            </div>
            <div className="latest-price-pill">最新 ${latestPrice?.toLocaleString('en-US', { maximumFractionDigits: 1 }) ?? '--'}</div>
          </div>
          {klines.length > 0 ? (
            <KlineChart klines={klines} signals={recentSignals} latestPrice={latestPrice} />
          ) : (
            <div className="empty-panel">暂无图表数据。</div>
          )}
        </div>

        <PredictionPanel signal={latestSignal} />
      </section>

      <section className="panel fade-rise-delay-2">
        <div className="panel-heading">
          <div>
            <div className="panel-kicker">近期记录</div>
            <h2>最新10条信号记录</h2>
          </div>
        </div>
        <SignalTable records={recentSignals} compact />
      </section>
    </div>
  );
}
