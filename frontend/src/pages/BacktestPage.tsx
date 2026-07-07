import { useCallback, useEffect, useState } from 'react';
import { BacktestSummary as BacktestSummaryPanel } from '../components/backtest/BacktestSummary';
import { EquityCurveChart } from '../components/backtest/EquityCurveChart';
import { PrecisionThresholdChart } from '../components/backtest/PrecisionThresholdChart';
import { SignalDistributionChart } from '../components/backtest/SignalDistributionChart';
import {
  getBacktestSummary,
  getEquityCurve,
  getPrecisionThreshold,
  getSignalDistribution,
} from '../services/api';
import type {
  BacktestSummary,
  EquityPoint,
  PrecisionThresholdPoint,
  SignalDistributionPoint,
} from '../types/market';

export function BacktestPage() {
  const [summary, setSummary] = useState<BacktestSummary>();
  const [equity, setEquity] = useState<EquityPoint[]>([]);
  const [distribution, setDistribution] = useState<SignalDistributionPoint[]>([]);
  const [precision, setPrecision] = useState<PrecisionThresholdPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();

  const loadBacktest = useCallback(async () => {
    setLoading(true);
    const [summaryResult, equityResult, distributionResult, precisionResult] = await Promise.all([
      getBacktestSummary(),
      getEquityCurve(),
      getSignalDistribution(),
      getPrecisionThreshold(),
    ]);

    setSummary((current) => (summaryResult.fromMock && current ? current : summaryResult.data));
    setEquity((current) => (equityResult.fromMock && current.length > 0 ? current : equityResult.data));
    setDistribution((current) => (distributionResult.fromMock && current.length > 0 ? current : distributionResult.data));
    setPrecision((current) => (precisionResult.fromMock && current.length > 0 ? current : precisionResult.data));
    setError(summaryResult.error ?? equityResult.error ?? distributionResult.error ?? precisionResult.error);
    setLoading(false);
  }, []);

  useEffect(() => {
    void loadBacktest();
  }, [loadBacktest]);

  return (
    <div className="page-stack" id="research-note">
      <div className="page-actions">
        <button type="button" className="terminal-button" onClick={loadBacktest} disabled={loading}>
          {loading ? '刷新中...' : '手动刷新'}
        </button>
      </div>

      {error && <div className="status-banner">{error}</div>}

      <div className="risk-banner">
        当前报告用于研究和工程验证。测试集回测不能用于继续调参，也不代表未来实盘收益；当前系统不应直接作为实盘交易建议。
      </div>

      <BacktestSummaryPanel summary={summary} />

      <section className="backtest-grid">
        <div className="panel fade-rise-delay">
          <div className="panel-heading">
            <div>
              <div className="panel-kicker">回测权益</div>
              <h2>权益曲线</h2>
            </div>
          </div>
          {equity.length > 0 ? <EquityCurveChart points={equity} /> : <div className="empty-panel">暂无权益数据。</div>}
        </div>

        <div className="panel fade-rise-delay">
          <div className="panel-heading">
            <div>
              <div className="panel-kicker">信号构成</div>
              <h2>分布</h2>
            </div>
          </div>
          {distribution.length > 0 ? (
            <SignalDistributionChart data={distribution} />
          ) : (
            <div className="empty-panel">暂无分布数据。</div>
          )}
        </div>

        <div className="panel wide-panel fade-rise-delay-2">
          <div className="panel-heading">
            <div>
              <div className="panel-kicker">验证阈值</div>
              <h2>阈值精度</h2>
            </div>
          </div>
          {precision.length > 0 ? (
            <PrecisionThresholdChart data={precision} />
          ) : (
            <div className="empty-panel">暂无阈值数据。</div>
          )}
        </div>
      </section>
    </div>
  );
}
