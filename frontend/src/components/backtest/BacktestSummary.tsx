import type { BacktestSummary as BacktestSummaryData } from '../../types/market';
import { formatInteger, formatPercent } from '../../utils/format';
import { MetricCard } from './MetricCard';

interface BacktestSummaryProps {
  summary?: BacktestSummaryData;
}

export function BacktestSummary({ summary }: BacktestSummaryProps) {
  if (!summary) {
    return (
      <div className="metric-grid fade-rise">
        <MetricCard label="回测" value="暂无数据" caption="后端或模拟数据返回后会显示指标。" />
      </div>
    );
  }

  return (
    <div className="metric-grid fade-rise">
      <MetricCard label="总信号数" value={formatInteger(summary.totalSignals)} caption="测试集全部模型信号记录" />
      <MetricCard label="买入信号数" value={formatInteger(summary.buySignals)} tone="positive" />
      <MetricCard label="观察信号数" value={formatInteger(summary.watchSignals)} tone="warning" />
      <MetricCard label="胜率" value={formatPercent(summary.winRate)} tone="positive" />
      <MetricCard label="精度" value={formatPercent(summary.precision)} tone="positive" caption="仅评估触发信号" />
      <MetricCard label="平均未来最大收益" value={formatPercent(summary.avgFutureMaxReturn)} tone="positive" />
      <MetricCard label="平均未来最小收益" value={formatPercent(summary.avgFutureMinReturn)} tone="negative" />
      <MetricCard label="日均信号数" value={summary.avgSignalsPerDay.toFixed(1)} />
      <MetricCard label="最大回撤" value={formatPercent(summary.maxDrawdown)} tone="negative" />
    </div>
  );
}
