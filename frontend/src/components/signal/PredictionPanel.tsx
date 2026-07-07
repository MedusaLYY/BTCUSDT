import type { CSSProperties } from 'react';
import type { PredictionSignal } from '../../types/market';
import { formatPercent, formatPrice, signalLabel, translateReason } from '../../utils/format';
import { SignalBadge } from './SignalBadge';

interface PredictionPanelProps {
  signal?: PredictionSignal;
}

function conservativeRuleStatus(signal: PredictionSignal) {
  if (signal.signal === 'BUY') {
    return [
      { label: '买入概率超过买入阈值', value: '通过' },
      { label: '收盘价高于 MA20', value: '通过' },
      { label: '成交量比率高于 1.1', value: '通过' },
      { label: 'RSI14 低于 75', value: '通过' },
    ];
  }

  return [
    { label: '买入概率超过观察阈值', value: signal.signal === 'WATCH' ? '通过' : '未通过' },
    { label: '升级 BUY 所需的均线/量能/RSI 明细', value: '接口未返回' },
    { label: '当前策略判断', value: `${signalLabel(signal.signal)}，保持保守` },
  ];
}

export function PredictionPanel({ signal }: PredictionPanelProps) {
  if (!signal) {
    return (
      <section className="panel prediction-panel empty-panel fade-rise-delay-2">
        <p>暂无预测结果。</p>
      </section>
    );
  }

  return (
    <section className="panel prediction-panel fade-rise-delay-2">
      <div className="panel-heading">
        <div>
          <div className="panel-kicker">信号引擎</div>
          <h2>未来30分钟</h2>
        </div>
        <SignalBadge signal={signal.signal} />
      </div>

      <div className="prediction-meter" style={{ '--meter-value': `${signal.buyProbability * 100}%` } as CSSProperties}>
        <span>买入概率</span>
        <strong>{formatPercent(signal.buyProbability, 0)}</strong>
      </div>

      <div className="prediction-stats">
        <div>
          <span>预测最大收益</span>
          <strong>{formatPercent(signal.predReturn)}</strong>
        </div>
        <div>
          <span>预测最高价</span>
          <strong>${formatPrice(signal.predHighPrice)}</strong>
        </div>
      </div>

      <div className="reason-box">
        <span className="section-label">信号原因</span>
        <div className="reason-list">
          {signal.reason.map((item) => (
            <p key={item}>{translateReason(item)}</p>
          ))}
        </div>
      </div>

      <div className="reason-box">
        <span className="section-label">未升级为买入原因</span>
        <div className="rule-check-list">
          {conservativeRuleStatus(signal).map((item) => (
            <div className="rule-check-item" key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
