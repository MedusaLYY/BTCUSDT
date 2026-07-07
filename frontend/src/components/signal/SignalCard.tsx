import type { PredictionSignal } from '../../types/market';
import { formatPercent, formatPrice } from '../../utils/format';
import { SignalBadge } from './SignalBadge';

interface SignalCardProps {
  signal?: PredictionSignal;
}

export function SignalCard({ signal }: SignalCardProps) {
  if (!signal) {
    return (
      <section className="panel signal-card empty-panel">
        <p>暂无最新信号。</p>
      </section>
    );
  }

  return (
    <section className="panel signal-card">
      <div className="panel-kicker">最新30分钟预测</div>
      <div className="signal-card-main">
        <SignalBadge signal={signal.signal} size="lg" />
        <div>
          <div className="signal-price">${formatPrice(signal.currentPrice)}</div>
          <div className="muted">{signal.symbol} / {signal.interval} / {signal.openTime}</div>
        </div>
      </div>
      <div className="signal-card-grid">
        <div>
          <span>买入概率</span>
          <strong>{formatPercent(signal.buyProbability, 0)}</strong>
        </div>
        <div>
          <span>预测收益</span>
          <strong>{formatPercent(signal.predReturn)}</strong>
        </div>
        <div>
          <span>预测最高价</span>
          <strong>${formatPrice(signal.predHighPrice)}</strong>
        </div>
      </div>
    </section>
  );
}
