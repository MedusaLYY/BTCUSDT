import type { PredictionSignal } from '../../types/market';
import { formatPrice, formatSignedPercent } from '../../utils/format';
import { SignalBadge } from '../signal/SignalBadge';

interface PriceHeaderProps {
  signal?: PredictionSignal;
  change24h?: number;
  updatedAt?: string;
  fromMock?: boolean;
}

export function PriceHeader({ signal, change24h = 0, updatedAt, fromMock = false }: PriceHeaderProps) {
  const updatedTime = updatedAt ? new Date(updatedAt).toLocaleTimeString() : '--';

  return (
    <section className="price-header fade-rise-delay">
      <article className="panel price-metric-card lift-on-hover">
        <span>当前价格</span>
        <strong>${formatPrice(signal?.currentPrice)}</strong>
        <small>BTCUSDT / 5m</small>
        {fromMock && <span className="mock-chip">模拟数据</span>}
      </article>

      <article className="panel price-metric-card lift-on-hover">
        <span>24小时涨跌</span>
        <strong className={change24h >= 0 ? 'positive' : 'negative'}>{formatSignedPercent(change24h)}</strong>
        <small>基于最近 5 分钟K线估算</small>
      </article>

      <article className="panel price-metric-card lift-on-hover">
        <span>当前K线时间</span>
        <strong>{signal?.openTime ?? '--'}</strong>
        <small>仅使用已收盘K线推理</small>
      </article>

      <article className="panel price-metric-card lift-on-hover">
        <span>最新信号</span>
        {signal ? <SignalBadge signal={signal.signal} size="lg" /> : <strong>--</strong>}
        <small>最后更新：{updatedTime}</small>
      </article>
    </section>
  );
}
