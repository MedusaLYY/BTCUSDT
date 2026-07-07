import type { HitStatus, SignalType } from '../types/market';

export function formatPrice(value?: number) {
  if (value === undefined || Number.isNaN(value)) return '--';

  return value.toLocaleString('en-US', {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
}

export function formatPercent(value?: number, digits = 2) {
  if (value === undefined || Number.isNaN(value)) return '--';

  return `${(value * 100).toFixed(digits)}%`;
}

export function formatSignedPercent(value?: number, digits = 2) {
  if (value === undefined || Number.isNaN(value)) return '--';

  const sign = value > 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(digits)}%`;
}

export function formatInteger(value?: number) {
  if (value === undefined || Number.isNaN(value)) return '--';

  return value.toLocaleString('en-US', {
    maximumFractionDigits: 0,
  });
}

export function signalLabel(signal: SignalType) {
  if (signal === 'BUY') return '买入';
  if (signal === 'WATCH') return '观察';
  return '不买入';
}

export function signalCode(signal: SignalType) {
  if (signal === 'BUY') return 'BUY';
  if (signal === 'WATCH') return 'WATCH';
  return 'NO_BUY';
}

export function hitStatusLabel(status?: HitStatus) {
  if (!status || status === 'PENDING') return '待确认';
  if (status === 'HIT') return '已兑现';
  return '未兑现';
}

export function filterLabel(value: 'ALL' | SignalType) {
  return value === 'ALL' ? '全部' : signalLabel(value);
}

export function translateReason(reason: string) {
  const trimmed = reason.trim();
  const watchMatch = trimmed.match(/^probability ([\d.]+) > watch threshold ([\d.]+)$/);
  if (watchMatch) {
    return `买入概率 ${formatPercent(Number(watchMatch[1]))} 高于观察阈值 ${formatPercent(Number(watchMatch[2]), 0)}，但尚未满足完整 BUY 触发条件。`;
  }

  const noBuyMatch = trimmed.match(/^probability ([\d.]+) <= watch threshold ([\d.]+)$/);
  if (noBuyMatch) {
    return `买入概率 ${formatPercent(Number(noBuyMatch[1]))} 未高于观察阈值 ${formatPercent(Number(noBuyMatch[2]), 0)}。`;
  }

  const buyThresholdMatch = trimmed.match(/^probability ([\d.]+) > ([\d.]+)$/);
  if (buyThresholdMatch) {
    return `买入概率 ${formatPercent(Number(buyThresholdMatch[1]))} 高于买入阈值 ${formatPercent(Number(buyThresholdMatch[2]), 0)}。`;
  }

  const knownReasons: Record<string, string> = {
    'close > ma_20': '收盘价高于 MA20。',
    'volume_ratio_20 > 1.1': '20周期成交量比率高于 1.1。',
    'rsi_14 < 75': 'RSI14 低于 75，短线未处于极端过热区。',
  };

  if (knownReasons[trimmed]) {
    return knownReasons[trimmed];
  }

  return trimmed;
}
