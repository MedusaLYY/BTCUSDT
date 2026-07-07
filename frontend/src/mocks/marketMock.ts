import type {
  BacktestSummary,
  EquityPoint,
  Kline,
  PrecisionThresholdPoint,
  PredictionSignal,
  SignalDistributionPoint,
  SignalRecord,
  SignalType,
} from '../types/market';

const baseTime = new Date('2026-07-06T16:00:00+08:00').getTime();
const fiveMinutes = 5 * 60 * 1000;

function formatDate(timestamp: number) {
  const date = new Date(timestamp);
  const pad = (value: number) => String(value).padStart(2, '0');

  return [
    date.getFullYear(),
    '-',
    pad(date.getMonth() + 1),
    '-',
    pad(date.getDate()),
    ' ',
    pad(date.getHours()),
    ':',
    pad(date.getMinutes()),
  ].join('');
}

function round(value: number, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function wave(index: number) {
  return Math.sin(index / 7) * 28 + Math.sin(index / 19) * 56 + Math.cos(index / 31) * 44;
}

export function generateMockKlines(limit = 520): Kline[] {
  const rows: Kline[] = [];
  let previousClose = 63000;

  for (let index = 0; index < limit; index += 1) {
    const timestamp = baseTime - (limit - index - 1) * fiveMinutes;
    const drift = wave(index) + Math.sin(index * 1.7) * 15;
    const open = previousClose;
    const close = 63000 + drift + Math.sin(index / 4) * 18;
    const spread = 22 + Math.abs(Math.sin(index / 5)) * 38;
    const high = Math.max(open, close) + spread;
    const low = Math.min(open, close) - spread * 0.78;
    const volume = 88 + Math.abs(Math.sin(index / 9)) * 96 + (index % 17) * 2.4;

    rows.push({
      openTime: formatDate(timestamp),
      open: round(open, 1),
      high: round(high, 1),
      low: round(low, 1),
      close: round(close, 1),
      volume: round(volume, 2),
    });

    previousClose = close;
  }

  return rows;
}

function signalFromProbability(probability: number): SignalType {
  if (probability >= 0.64) return 'BUY';
  if (probability >= 0.55) return 'WATCH';
  return 'NO_BUY';
}

function signalReason(signal: SignalType, probability: number, predReturn: number) {
  if (signal === 'BUY') {
    return [
      `模型估计未来30分钟买入概率为 ${(probability * 100).toFixed(0)}%。`,
      '动量和短周期成交量扩张高于保守触发线。',
      `预测最大收益为 ${(predReturn * 100).toFixed(2)}%，高于买入阈值。`,
    ];
  }

  if (signal === 'WATCH') {
    return [
      `模型估计未来30分钟买入概率为 ${(probability * 100).toFixed(0)}%。`,
      '方向压力偏正，但尚未完全达到买入触发条件。',
      '当前保守策略保持观察，等待价格结构和成交量进一步确认。',
    ];
  }

  return [
    `模型估计未来30分钟买入概率为 ${(probability * 100).toFixed(0)}%。`,
    '当前形态未达到保守买入阈值。',
    '在概率或动量改善前，风控保持不买入状态。',
  ];
}

export const mockKlines = generateMockKlines(520);

export const mockSignalRecords: SignalRecord[] = mockKlines
  .slice(-240)
  .filter((_, index) => index % 3 === 0)
  .slice(-80)
  .map((kline, index, rows) => {
    const momentum = (kline.close - kline.open) / kline.open;
    const probability = Math.min(0.78, Math.max(0.38, 0.52 + momentum * 35 + Math.sin(index / 4) * 0.08));
    const signal = signalFromProbability(probability);
    const predReturn = Math.max(0.0004, probability * 0.0042 - 0.0014);
    const isPending = index > rows.length - 7;
    const actualFutureMaxReturn = isPending ? undefined : predReturn + Math.sin(index / 3) * 0.0018 - 0.0005;
    const hitStatus =
      actualFutureMaxReturn === undefined
        ? 'PENDING'
        : actualFutureMaxReturn >= predReturn * 0.72
          ? 'HIT'
          : 'MISS';

    return {
      id: `SIG-${String(index + 1).padStart(4, '0')}`,
      openTime: kline.openTime,
      currentPrice: kline.close,
      signal,
      buyProbability: round(probability, 4),
      predReturn: round(predReturn, 4),
      predHighPrice: round(kline.close * (1 + predReturn), 1),
      actualFutureMaxReturn: actualFutureMaxReturn === undefined ? undefined : round(actualFutureMaxReturn, 4),
      hitStatus,
    };
  });

const latestKline = mockKlines[mockKlines.length - 1];
const latestProbability = 0.58;
const latestPredReturn = 0.0021;

export const mockLatestSignal: PredictionSignal = {
  symbol: 'BTCUSDT',
  interval: '5m',
  openTime: latestKline.openTime,
  currentPrice: latestKline.close,
  signal: 'WATCH',
  buyProbability: latestProbability,
  predReturn: latestPredReturn,
  predHighPrice: round(latestKline.close * (1 + latestPredReturn), 1),
  reason: signalReason('WATCH', latestProbability, latestPredReturn),
};

export const mockBacktestSummary: BacktestSummary = {
  totalSignals: 1268,
  buySignals: 214,
  watchSignals: 432,
  winRate: 0.573,
  precision: 0.618,
  avgFutureMaxReturn: 0.0034,
  avgFutureMinReturn: -0.0018,
  avgSignalsPerDay: 17.2,
  maxDrawdown: -0.041,
};

export const mockEquityCurve: EquityPoint[] = Array.from({ length: 90 }, (_, index) => {
  const timestamp = new Date('2026-04-08T00:00:00+08:00').getTime() + index * 24 * 60 * 60 * 1000;
  const trend = index * 18;
  const pulse = Math.sin(index / 5) * 120 + Math.cos(index / 13) * 90;
  const drawdown = index > 48 && index < 59 ? (index - 48) * -32 : 0;

  return {
    time: formatDate(timestamp).slice(0, 10),
    equity: round(10000 + trend + pulse + drawdown, 2),
  };
});

export const mockSignalDistribution: SignalDistributionPoint[] = [
  { signal: 'BUY', count: mockBacktestSummary.buySignals },
  { signal: 'WATCH', count: mockBacktestSummary.watchSignals },
  {
    signal: 'NO_BUY',
    count: mockBacktestSummary.totalSignals - mockBacktestSummary.buySignals - mockBacktestSummary.watchSignals,
  },
];

export const mockPrecisionThreshold: PrecisionThresholdPoint[] = [
  { threshold: 0.5, precision: 0.53, signalCount: 698 },
  { threshold: 0.55, precision: 0.58, signalCount: 432 },
  { threshold: 0.6, precision: 0.61, signalCount: 286 },
  { threshold: 0.62, precision: 0.64, signalCount: 214 },
  { threshold: 0.65, precision: 0.67, signalCount: 146 },
  { threshold: 0.7, precision: 0.71, signalCount: 69 },
];
