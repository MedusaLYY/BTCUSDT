export type SignalType = 'BUY' | 'WATCH' | 'NO_BUY';

export type HitStatus = 'HIT' | 'MISS' | 'PENDING';

export interface Kline {
  openTime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionSignal {
  symbol: string;
  interval: string;
  openTime: string;
  currentPrice: number;
  signal: SignalType;
  buyProbability: number;
  predReturn: number;
  predHighPrice: number;
  reason: string[];
}

export interface SignalRecord {
  id: string;
  openTime: string;
  currentPrice: number;
  signal: SignalType;
  buyProbability: number;
  predReturn: number;
  predHighPrice: number;
  actualFutureMaxReturn?: number;
  hitStatus?: HitStatus;
}

export interface BacktestSummary {
  totalSignals: number;
  buySignals: number;
  watchSignals: number;
  winRate: number;
  precision: number;
  avgFutureMaxReturn: number;
  avgFutureMinReturn: number;
  avgSignalsPerDay: number;
  maxDrawdown: number;
}

export interface EquityPoint {
  time: string;
  equity: number;
}

export interface SignalDistributionPoint {
  signal: SignalType;
  count: number;
}

export interface PrecisionThresholdPoint {
  threshold: number;
  precision: number;
  signalCount: number;
}

export interface ApiResult<T> {
  data: T;
  fromMock: boolean;
  error?: string;
  updatedAt: string;
}
