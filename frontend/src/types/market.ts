export type SignalType = 'BUY' | 'WATCH' | 'NO_BUY';

export type HitStatus = 'HIT' | 'MISS' | 'PENDING';
export type SettlementStatus = 'PENDING' | 'SETTLED';

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
  settlement_status?: SettlementStatus;
  actual_future_max_return_30m?: number | null;
  actual_y_buy?: boolean | null;
  classification_hit?: boolean | null;
  signal_hit?: boolean | null;
  return_abs_error?: number | null;
}

export interface LiveMetrics {
  settled_count: number;
  pending_count: number;
  classification_hit_rate: number | null;
  signal_hit_rate: number | null;
  buy_signal_hit_rate: number | null;
  watch_signal_hit_rate: number | null;
  no_buy_correct_rate: number | null;
  mean_return_error: number | null;
  mean_abs_return_error: number | null;
  last_30_signal_hit_rate: number | null;
  last_100_signal_hit_rate: number | null;
  last_500_signal_hit_rate: number | null;
  last_7d_signal_hit_rate: number | null;
  last_30d_signal_hit_rate: number | null;
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
