import {
  mockBacktestSummary,
  mockEquityCurve,
  mockKlines,
  mockLatestSignal,
  mockLiveMetrics,
  mockPrecisionThreshold,
  mockSignalDistribution,
  mockSignalRecords,
} from '../mocks/marketMock';
import type {
  ApiResult,
  BacktestSummary,
  EquityPoint,
  Kline,
  LiveMetrics,
  PrecisionThresholdPoint,
  PredictionSignal,
  SignalDistributionPoint,
  SignalRecord,
} from '../types/market';

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? '';

function buildUrl(path: string) {
  return `${apiBaseUrl}${path}`;
}

function nowIso() {
  return new Date().toISOString();
}

async function requestJson<T>(path: string): Promise<T> {
  const response = await fetch(buildUrl(path), {
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.json() as Promise<T>;
}

async function withMockFallback<T>(path: string, fallback: T): Promise<ApiResult<T>> {
  try {
    const data = await requestJson<T>(path);

    return {
      data,
      fromMock: false,
      updatedAt: nowIso(),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : '未知 API 错误';

    return {
      data: fallback,
      fromMock: true,
      error: `后端暂不可用，当前展示模拟数据。${message}`,
      updatedAt: nowIso(),
    };
  }
}

export function getKlines(symbol = 'BTCUSDT', interval = '5m', limit = 500) {
  const fallback = mockKlines.slice(-limit);
  const query = new URLSearchParams({
    symbol,
    interval,
    limit: String(limit),
  });

  return withMockFallback<Kline[]>(`/api/klines?${query.toString()}`, fallback);
}

export function getLatestSignal() {
  return withMockFallback<PredictionSignal>('/api/latest-signal', mockLatestSignal);
}

export function getSignals() {
  return withMockFallback<SignalRecord[]>('/api/signals', mockSignalRecords);
}

export function getLivePredictions(limit = 30) {
  const query = new URLSearchParams({ limit: String(limit) });
  return withMockFallback<SignalRecord[]>(
    `/api/live/predictions?${query.toString()}`,
    mockSignalRecords.slice(-limit),
  );
}

export function getLiveMetrics() {
  return withMockFallback<LiveMetrics>('/api/live/metrics', mockLiveMetrics);
}

export function getBacktestSummary() {
  return withMockFallback<BacktestSummary>('/api/backtest/summary', mockBacktestSummary);
}

export function getEquityCurve() {
  return withMockFallback<EquityPoint[]>('/api/backtest/equity-curve', mockEquityCurve);
}

export function getSignalDistribution() {
  return withMockFallback<SignalDistributionPoint[]>(
    '/api/backtest/signal-distribution',
    mockSignalDistribution,
  );
}

export function getPrecisionThreshold() {
  return withMockFallback<PrecisionThresholdPoint[]>(
    '/api/backtest/precision-threshold',
    mockPrecisionThreshold,
  );
}
