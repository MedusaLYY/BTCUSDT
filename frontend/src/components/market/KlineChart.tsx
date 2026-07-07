import { useMemo } from 'react';
import type { EChartsOption } from 'echarts';
import type { Kline, SignalRecord } from '../../types/market';
import { formatPrice, signalLabel } from '../../utils/format';
import { EChart } from './EChart';

interface KlineChartProps {
  klines: Kline[];
  signals?: SignalRecord[];
  latestPrice?: number;
}

const signalColors = {
  BUY: '#16A34A',
  WATCH: '#D97706',
  NO_BUY: '#DC2626',
};

export function KlineChart({ klines, signals = [], latestPrice }: KlineChartProps) {
  const option = useMemo<EChartsOption>(() => {
    const times = klines.map((item) => item.openTime);
    const candles = klines.map((item) => [item.open, item.close, item.low, item.high]);
    const volume = klines.map((item) => item.volume);
    const markers = signals
      .filter((item) => item.signal !== 'NO_BUY')
      .map((item) => ({
        name: signalLabel(item.signal),
        coord: [item.openTime, item.currentPrice],
        value: signalLabel(item.signal),
        symbol: item.signal === 'BUY' ? 'pin' : 'diamond',
        symbolSize: item.signal === 'BUY' ? 42 : 18,
        itemStyle: {
          color: signalColors[item.signal],
          borderColor: '#ffffff',
          borderWidth: 2,
        },
        label: {
          color: '#ffffff',
          fontWeight: 700,
        },
      }));

    return {
      backgroundColor: 'transparent',
      animationDuration: 500,
      grid: [
        { left: 54, right: 74, top: 28, height: '67%' },
        { left: 54, right: 74, top: '78%', height: '14%' },
      ],
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        borderColor: 'rgba(0,0,0,0.08)',
        backgroundColor: 'rgba(255,255,255,0.96)',
        textStyle: { color: '#111827' },
        formatter: (params: any) => {
          const candle = params.find((item: any) => item.seriesType === 'candlestick');
          const vol = params.find((item: any) => item.seriesName === '成交量');

          if (!candle) return '';

          const [open, close, low, high] = candle.data;
          return [
            `<strong>${candle.axisValue}</strong>`,
            `开盘价：${formatPrice(open)}`,
            `最高价：${formatPrice(high)}`,
            `最低价：${formatPrice(low)}`,
            `收盘价：${formatPrice(close)}`,
            `成交量：${vol ? Number(vol.data).toFixed(2) : '--'}`,
          ].join('<br />');
        },
      },
      axisPointer: {
        link: [{ xAxisIndex: 'all' }],
      },
      xAxis: [
        {
          type: 'category',
          data: times,
          boundaryGap: true,
          axisLine: { lineStyle: { color: 'rgba(0,0,0,0.14)' } },
          axisLabel: { color: '#6F6F6F' },
          splitLine: { show: false },
          min: 'dataMin',
          max: 'dataMax',
        },
        {
          type: 'category',
          gridIndex: 1,
          data: times,
          boundaryGap: true,
          axisLine: { lineStyle: { color: 'rgba(0,0,0,0.14)' } },
          axisLabel: { show: false },
          splitLine: { show: false },
          min: 'dataMin',
          max: 'dataMax',
        },
      ],
      yAxis: [
        {
          scale: true,
          axisLine: { show: false },
          axisLabel: { color: '#6F6F6F' },
          splitLine: { lineStyle: { color: 'rgba(0,0,0,0.06)' } },
        },
        {
          scale: true,
          gridIndex: 1,
          axisLine: { show: false },
          axisLabel: { color: '#9A9A9A' },
          splitLine: { show: false },
        },
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0, 1],
          start: 48,
          end: 100,
        },
        {
          type: 'slider',
          xAxisIndex: [0, 1],
          bottom: 4,
          height: 18,
          borderColor: 'rgba(0,0,0,0.08)',
          fillerColor: 'rgba(22, 163, 74, 0.12)',
          handleStyle: { color: '#16A34A' },
          textStyle: { color: '#6F6F6F' },
        },
      ],
      series: [
        {
          name: 'BTCUSDT',
          type: 'candlestick',
          data: candles,
          itemStyle: {
            color: '#16A34A',
            color0: '#DC2626',
            borderColor: '#16A34A',
            borderColor0: '#DC2626',
          },
          markPoint: {
            data: markers,
          },
          markLine: latestPrice
            ? {
                silent: true,
                symbol: 'none',
                label: {
                  formatter: `最新 ${formatPrice(latestPrice)}`,
                  color: '#111827',
                  backgroundColor: '#ffffff',
                  borderColor: 'rgba(0,0,0,0.12)',
                  borderWidth: 1,
                  padding: [4, 6],
                },
                lineStyle: {
                  color: '#D97706',
                  type: 'dashed',
                  width: 1,
                },
                data: [{ yAxis: latestPrice }],
              }
            : undefined,
        },
        {
          name: '成交量',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: volume,
          itemStyle: {
            color: 'rgba(37, 99, 235, 0.22)',
          },
        },
      ],
    };
  }, [klines, latestPrice, signals]);

  return <EChart option={option} className="chart-host chart-host-large" />;
}
