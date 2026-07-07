import { useMemo } from 'react';
import type { EChartsOption } from 'echarts';
import type { Kline, SignalRecord } from '../../types/market';
import { formatPrice, signalLabel } from '../../utils/format';
import { EChart } from './EChart';

interface LinePriceChartProps {
  klines: Kline[];
  signals?: SignalRecord[];
  latestPrice?: number;
}

const markerColor = {
  BUY: '#16A34A',
  WATCH: '#D97706',
  NO_BUY: '#DC2626',
};

export function LinePriceChart({ klines, signals = [], latestPrice }: LinePriceChartProps) {
  const option = useMemo<EChartsOption>(() => {
    const times = klines.map((item) => item.openTime);
    const closes = klines.map((item) => item.close);
    const markers = signals
      .filter((item) => item.signal !== 'NO_BUY')
      .map((item) => ({
        name: signalLabel(item.signal),
        value: [item.openTime, item.currentPrice, signalLabel(item.signal)],
        itemStyle: {
          color: markerColor[item.signal],
          borderColor: '#ffffff',
          borderWidth: 2,
        },
      }));

    return {
      backgroundColor: 'transparent',
      animationDuration: 500,
      grid: { left: 54, right: 74, top: 30, bottom: 44 },
      tooltip: {
        trigger: 'axis',
        borderColor: 'rgba(0,0,0,0.08)',
        backgroundColor: 'rgba(255,255,255,0.96)',
        textStyle: { color: '#111827' },
        formatter: (params: any) => {
          const price = params.find((item: any) => item.seriesName === '收盘价');
          if (!price) return '';
          return `<strong>${price.axisValue}</strong><br />收盘价：${formatPrice(price.data)}`;
        },
      },
      xAxis: {
        type: 'category',
        data: times,
        boundaryGap: false,
        axisLine: { lineStyle: { color: 'rgba(0,0,0,0.14)' } },
        axisLabel: { color: '#6F6F6F' },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLine: { show: false },
        axisLabel: { color: '#6F6F6F' },
        splitLine: { lineStyle: { color: 'rgba(0,0,0,0.06)' } },
      },
      dataZoom: [
        { type: 'inside', start: 48, end: 100 },
        {
          type: 'slider',
          bottom: 4,
          height: 18,
          borderColor: 'rgba(0,0,0,0.08)',
          fillerColor: 'rgba(37, 99, 235, 0.1)',
          handleStyle: { color: '#2563EB' },
          textStyle: { color: '#6F6F6F' },
        },
      ],
      series: [
        {
          name: '收盘价',
          type: 'line',
          data: closes,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            width: 2,
            color: '#2563EB',
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(37, 99, 235, 0.18)' },
                { offset: 1, color: 'rgba(37, 99, 235, 0.01)' },
              ],
            },
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
                },
                data: [{ yAxis: latestPrice }],
              }
            : undefined,
        },
        {
          name: '信号',
          type: 'scatter',
          data: markers,
          symbolSize: 13,
          encode: { x: 0, y: 1 },
          tooltip: {
            formatter: (params: any) => `${params.data.value[2]}<br />${params.data.value[0]}<br />${formatPrice(params.data.value[1])}`,
          },
        },
      ],
    };
  }, [klines, latestPrice, signals]);

  return <EChart option={option} className="chart-host chart-host-large" />;
}
