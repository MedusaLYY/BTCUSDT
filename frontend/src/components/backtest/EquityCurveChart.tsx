import { useMemo } from 'react';
import type { EChartsOption } from 'echarts';
import type { EquityPoint } from '../../types/market';
import { EChart } from '../market/EChart';

interface EquityCurveChartProps {
  points: EquityPoint[];
}

export function EquityCurveChart({ points }: EquityCurveChartProps) {
  const option = useMemo<EChartsOption>(() => ({
    backgroundColor: 'transparent',
    grid: { left: 58, right: 22, top: 32, bottom: 36 },
    tooltip: {
      trigger: 'axis',
      borderColor: 'rgba(0,0,0,0.08)',
      backgroundColor: 'rgba(255,255,255,0.96)',
      textStyle: { color: '#111827' },
    },
    xAxis: {
      type: 'category',
      data: points.map((item) => item.time),
      axisLine: { lineStyle: { color: 'rgba(0,0,0,0.14)' } },
      axisLabel: { color: '#6F6F6F' },
    },
    yAxis: {
      type: 'value',
      scale: true,
      axisLine: { show: false },
      axisLabel: { color: '#6F6F6F' },
      splitLine: { lineStyle: { color: 'rgba(0,0,0,0.06)' } },
    },
    series: [
      {
        name: '权益',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: points.map((item) => item.equity),
        lineStyle: { color: '#16A34A', width: 2 },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(22, 163, 74, 0.18)' },
              { offset: 1, color: 'rgba(22, 163, 74, 0.02)' },
            ],
          },
        },
      },
    ],
  }), [points]);

  return <EChart option={option} className="chart-host chart-host-medium" />;
}
