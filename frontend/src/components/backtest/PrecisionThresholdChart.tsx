import { useMemo } from 'react';
import type { EChartsOption } from 'echarts';
import type { PrecisionThresholdPoint } from '../../types/market';
import { EChart } from '../market/EChart';

interface PrecisionThresholdChartProps {
  data: PrecisionThresholdPoint[];
}

export function PrecisionThresholdChart({ data }: PrecisionThresholdChartProps) {
  const option = useMemo<EChartsOption>(() => ({
    backgroundColor: 'transparent',
    grid: { left: 48, right: 48, top: 32, bottom: 42 },
    tooltip: {
      trigger: 'axis',
      borderColor: 'rgba(0,0,0,0.08)',
      backgroundColor: 'rgba(255,255,255,0.96)',
      textStyle: { color: '#111827' },
      formatter: (params: any) => {
        const precision = params.find((item: any) => item.seriesName === '精度');
        const count = params.find((item: any) => item.seriesName === '信号数量');

        return [
          `<strong>阈值 ${precision?.axisValue}</strong>`,
          `精度：${precision ? (precision.data * 100).toFixed(1) : '--'}%`,
          `信号数量：${count?.data ?? '--'}`,
        ].join('<br />');
      },
    },
    xAxis: {
      type: 'category',
      data: data.map((item) => item.threshold.toFixed(2)),
      axisLine: { lineStyle: { color: 'rgba(0,0,0,0.14)' } },
      axisLabel: { color: '#6F6F6F' },
    },
    yAxis: [
      {
        type: 'value',
        axisLabel: { color: '#6F6F6F', formatter: (value: number) => `${(value * 100).toFixed(0)}%` },
        splitLine: { lineStyle: { color: 'rgba(0,0,0,0.06)' } },
      },
      {
        type: 'value',
        axisLabel: { color: '#6F6F6F' },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: '精度',
        type: 'line',
        smooth: true,
        symbolSize: 8,
        data: data.map((item) => item.precision),
        lineStyle: { color: '#16A34A', width: 2 },
        itemStyle: { color: '#16A34A' },
      },
      {
        name: '信号数量',
        type: 'bar',
        yAxisIndex: 1,
        data: data.map((item) => item.signalCount),
        itemStyle: { color: 'rgba(37, 99, 235, 0.28)' },
      },
    ],
  }), [data]);

  return <EChart option={option} className="chart-host chart-host-medium" />;
}
