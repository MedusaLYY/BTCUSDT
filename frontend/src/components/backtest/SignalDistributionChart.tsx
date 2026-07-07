import { useMemo } from 'react';
import type { EChartsOption } from 'echarts';
import type { SignalDistributionPoint } from '../../types/market';
import { signalLabel } from '../../utils/format';
import { EChart } from '../market/EChart';

interface SignalDistributionChartProps {
  data: SignalDistributionPoint[];
}

const colors = {
  BUY: '#16A34A',
  WATCH: '#D97706',
  NO_BUY: '#DC2626',
};

export function SignalDistributionChart({ data }: SignalDistributionChartProps) {
  const option = useMemo<EChartsOption>(() => ({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      borderColor: 'rgba(0,0,0,0.08)',
      backgroundColor: 'rgba(255,255,255,0.96)',
      textStyle: { color: '#111827' },
    },
    legend: {
      bottom: 0,
      textStyle: { color: '#6F6F6F' },
    },
    series: [
      {
        name: '信号',
        type: 'pie',
        radius: ['52%', '76%'],
        center: ['50%', '45%'],
        avoidLabelOverlap: true,
        label: {
          color: '#111827',
          formatter: '{b}\n{d}%',
        },
        data: data.map((item) => ({
          name: signalLabel(item.signal),
          value: item.count,
          itemStyle: { color: colors[item.signal] },
        })),
      },
    ],
  }), [data]);

  return <EChart option={option} className="chart-host chart-host-medium" />;
}
