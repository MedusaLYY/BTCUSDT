interface ChartToolbarProps {
  chartType: 'kline' | 'line';
  autoRefresh: boolean;
  loading?: boolean;
  onChartTypeChange: (type: 'kline' | 'line') => void;
  onAutoRefreshChange: (enabled: boolean) => void;
  onRefresh: () => void;
}

export function ChartToolbar({
  chartType,
  autoRefresh,
  loading = false,
  onChartTypeChange,
  onAutoRefreshChange,
  onRefresh,
}: ChartToolbarProps) {
  return (
    <div className="chart-toolbar">
      <div className="segmented-control" aria-label="图表类型">
        <button type="button" className={chartType === 'kline' ? 'active' : ''} onClick={() => onChartTypeChange('kline')}>
          K线
        </button>
        <button type="button" className={chartType === 'line' ? 'active' : ''} onClick={() => onChartTypeChange('line')}>
          折线
        </button>
      </div>

      <label className="switch-control">
        <input
          type="checkbox"
          checked={autoRefresh}
          onChange={(event) => onAutoRefreshChange(event.target.checked)}
        />
        <span>30秒自动刷新</span>
      </label>

      <button type="button" className="terminal-button" onClick={onRefresh} disabled={loading}>
        {loading ? '刷新中...' : '刷新'}
      </button>
    </div>
  );
}
