import { useCallback, useEffect, useMemo, useState } from 'react';
import { SignalBadge } from '../components/signal/SignalBadge';
import { SignalTable } from '../components/signal/SignalTable';
import { getSignals } from '../services/api';
import type { SignalRecord, SignalType } from '../types/market';
import { filterLabel } from '../utils/format';

type SignalFilter = 'ALL' | SignalType;

const filters: SignalFilter[] = ['ALL', 'BUY', 'WATCH', 'NO_BUY'];

export function SignalsPage() {
  const [records, setRecords] = useState<SignalRecord[]>([]);
  const [filter, setFilter] = useState<SignalFilter>('ALL');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();

  const loadSignals = useCallback(async () => {
    setLoading(true);
    const result = await getSignals();

    setRecords((current) => (result.fromMock && current.length > 0 ? current : result.data));
    setError(result.error);
    setLoading(false);
  }, []);

  useEffect(() => {
    void loadSignals();
  }, [loadSignals]);

  const filteredRecords = useMemo(() => {
    return records
      .filter((record) => filter === 'ALL' || record.signal === filter)
      .sort((a, b) => b.openTime.localeCompare(a.openTime));
  }, [filter, records]);

  const buyCount = useMemo(() => records.filter((record) => record.signal === 'BUY').length, [records]);

  return (
    <div className="page-stack">
      {error && <div className="status-banner">{error}</div>}

      <section className="panel fade-rise">
        <div className="panel-heading responsive-heading">
          <div>
            <div className="panel-kicker">历史信号</div>
            <h2>{filteredRecords.length} 条记录</h2>
          </div>
          <div className="chart-toolbar">
            <div className="segmented-control" aria-label="信号筛选">
              {filters.map((item) => (
                <button
                  key={item}
                  type="button"
                  className={filter === item ? 'active' : ''}
                  onClick={() => setFilter(item)}
                >
                  {filterLabel(item)}
                </button>
              ))}
            </div>
            <button type="button" className="terminal-button" onClick={loadSignals} disabled={loading}>
              {loading ? '刷新中...' : '刷新'}
            </button>
          </div>
        </div>

        <div className="conservative-note">
          当前策略偏保守，BUY 信号数量可能很少。当前记录中买入信号 {buyCount} 条；观察信号并不等于买入建议。
        </div>

        {filter !== 'ALL' && (
          <div className="filter-readout">
            当前显示 <SignalBadge signal={filter} size="sm" /> 记录，按最新时间排序。
          </div>
        )}

        <SignalTable records={filteredRecords} />
      </section>
    </div>
  );
}
