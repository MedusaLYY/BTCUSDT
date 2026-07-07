import type { SignalRecord } from '../../types/market';
import { formatPercent, formatPrice, hitStatusLabel } from '../../utils/format';
import { SignalBadge } from './SignalBadge';

interface SignalTableProps {
  records: SignalRecord[];
  compact?: boolean;
}

export function SignalTable({ records, compact = false }: SignalTableProps) {
  if (records.length === 0) {
    return (
      <div className="table-empty">
        <span>暂无信号记录。</span>
      </div>
    );
  }

  return (
    <div className={`table-shell ${compact ? 'table-compact' : ''}`}>
      <table>
        <thead>
          <tr>
            <th>时间</th>
            <th>信号</th>
            <th>价格</th>
            <th>概率</th>
            <th>预测收益</th>
            <th>预测最高价</th>
            {!compact && <th>实际最大收益</th>}
            {!compact && <th>预测兑现状态</th>}
          </tr>
        </thead>
        <tbody>
          {records.map((record) => (
            <tr key={record.id}>
              <td className="mono">{record.openTime}</td>
              <td>
                <SignalBadge signal={record.signal} size="sm" />
              </td>
              <td>${formatPrice(record.currentPrice)}</td>
              <td>{formatPercent(record.buyProbability, 0)}</td>
              <td>{formatPercent(record.predReturn)}</td>
              <td>${formatPrice(record.predHighPrice)}</td>
              {!compact && (
                <td>{record.actualFutureMaxReturn === undefined ? '待确认' : formatPercent(record.actualFutureMaxReturn)}</td>
              )}
              {!compact && (
                <td>
                  <span className={`hit-status hit-${record.hitStatus ?? 'PENDING'}`}>
                    {hitStatusLabel(record.hitStatus)}
                  </span>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
