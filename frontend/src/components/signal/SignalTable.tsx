import type { SignalRecord } from '../../types/market';
import { formatPercent, formatPrice, hitStatusLabel } from '../../utils/format';
import { SignalBadge } from './SignalBadge';

interface SignalTableProps {
  records: SignalRecord[];
  compact?: boolean;
}

function optionalNumber(value?: number | null) {
  return value === null ? undefined : value;
}

function settlementLabel(record: SignalRecord) {
  return record.settlement_status ?? record.hitStatus ?? 'PENDING';
}

function signalHitLabel(record: SignalRecord) {
  if (record.signal_hit === true) return 'HIT';
  if (record.signal_hit === false) return 'MISS';
  return hitStatusLabel(record.hitStatus);
}

function actualReturn(record: SignalRecord) {
  return record.actual_future_max_return_30m ?? record.actualFutureMaxReturn;
}

export function SignalTable({ records, compact = false }: SignalTableProps) {
  if (records.length === 0) {
    return (
      <div className="table-empty">
        <span>No signal records.</span>
      </div>
    );
  }

  return (
    <div className={`table-shell ${compact ? 'table-compact' : ''}`}>
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Signal</th>
            <th>Price</th>
            <th>Prob</th>
            <th>Pred return</th>
            {!compact && <th>Pred high</th>}
            <th>Status</th>
            <th>Signal hit</th>
            <th>Actual max</th>
            <th>Abs error</th>
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
              {!compact && <td>${formatPrice(record.predHighPrice)}</td>}
              <td>
                <span className={`hit-status hit-${settlementLabel(record)}`}>
                  {settlementLabel(record)}
                </span>
              </td>
              <td>
                <span className={`hit-status hit-${record.signal_hit === true ? 'HIT' : record.signal_hit === false ? 'MISS' : 'PENDING'}`}>
                  {signalHitLabel(record)}
                </span>
              </td>
              <td>{formatPercent(optionalNumber(actualReturn(record)))}</td>
              <td>{formatPercent(optionalNumber(record.return_abs_error))}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
