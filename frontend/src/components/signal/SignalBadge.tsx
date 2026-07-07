import type { SignalType } from '../../types/market';
import { signalLabel } from '../../utils/format';

interface SignalBadgeProps {
  signal: SignalType;
  size?: 'sm' | 'md' | 'lg';
}

export function SignalBadge({ signal, size = 'md' }: SignalBadgeProps) {
  return <span className={`signal-badge signal-${signal.toLowerCase()} signal-${size}`}>{signalLabel(signal)}</span>;
}
