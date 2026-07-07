interface MetricCardProps {
  label: string;
  value: string;
  tone?: 'default' | 'positive' | 'warning' | 'negative';
  caption?: string;
}

export function MetricCard({ label, value, tone = 'default', caption }: MetricCardProps) {
  return (
    <article className={`metric-card metric-${tone} lift-on-hover`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {caption && <small>{caption}</small>}
    </article>
  );
}
