interface HeaderProps {
  description: string;
  eyebrow: string;
  title: string;
}

export function Header({ eyebrow, title, description }: HeaderProps) {
  return (
    <header className="app-header fade-rise">
      <div>
        <span className="panel-kicker">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>

      <div className="header-meta">
        <span className="research-pill">未来30分钟预测周期</span>
        <span className="research-pill">研究系统，非投资建议</span>
      </div>
    </header>
  );
}
