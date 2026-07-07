import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { Header } from './Header';

const navItems = [
  { to: '/dashboard', label: '看板' },
  { to: '/chart', label: '图表' },
  { to: '/signals', label: '信号' },
  { to: '/backtest', label: '回测' },
];

const pageMeta: Record<string, { title: string; description: string; eyebrow: string }> = {
  '/': {
    eyebrow: 'BTCUSDT / 5m Research Terminal',
    title: 'BTCUSDT 5分钟K线短期预测系统',
    description: '实时展示当前K线、模型预测信号、历史信号记录和回测结果。该系统用于研究和工程验证，不构成投资建议。',
  },
  '/dashboard': {
    eyebrow: 'BTCUSDT / 5m Research Terminal',
    title: 'BTCUSDT 5分钟K线短期预测系统',
    description: '实时展示当前K线、模型预测信号、历史信号记录和回测结果。该系统用于研究和工程验证，不构成投资建议。',
  },
  '/chart': {
    eyebrow: 'Market Structure',
    title: '5分钟K线图与信号标记',
    description: '查看 BTCUSDT 的短周期市场结构，叠加观察和买入信号，辅助检查模型输出所在的价格位置。',
  },
  '/signals': {
    eyebrow: 'Signal Records',
    title: '历史信号记录',
    description: '按时间查看模型输出、预测收益、预测最高价和后验预测兑现状态。BUY 很少出现是当前保守策略的预期表现。',
  },
  '/backtest': {
    eyebrow: 'Backtest Research',
    title: '回测与阈值研究',
    description: '查看测试集回测、信号分布、阈值精度和风险指标。回测结果只用于研究，不代表未来收益。',
  },
};

export function AppLayout() {
  const location = useLocation();
  const meta = pageMeta[location.pathname] ?? pageMeta['/'];

  const refreshPage = () => {
    window.location.reload();
  };

  return (
    <div className="terminal-shell">
      <header className="top-nav-shell">
        <div className="top-nav">
          <Link className="brand-block" to="/dashboard" aria-label="返回预测看板">
            <span className="brand-mark">B</span>
            <div>
              <strong>BTC Terminal</strong>
              <span>5分钟预测终端</span>
            </div>
          </Link>

          <nav className="nav-list" aria-label="主导航">
            {navItems.map((item) => (
              <NavLink
                className={({ isActive }) => (isActive || (item.to === '/dashboard' && location.pathname === '/') ? 'active' : '')}
                key={item.to}
                to={item.to}
              >
                {item.label}
              </NavLink>
            ))}
            <a href="/backtest#research-note">研究</a>
          </nav>

          <button className="terminal-button nav-action" onClick={refreshPage} type="button">
            刷新数据
          </button>
        </div>
      </header>

      <main className="app-main">
        <Header eyebrow={meta.eyebrow} title={meta.title} description={meta.description} />
        <Outlet />
      </main>
    </div>
  );
}
