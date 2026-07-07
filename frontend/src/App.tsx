import { Route, Routes } from 'react-router-dom';
import { AppLayout } from './components/layout/AppLayout';
import { BacktestPage } from './pages/BacktestPage';
import { ChartPage } from './pages/ChartPage';
import { DashboardPage } from './pages/DashboardPage';
import { SignalsPage } from './pages/SignalsPage';

export function App() {
  return (
    <Routes>
      <Route path="/" element={<AppLayout />}>
        <Route index element={<DashboardPage />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="chart" element={<ChartPage />} />
        <Route path="signals" element={<SignalsPage />} />
        <Route path="backtest" element={<BacktestPage />} />
      </Route>
    </Routes>
  );
}
