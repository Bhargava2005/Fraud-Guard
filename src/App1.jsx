import React, { useEffect, useState } from 'react';
import { fetchDashboardData } from './utils/dataLoader.js';
import MetricCard from './components/MetricCard.jsx';
import NetworkGraph from './components/NetworkGraph.jsx';
import Inspect from './components/Inspect.jsx';
import { Search } from 'lucide-react';
import ChartWidget from './components/ChartWidget.jsx';
import FeatureImportance from './components/FeatureImportance.jsx';
import AdditionalCharts from './components/AdditionalCharts.jsx';
import Simulation from './components/Simulation.jsx';
import LandingPage from './components/LandingPage.jsx';
import GeoGeographicalAnalysis from './components/Geographicalanalysis.jsx';  // ✅ already imported
import { Activity, Database, PlayCircle, LayoutDashboard, ShieldCheck, Info, Home, MapPin } from 'lucide-react';  // ✅ added MapPin icon

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('home');

  useEffect(() => {
    const loadData = async () => {
      const result = await fetchDashboardData();
      setData(result);
      setLoading(false);
    };
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--bg-light)]">
        <div className="relative">
          <ShieldCheck className="w-16 h-16 text-[var(--accent-blue)] animate-pulse" />
          <div className="absolute inset-0 rounded-full border-2 border-[var(--accent-teal)] animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite]"></div>
        </div>
        <span className="ml-4 text-gray-800 text-xl font-medium tracking-wide">Initializing FraudGuard...</span>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex bg-[var(--bg-light)] text-[var(--text-primary)] font-sans overflow-x-hidden">

      {/* Sidebar Navigation */}
      <aside className="w-64 panel border-r border-[var(--border-subtle)] flex flex-col hidden md:flex rounded-none shadow-none z-20">
        <div className="p-6 border-b border-[var(--border-subtle)] flex items-center gap-3 bg-white">
          <ShieldCheck className="text-[var(--accent-blue)] w-8 h-8" />
          <h1 className="text-2xl font-bold text-gradient">FraudGuard</h1>
        </div>

        <nav className="flex-1 p-4 space-y-2 bg-white">
          <NavItem
            icon={<Home />} label="Welcome"
            isActive={activeTab === 'home'}
            onClick={() => setActiveTab('home')}
          />
          <NavItem
            icon={<LayoutDashboard />} label="System Overview"
            isActive={activeTab === 'overview'}
            onClick={() => setActiveTab('overview')}
          />
          <NavItem
            icon={<Database />} label="EDA & Model Metrics"
            isActive={activeTab === 'eda'}
            onClick={() => setActiveTab('eda')}
          />
          <NavItem
            icon={<PlayCircle />} label="Simulation"
            isActive={activeTab === 'simulation'}
            onClick={() => setActiveTab('simulation')}
          />
          {/* ✅ NEW: Geo Analysis tab */}
          <NavItem
            icon={<MapPin />} label="Geo Risk Analysis"
            isActive={activeTab === 'geo'}
            onClick={() => setActiveTab('geo')}
          />
          <NavItem
            icon={<Search />} label="Inspect Order"
            isActive={activeTab === 'inspect'}
            onClick={() => setActiveTab('inspect')}
          />
        </nav>

        <div className="p-6 border-t border-[var(--border-subtle)] bg-gray-50">
          <div className="flex items-center gap-3 text-sm text-[var(--text-secondary)] font-medium">
            <span className="w-2.5 h-2.5 rounded-full bg-[var(--accent-teal)] animate-pulse"></span>
            System Active
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto bg-[var(--bg-light)] w-full">

        {/* ✅ Header shown for all tabs EXCEPT home and geo
            geo gets no header/padding — it owns the full area */}
        {activeTab !== 'home' && activeTab !== 'geo' && (
          <div className="p-4 md:p-8">
            <header className="mb-10 animate-fade-in">
              <h2 className="text-4xl font-semibold mb-3 text-gray-800 tracking-tight">
                {activeTab === 'overview'   && 'System Overview'}
                {activeTab === 'eda'        && 'Exploratory Data Analysis'}
                {activeTab === 'simulation' && 'Live Simulation'}
              </h2>
              <p className="text-[var(--text-secondary)] text-lg max-w-3xl leading-relaxed">
                {activeTab === 'overview'   && 'High-level real-time performance metrics and dataset specifications.'}
                {activeTab === 'eda'        && 'Analyzing transactional trends, model metrics, and network behavior across multiple combined retail platforms.'}
                {activeTab === 'simulation' && 'Upload a custom active transaction log to test the FraudGuard predictive pipeline.'}
              </p>
            </header>
          </div>
        )}

        {/* ── Tab Contents ── */}

        {activeTab === 'home' && (
          <div className="p-4 md:p-8 pb-24">
            <LandingPage onStartSimulation={() => setActiveTab('simulation')} />
          </div>
        )}

        {activeTab === 'overview' && (
          <div className="p-4 md:p-8 pb-24 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 max-w-7xl animate-fade-in">
              <MetricCard title={data.metrics.gat.title}       value={data.metrics.gat.accuracy}       type="accuracy" delay={100} />
              <MetricCard title={data.metrics.graphsage.title} value={data.metrics.graphsage.accuracy} type="accuracy" delay={200} />
              <MetricCard title={data.metrics.xgboost.title}   value={data.metrics.xgboost.loss} label="Log Loss" type="loss" delay={300} />
            </div>

            <div className="panel p-8 bg-white max-w-7xl animate-fade-in">
              <h3 className="text-xl font-semibold mb-8 border-b pb-4 text-gray-800 flex items-center gap-3">
                <Info className="text-[var(--accent-blue)] w-6 h-6" /> Project Insights & Training Baseline
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-y-10 gap-x-8">
                <div><span className="block text-sm text-[var(--accent-blue)] font-bold uppercase tracking-wider mb-2">Total Rows</span><span className="text-2xl font-semibold text-gray-800">{data.systemInsights.totalDatasetSize}</span></div>
                <div><span className="block text-sm text-[var(--accent-blue)] font-bold uppercase tracking-wider mb-2">Unique Users</span><span className="text-2xl font-semibold text-gray-800">{data.systemInsights.uniqueUsers}</span></div>
                <div><span className="block text-sm text-[var(--accent-blue)] font-bold uppercase tracking-wider mb-2">Unique Products</span><span className="text-2xl font-semibold text-gray-800">{data.systemInsights.uniqueProducts}</span></div>
                <div><span className="block text-sm text-[var(--accent-blue)] font-bold uppercase tracking-wider mb-2">Total Returns</span><span className="text-2xl font-semibold text-gray-800">{data.systemInsights.totalReturns}</span></div>
                <div><span className="block text-sm text-[var(--accent-blue)] font-bold uppercase tracking-wider mb-2">Global Return Rate</span><span className="text-2xl font-semibold text-gray-800">{data.systemInsights.globalReturnRate}</span></div>
                <div><span className="block text-sm text-[var(--accent-blue)] font-bold uppercase tracking-wider mb-2">Platforms Scanned</span><span className="text-lg font-medium text-gray-700">{data.systemInsights.platformsIncluded}</span></div>
                <div><span className="block text-sm text-[var(--accent-blue)] font-bold uppercase tracking-wider mb-2">GNN Training Time</span><span className="text-2xl font-semibold text-gray-800">{data.systemInsights.trainingTime}</span></div>
                <div><span className="block text-sm text-[var(--accent-blue)] font-bold uppercase tracking-wider mb-2">Inference Speed</span><span className="text-xl font-semibold text-gray-800">{data.systemInsights.inferenceTime}</span></div>
              </div>
            </div>

            <div className="panel p-8 bg-white max-w-7xl animate-fade-in" style={{ animationDelay: '200ms' }}>
              <h3 className="text-lg font-semibold mb-4 text-[var(--accent-teal)]">Executive Summary</h3>
              <p className="text-gray-700 leading-relaxed text-lg mb-4">
                The rapid expansion of the e-commerce sector has revolutionized retail, but introduced significant financial risks in the form of return fraud. Unlike traditional payment fraud (stolen credit cards), return fraud—specifically <strong>"Wardrobing"</strong> or <strong>"Return Abuse"</strong>—is committed by legitimate customers who purchase items with the intent to use them briefly and return them for a full refund.
              </p>
              <p className="text-gray-700 leading-relaxed text-lg">
                Traditional tabular machine learning models treat every transaction as an independent "island," ignoring structural 'guilt-by-association' relationships. FraudGuard dynamically maps users and items as a large bipartite graph, successfully identifying anomalous purchase histories across platforms utilizing powerful Graph Neural Networks.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'eda' && (
          <div className="p-4 md:p-8 pb-24 space-y-12 animate-fade-in w-full max-w-[1600px]">
            <div className="w-full">
              <AdditionalCharts platformData={data.platformDistribution} hotspotsData={data.hotspots} />
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 w-full">
              <div className="flex flex-col w-full min-w-0">
                <FeatureImportance data={data.featureImportance.map(f => ({ ...f, score: f.score * 100 }))} />
                <p className="mt-4 text-[var(--text-secondary)] text-sm px-2 bg-white panel p-4 shadow-sm border border-[var(--border-subtle)]">
                  <strong>XGBoost Feature Importance:</strong> Uncovered that "User Global Return Rate" is the most critical singular numerical feature in predicting return fraud.
                </p>
              </div>
              <div className="flex flex-col w-full min-w-0">
                <div className="panel p-4 rounded-2xl bg-white overflow-hidden w-full flex items-center justify-center">
                  <img src="/images/map_visualization.png" alt="Geospatial Sales Density Map" className="w-full h-auto max-h-[400px] object-contain" />
                </div>
                <p className="mt-4 text-[var(--text-secondary)] text-sm px-2 bg-white panel p-4 shadow-sm border border-[var(--border-subtle)]">
                  <strong>Geospatial Sales Density:</strong> Visualizing localized transactions based on Latitude and Longitude to identify high-density transactional hubs.
                </p>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-6 w-full min-w-0">
              <div className="w-full xl:max-w-4xl mx-auto">
                <ChartWidget title="Monthly Aggregate Valid Sales Trajectory" data={data.salesTrend} delay={300} />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full min-w-0">
                <div className="panel p-4 bg-white flex flex-col items-center w-full min-w-0">
                  <h4 className="font-semibold text-gray-700 mb-4 text-center">Distributions</h4>
                  <img src="/images/image1.png" alt="Variables Distribution" className="w-full h-auto max-h-[250px] object-contain mix-blend-multiply" />
                  <p className="mt-4 text-xs text-gray-500 text-center px-2">Analyzing the distributions and spread of key e-commerce numeric variables.</p>
                </div>
                <div className="panel p-4 bg-white flex flex-col items-center w-full min-w-0">
                  <h4 className="font-semibold text-gray-700 mb-4 text-center">Class Imbalance</h4>
                  <img src="/images/image2.png" alt="Target Class Imbalance" className="w-full h-auto max-h-[250px] object-contain mix-blend-multiply" />
                  <p className="mt-4 text-xs text-gray-500 text-center px-2">Highlighting the extreme variance between standard valid purchases and rare fraudulent returns.</p>
                </div>
                <div className="panel p-4 bg-white flex flex-col items-center w-full min-w-0">
                  <h4 className="font-semibold text-gray-700 mb-4 text-center">Correlations</h4>
                  <img src="/images/image3.png" alt="Feature Correlations" className="w-full h-auto max-h-[250px] object-contain mix-blend-multiply" />
                  <p className="mt-4 text-xs text-gray-500 text-center px-2">Examining the mathematical relationships and feature dependencies within the transaction dataset.</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'simulation' && (
          <div className="p-4 md:p-8 pb-24">
            <Simulation />
          </div>
        )}

        {/* ✅ NEW: Geo Analysis tab — full height, no padding, owns its own layout */}
        {activeTab === 'geo' && (
          <div style={{ height: 'calc(100vh - 0px)', overflow: 'hidden' }}>
            <GeoGeographicalAnalysis />
          </div>
        )}
        {activeTab === 'inspect' && (
          <div className="p-4 md:p-8 pb-24">
            <Inspect />
          </div>
        )}
      </main>
    </div>
  );
}

const NavItem = ({ icon, label, isActive, onClick }) => {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 ${
        isActive
          ? 'bg-blue-50 text-[var(--accent-blue)] border-l-4 border-[var(--accent-blue)] font-semibold'
          : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50 border-l-4 border-transparent'
      }`}
    >
      <span className={isActive ? 'text-[var(--accent-blue)]' : ''}>{icon}</span>
      <span>{label}</span>
    </button>
  );
};

export default App;