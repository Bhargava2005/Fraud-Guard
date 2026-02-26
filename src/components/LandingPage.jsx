import React from 'react';
import { ShieldCheck, Crosshair, Map, Activity, Play, ChevronRight, AreaChart, GitMerge, Database, Cpu, Trophy } from 'lucide-react';

const LandingPage = ({ onStartSimulation }) => {
    return (
        <div className="space-y-12 animate-fade-in max-w-7xl mx-auto pb-16">

            {/* Hero Section */}
            <section className="bg-white rounded-3xl p-10 md:p-16 text-gray-800 relative overflow-hidden shadow-[0_8px_40px_-10px_rgba(59,130,246,0.15)] border border-blue-50 group">
                {/* Decorative background elements */}
                <div className="absolute top-0 right-0 -mr-20 -mt-20 w-96 h-96 bg-gradient-to-br from-blue-50 to-teal-50 rounded-full blur-3xl opacity-60 group-hover:opacity-100 transition-opacity duration-700"></div>

                <div className="absolute top-0 right-0 p-12 opacity-5">
                    <ShieldCheck className="w-96 h-96 text-blue-900" />
                </div>

                <div className="relative z-10 max-w-2xl">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 text-blue-600 font-medium text-sm mb-6 border border-blue-100">
                        <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
                        Final Year Project Overview
                    </div>
                    <h1 className="text-4xl md:text-6xl font-extrabold mb-6 leading-tight text-gray-900">
                        Detecting E-Commerce Return Fraud with <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-[var(--accent-teal)]">Graph Neural Networks</span>
                    </h1>
                    <p className="text-lg md:text-xl text-gray-600 mb-8 leading-relaxed font-medium">
                        FraudGuard is an advanced machine learning pipeline developed to identify "wardrobing" and return abuse across major Indian e-commerce platforms (Amazon, JioMart, Meesho, Myntra) using GraphSAGE algorithms.
                    </p>

                    <button
                        onClick={onStartSimulation}
                        className="group flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-[var(--accent-blue)] to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold rounded-xl transition-all duration-300 shadow-[0_8px_20px_-6px_rgba(59,130,246,0.5)] hover:-translate-y-1"
                    >
                        Demo the Resulting Pipeline
                        <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </button>
                </div>
            </section>

            {/* Project Workflow Grid */}
            <section>
                <div className="text-center mb-10">
                    <h2 className="text-3xl font-bold text-gray-800 mb-4">The Development Journey</h2>
                    <p className="text-gray-500 max-w-2xl mx-auto text-lg">Rather than just an interface, FraudGuard represents a complete end-to-end data science study analyzing real-world retail anomalies.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">

                    {/* Step 1 */}
                    <div className="panel bg-white p-6 rounded-2xl hover:-translate-y-2 transition-transform duration-300 border border-t-4 border-t-blue-500 shadow-md group">
                        <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center mb-5 text-blue-600">
                            <Database className="w-6 h-6" />
                        </div>
                        <h3 className="text-lg font-bold text-gray-800 mb-2">1. Data Aggregation</h3>
                        <p className="text-gray-600 leading-relaxed text-sm">
                            We began by gathering diverse datasets scattered across top e-commerce websites like Amazon, Jio, and Meesho, cleaning and standardizing them into a singular transactional bipartite graph structure.
                        </p>
                    </div>

                    {/* Step 2 */}
                    <div className="panel bg-white p-6 rounded-2xl hover:-translate-y-2 transition-transform duration-300 border border-t-4 border-t-orange-500 shadow-md group">
                        <div className="w-12 h-12 bg-orange-50 rounded-xl flex items-center justify-center mb-5 text-orange-600">
                            <Map className="w-6 h-6" />
                        </div>
                        <h3 className="text-lg font-bold text-gray-800 mb-2">2. Hotspot Analysis</h3>
                        <p className="text-gray-600 leading-relaxed text-sm">
                            Upon exploring the data, we identified geographic 'Hotspots'. These visualizations showed that returns aren't evenly distributed, but highly concentrated in certain localized high-density regions.
                        </p>
                    </div>

                    {/* Step 3 */}
                    <div className="panel bg-white p-6 rounded-2xl hover:-translate-y-2 transition-transform duration-300 border border-t-4 border-t-purple-500 shadow-md group">
                        <div className="w-12 h-12 bg-purple-50 rounded-xl flex items-center justify-center mb-5 text-purple-600">
                            <Cpu className="w-6 h-6" />
                        </div>
                        <h3 className="text-lg font-bold text-gray-800 mb-2">3. Model Training</h3>
                        <p className="text-gray-600 leading-relaxed text-sm">
                            We engineered features and trained three distinct machine learning models to compare their performance under class imbalance: <strong>XGBoost</strong>, <strong>GraphSAGE</strong>, and <strong>Graph Attention Networks (GAT)</strong>.
                        </p>
                    </div>

                    {/* Step 4 */}
                    <div className="panel bg-blue-50 p-6 rounded-2xl hover:-translate-y-2 transition-transform duration-300 border border-t-4 border-t-green-500 shadow-lg group relative overflow-hidden">
                        <div className="absolute -right-4 -bottom-4 opacity-5">
                            <Trophy className="w-32 h-32" />
                        </div>
                        <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mb-5 text-green-600 relative z-10">
                            <Trophy className="w-6 h-6" />
                        </div>
                        <h3 className="text-lg font-bold text-gray-800 mb-2 relative z-10">4. The GraphSAGE Selection</h3>
                        <p className="text-gray-700 leading-relaxed text-sm relative z-10">
                            By testing all models on unseen data, we discovered that <strong>GraphSAGE</strong> was the optimal model for our dataset. It captured localized user-item relationships while generalizing better than GAT and XGBoost.
                        </p>
                    </div>

                </div>
            </section>

            {/* Architecture Highlight */}
            <section className="bg-white rounded-3xl border border-gray-100 shadow-sm overflow-hidden flex flex-col md:flex-row">
                <div className="md:w-1/2 bg-slate-50 border-r border-gray-100 p-10 flex flex-col items-center justify-center text-center">
                    <GitMerge className="w-20 h-20 text-indigo-500 mb-6" />
                    <h3 className="text-2xl font-bold text-gray-800 mb-3">Live React Front-End</h3>
                    <p className="text-gray-600 leading-relaxed max-w-md">
                        This interface allows you to view the resulting dataset metrics, dive into the mathematical distributions, and upload external files to actively simulate our elected predictive pipeline.
                    </p>
                </div>
                <div className="p-10 md:p-12 md:w-1/2 flex flex-col justify-center">
                    <h3 className="text-xl font-bold text-gray-800 mb-6">Explore the Interface</h3>
                    <ul className="space-y-6">
                        <li className="flex items-start gap-4">
                            <div className="p-2 rounded-lg bg-blue-50 text-blue-600"><Activity className="w-5 h-5" /></div>
                            <div>
                                <h4 className="font-semibold text-gray-800">System Overview</h4>
                                <p className="text-sm text-gray-500 mt-1">Review the final metrics of the three trained models and high-level statistics about the datasets tested.</p>
                            </div>
                        </li>
                        <li className="flex items-start gap-4">
                            <div className="p-2 rounded-lg bg-teal-50 text-teal-600"><Database className="w-5 h-5" /></div>
                            <div>
                                <h4 className="font-semibold text-gray-800">EDA & Model Metrics</h4>
                                <p className="text-sm text-gray-500 mt-1">Discover correlations, feature importances, target class imbalance, and the localized Hotspot maps.</p>
                            </div>
                        </li>
                        <li className="flex items-start gap-4">
                            <div className="p-2 rounded-lg bg-indigo-50 text-indigo-600"><Play className="w-5 h-5" /></div>
                            <div>
                                <h4 className="font-semibold text-gray-800">Simulation Environment</h4>
                                <p className="text-sm text-gray-500 mt-1">Upload a real-world `.csv` log to visually funnel the transactions through the GraphSAGE pipeline to detect missing variables and live anomalous nodes.</p>
                            </div>
                        </li>
                    </ul>
                </div>
            </section>
        </div>
    );
};

export default LandingPage;
