import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const FeatureImportance = ({ data = [], delay = 0 }) => {
    return (
        <div className="panel p-6 w-full h-[400px] flex flex-col animate-fade-in bg-white" style={{ animationDelay: `${delay}ms` }}>
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-[var(--text-primary)] font-semibold text-lg">XGBoost Relative Feature Importance</h3>
                <span className="text-xs font-medium px-3 py-1 bg-blue-50 text-[var(--accent-blue)] rounded-full border border-blue-100">
                    Ensemble Model
                </span>
            </div>

            <div className="flex-1 w-full min-w-0 min-h-0 relative">
                <ResponsiveContainer width="99%" height="100%">
                    <BarChart data={data} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={true} vertical={false} />
                        <XAxis
                            type="number"
                            stroke="#64748b"
                            axisLine={false}
                            tickLine={false}
                        />
                        <YAxis
                            dataKey="feature"
                            type="category"
                            stroke="#1e293b"
                            axisLine={false}
                            tickLine={false}
                            width={140}
                            tick={{ fontSize: 12, fill: '#475569' }}
                        />
                        <Tooltip
                            cursor={{ fill: '#f8fafc' }}
                            contentStyle={{
                                backgroundColor: '#ffffff',
                                border: '1px solid #e2e8f0',
                                borderRadius: '8px',
                                color: '#1e293b',
                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                            }}
                        />
                        <Bar dataKey="score" radius={[0, 4, 4, 0]} animationDuration={1500}>
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={index === 0 ? '#10b981' : '#3b82f6'} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default FeatureImportance;
