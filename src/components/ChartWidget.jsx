import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const ChartWidget = ({ data = [], title, delay = 0 }) => {
    return (
        <div className="panel p-6 w-full h-[400px] flex flex-col animate-fade-in bg-white" style={{ animationDelay: `${delay}ms` }}>
            <h3 className="text-[var(--text-primary)] font-semibold mb-2 text-lg">{title}</h3>
            <p className="text-[var(--text-secondary)] text-sm mb-4">Tracking dynamic sales volume representing typical transaction loads against which anomalous behaviors (sudden spikes in returns/wardrobing) are evaluated by our Graph Neural Networks.</p>
            <div className="flex-1 w-full min-w-0 min-h-0 relative">
                <ResponsiveContainer width="99%" height="100%">
                    <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorSales" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                        <XAxis
                            dataKey="month"
                            stroke="#64748b"
                            tick={{ fill: '#64748b' }}
                            axisLine={false}
                            tickLine={false}
                            dy={10}
                        />
                        <YAxis
                            stroke="#64748b"
                            tick={{ fill: '#64748b' }}
                            axisLine={false}
                            tickLine={false}
                            dx={-10}
                            tickFormatter={(value) => `$${value}k`}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#ffffff',
                                border: '1px solid #e2e8f0',
                                borderRadius: '8px',
                                color: '#1e293b',
                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                            }}
                        />
                        <Area
                            type="monotone"
                            dataKey="sales"
                            stroke="#3b82f6"
                            strokeWidth={3}
                            fillOpacity={1}
                            fill="url(#colorSales)"
                            animationDuration={2000}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default ChartWidget;
