import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

const AdditionalCharts = ({ platformData = [], hotspotsData = [] }) => {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full min-w-0">
            <div className="panel p-6 bg-white flex flex-col items-center w-full min-w-0">
                <h3 className="text-[var(--text-primary)] font-semibold text-lg mb-2 text-center">Platform Distribution</h3>
                <p className="text-[var(--text-secondary)] text-sm text-center mb-4 max-w-sm">Combining data from Amazon, Flipkart, Meesho, and Ajio to prevent fraudsters from hiding their behavior.</p>
                <div className="w-full h-[250px] min-w-0 min-h-0 relative">
                    <ResponsiveContainer width="99%" height="100%">
                        <PieChart>
                            <Pie
                                data={platformData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                                nameKey="name"
                                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            >
                                {platformData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#ffffff',
                                    border: '1px solid #e2e8f0',
                                    borderRadius: '8px',
                                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                                }}
                            />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
                <p className="mt-4 text-[var(--text-secondary)] text-sm px-2">
                    <strong>Insight:</strong> By consolidating the "Unified Indian E-Commerce Dataset", FraudGuard tracks a suspect's User Global Return Rate regardless of platform switching.
                </p>
            </div>

            <div className="panel p-6 bg-white flex flex-col items-center w-full min-w-0">
                <h3 className="text-[var(--text-primary)] font-semibold text-lg mb-2 text-center">Regional Return Hotspots</h3>
                <p className="text-[var(--text-secondary)] text-sm text-center mb-4 max-w-sm">Identifying synthesized geographic hubs with the highest "Wardrobing" indicators.</p>
                <div className="w-full h-[250px] min-w-0 min-h-0 relative">
                    <ResponsiveContainer width="99%" height="100%">
                        <BarChart data={hotspotsData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                            <XAxis dataKey="city" stroke="#64748b" tickLine={false} axisLine={false} />
                            <YAxis
                                stroke="#64748b"
                                tickLine={false}
                                axisLine={false}
                                tickFormatter={(value) => `${value}%`}
                            />
                            <Tooltip
                                cursor={{ fill: '#f8fafc' }}
                                contentStyle={{
                                    backgroundColor: '#ffffff',
                                    border: '1px solid #e2e8f0',
                                    borderRadius: '8px'
                                }}
                            />
                            <Bar dataKey="returnRate" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                                {hotspotsData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={index === 0 ? '#ef4444' : '#3b82f6'} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
                <p className="mt-4 text-[var(--text-secondary)] text-sm px-2">
                    <strong>Insight:</strong> Geospatially augmented data helps detect clustered fraud rings coordinating abuse from specific hubs like Delhi NCR.
                </p>
            </div>
        </div>
    );
};

export default AdditionalCharts;
