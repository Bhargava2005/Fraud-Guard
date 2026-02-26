import React from 'react';
import { Activity, TrendingUp, AlertCircle } from 'lucide-react';

const MetricCard = ({ title, value, previous, label, type, delay = 0 }) => {
    const getIcon = () => {
        switch (type) {
            case 'accuracy': return <TrendingUp className="text-[var(--accent-blue)] w-6 h-6" />;
            case 'loss': return <AlertCircle className="text-[var(--accent-red)] w-6 h-6" />;
            default: return <Activity className="text-gray-400 w-6 h-6" />;
        }
    };

    const isPositive = previous && value > previous;
    const isLoss = type === 'loss';
    const goodIndicator = isLoss ? !isPositive : isPositive;

    return (
        <div
            className="panel p-6 flex flex-col justify-between animate-fade-in bg-white relative overflow-hidden"
            style={{ animationDelay: `${delay}ms` }}
        >
            <div className={`absolute top-0 left-0 w-full h-1 ${type === 'accuracy' ? 'bg-[var(--accent-teal)]' : 'bg-[var(--accent-blue)]'}`}></div>

            <div className="flex justify-between items-center mb-4">
                <h3 className="text-sm font-semibold text-[var(--text-secondary)] tracking-wide uppercase">{title}</h3>
                <div className="p-2 bg-gray-50 rounded-lg border border-gray-100">
                    {getIcon()}
                </div>
            </div>

            <div>
                <div className="text-4xl font-bold text-[var(--text-primary)] mb-2 tracking-tight">
                    {type === 'accuracy' ? `${value.toFixed(1)}%` : value.toFixed(3)}
                </div>

                {previous && (
                    <div className="flex items-center text-sm font-medium">
                        <span className={goodIndicator ? 'text-[var(--accent-teal)]' : 'text-[var(--accent-red)]'}>
                            {goodIndicator ? '▲ ' : '▼ '}{Math.abs((value - previous) / previous * 100).toFixed(1)}%
                        </span>
                        <span className="text-[var(--text-secondary)] ml-2">vs tabular baseline</span>
                    </div>
                )}
                {label && <div className="text-sm text-[var(--text-secondary)] mt-1">{label}</div>}
            </div>
        </div>
    );
};

export default MetricCard;
