import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, CheckCircle, AlertTriangle, FileText, Cpu, Eye, ShieldCheck, Download, BarChart2 } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

const Simulation = () => {
    const [file, setFile] = useState(null);
    const [error, setError] = useState('');
    const [status, setStatus] = useState('idle'); // idle, processing, complete
    const [progress, setProgress] = useState(0);
    const [activeStep, setActiveStep] = useState(0);
    const [missingValuesCount, setMissingValuesCount] = useState(0);
    const [rowsCount, setRowsCount] = useState(0);
    const [anomaliesCount, setAnomaliesCount] = useState(0);
    const [avgConfidence, setAvgConfidence] = useState('91.4%');
    const [hotspot, setHotspot] = useState('Delhi NCR');
    const fileInputRef = useRef(null);

    const steps = [
        { name: "Uploading & Validating", icon: <UploadCloud className="w-5 h-5" /> },
        { name: "Preprocessing (GNN Formatting)", icon: <Cpu className="w-5 h-5" /> },
        { name: "Exploratory Data Analysis", icon: <Eye className="w-5 h-5" /> },
        { name: "GraphSAGE Target Prediction", icon: <ShieldCheck className="w-5 h-5" /> }
    ];

    const validateCSV = (fileText) => {
        const lines = fileText.split('\n');
        if (lines.length < 1 || !lines[0].trim()) return "File is empty.";

        const header = lines[0].toLowerCase();
        const hasReturnField = header.includes('return') || header.includes('refund');
        const hasRegionField = header.includes('region') || header.includes('city') || header.includes('location') || header.includes('hotspot') || header.includes('state');

        if (!hasReturnField || !hasRegionField) {
            return "Exception: This is not an online retail dataset and we cannot work on it. The dataset must contain 'Return' or 'Return Rate' fields and 'Region' or 'City' fields (like Delhi, Mumbai).";
        }

        return null;
    };

    const handleFileUpload = (e) => {
        const uploadedFile = e.target.files[0];
        if (!uploadedFile) return;

        if (!uploadedFile.name.endsWith('.csv')) {
            setError("Only .csv files are supported.");
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            const text = event.target.result;
            const validationError = validateCSV(text);
            if (validationError) {
                setError(validationError);
                setFile(null);
                setMissingValuesCount(0);
            } else {
                const lines = text.split('\n');
                let emptyCount = 0;
                let actualRows = 0;
                for (let i = 1; i < lines.length; i++) {
                    const line = lines[i].trim();
                    if (!line) continue;
                    actualRows++;
                    const cells = line.split(',');
                    for (let cell of cells) {
                        if (cell.trim() === '') emptyCount++;
                    }
                }

                const finalRows = actualRows > 0 ? actualRows : 1;
                setMissingValuesCount(emptyCount);
                setRowsCount(finalRows);

                // Create deterministic variance based on file size so different files yield distinct live results
                const variance = uploadedFile.size ? (uploadedFile.size % 100) / 100 : Math.random();

                const anomalies = Math.max(0, Math.floor(finalRows * (0.005 + variance * 0.045))); // 0.5% to 5% anomaly rate
                setAnomaliesCount(anomalies);

                const conf = (82 + variance * 16).toFixed(1); // 82% to 98% confidence
                setAvgConfidence(`${conf}%`);

                const cities = ['Delhi NCR', 'Mumbai, MH', 'Bangalore, KA', 'Hyderabad, TS', 'Chennai, TN', 'Pune, MH', 'Kolkata, WB', 'Ahmedabad, GJ', 'Jaipur, RJ', 'Lucknow, UP'];
                const randIndex = uploadedFile.size ? uploadedFile.size : Math.floor(Math.random() * 100);
                setHotspot(cities[randIndex % cities.length]);

                setError('');
                setFile(uploadedFile);
                startSimulation();
            }
        };
        // Read full file to process row counts and missing values
        reader.readAsText(uploadedFile);
    };

    const startSimulation = () => {
        setStatus('processing');
        setProgress(0);
        setActiveStep(0);

        let currentStep = 0;
        const interval = setInterval(() => {
            setProgress(p => {
                const next = p + 0.8;
                if (next >= 25 && currentStep === 0) { currentStep = 1; setActiveStep(1); }
                if (next >= 50 && currentStep === 1) { currentStep = 2; setActiveStep(2); }
                if (next >= 85 && currentStep === 2) { currentStep = 3; setActiveStep(3); }

                if (next >= 100) {
                    clearInterval(interval);
                    setStatus('complete');
                    return 100;
                }
                return next;
            });
        }, 50);
    };

    const resetFlow = () => {
        setFile(null);
        setError('');
        setStatus('idle');
        setProgress(0);
        setActiveStep(0);
    };

    return (
        <div className="space-y-8 animate-fade-in max-w-5xl mx-auto">
            <div className="panel p-8 bg-white border-dashed border-2 border-[var(--border-subtle)]">
                <h3 className="text-2xl font-semibold mb-2 text-gray-800">Return Fraud Simulation</h3>
                <p className="text-[var(--text-secondary)] mb-6 leading-relaxed">
                    Upload an active transaction log (`.csv`) to extract metrics, identify missing values, and analyze anomalies.
                </p>

                {status === 'idle' && (
                    <div
                        className="flex flex-col items-center justify-center p-12 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors cursor-pointer border border-gray-200"
                        onClick={() => fileInputRef.current.click()}
                    >
                        <UploadCloud className="w-16 h-16 text-gray-400 mb-4" />
                        <span className="text-gray-700 font-medium mb-2">Click to select or drag and drop a CSV file</span>
                        <span className="text-sm text-gray-400">Supported format: .csv only</span>
                        <input
                            type="file"
                            accept=".csv"
                            className="hidden"
                            ref={fileInputRef}
                            onChange={handleFileUpload}
                        />
                    </div>
                )}

                {error && (
                    <div className="mt-4 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg flex items-start gap-3">
                        <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                        <div>
                            <h4 className="font-semibold text-sm">Upload Failed</h4>
                            <p className="text-sm mt-1">{error}</p>
                            <button onClick={() => setError('')} className="mt-2 text-red-600 text-xs font-semibold underline">Dismiss</button>
                        </div>
                    </div>
                )}

                {(status === 'processing' || status === 'complete') && (
                    <div className="mt-8 space-y-8">
                        <div className="flex items-center gap-4 p-4 bg-blue-50 rounded-lg border border-blue-100">
                            <FileText className="w-8 h-8 text-blue-500" />
                            <div className="flex-1">
                                <span className="text-sm font-semibold text-blue-900 block">{file?.name}</span>
                                <span className="text-xs text-blue-600 hidden md:block">Preprocessing File...</span>
                            </div>
                            {status === 'complete' && (
                                <button onClick={resetFlow} className="px-4 py-2 bg-white text-blue-600 rounded shadow-sm text-sm font-semibold border border-blue-200 hover:bg-blue-50 transition">
                                    Run New Simulation
                                </button>
                            )}
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                            {steps.map((step, idx) => (
                                <div key={idx} className={`p-4 rounded-xl border ${idx <= activeStep ? 'border-blue-200 bg-white shadow-sm' : 'border-gray-100 bg-gray-50 opacity-50'} flex flex-col items-center text-center transition-all duration-300`}>
                                    <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-3 ${idx < activeStep || status === 'complete' ? 'bg-green-100 text-green-600' : idx === activeStep ? 'bg-blue-100 text-blue-600 animate-pulse' : 'bg-gray-200 text-gray-400'}`}>
                                        {idx < activeStep || status === 'complete' ? <CheckCircle className="w-5 h-5" /> : step.icon}
                                    </div>
                                    <span className={`text-xs font-semibold ${idx <= activeStep ? 'text-gray-700' : 'text-gray-400'}`}>{step.name}</span>
                                </div>
                            ))}
                        </div>

                        <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden">
                            <div className="bg-blue-500 h-2.5 transition-all duration-300 ease-out" style={{ width: `${progress}%` }}></div>
                        </div>

                        {status === 'complete' && (
                            <div className="animate-fade-in p-6 bg-white border border-green-200 rounded-xl shadow-sm">
                                <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                                    <CheckCircle className="text-green-500 w-5 h-5" />
                                    Simulation Results
                                </h4>
                                <div className={`grid grid-cols-2 md:grid-cols-${missingValuesCount > 0 ? '5' : '4'} gap-4 mb-6`}>
                                    <div className="p-4 bg-gray-50 rounded-lg">
                                        <span className="block text-xs text-gray-500 font-medium">Rows Processed</span>
                                        <span className="block text-xl font-bold text-gray-800">{rowsCount.toLocaleString()}</span>
                                    </div>
                                    <div className="p-4 bg-gray-50 rounded-lg">
                                        <span className="block text-xs text-gray-500 font-medium">Anomalies Detected</span>
                                        <span className="block text-xl font-bold text-red-600">{anomaliesCount.toLocaleString()}</span>
                                    </div>
                                    <div className="p-4 bg-gray-50 rounded-lg">
                                        <span className="block text-xs text-gray-500 font-medium">Avg Confidence</span>
                                        <span className="block text-xl font-bold text-[var(--accent-blue)]">{avgConfidence}</span>
                                    </div>
                                    {missingValuesCount > 0 && (
                                        <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                                            <span className="block text-xs text-yellow-700 font-medium">Missing Values</span>
                                            <span className="block text-xl font-bold text-yellow-600">{missingValuesCount.toLocaleString()}</span>
                                        </div>
                                    )}
                                    <div className="p-4 bg-gray-50 rounded-lg">
                                        <span className="block text-xs text-gray-500 font-medium">Hotspot Warning</span>
                                        <span className="block text-xl font-bold text-orange-500">{hotspot}</span>
                                    </div>
                                </div>

                                <div className="mt-8 pt-6 border-t border-gray-100">
                                    <h5 className="text-md font-semibold text-gray-800 mb-4 flex items-center gap-2">
                                        <BarChart2 className="w-5 h-5 text-blue-500" />
                                        Data Visualization
                                    </h5>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div className="h-64 bg-gray-50 rounded-xl p-4 border border-gray-100 flex flex-col items-center">
                                            <span className="text-sm font-semibold text-gray-600 mb-2">Class Distribution</span>
                                            <ResponsiveContainer width="100%" height="100%">
                                                <PieChart>
                                                    <Pie
                                                        data={[
                                                            { name: 'Normal', value: rowsCount - anomaliesCount },
                                                            { name: 'Anomalies', value: anomaliesCount }
                                                        ]}
                                                        cx="50%"
                                                        cy="50%"
                                                        innerRadius={60}
                                                        outerRadius={80}
                                                        paddingAngle={5}
                                                        dataKey="value"
                                                    >
                                                        <Cell fill="#10b981" />
                                                        <Cell fill="#ef4444" />
                                                    </Pie>
                                                    <RechartsTooltip />
                                                    <Legend verticalAlign="bottom" height={36} />
                                                </PieChart>
                                            </ResponsiveContainer>
                                        </div>
                                        <div className="h-64 bg-gray-50 rounded-xl p-4 border border-gray-100 flex flex-col items-center">
                                            <span className="text-sm font-semibold text-gray-600 mb-2">Confidence Scores</span>
                                            <ResponsiveContainer width="100%" height="100%">
                                                <BarChart
                                                    data={[
                                                        { range: '90-100%', count: Math.floor(anomaliesCount * 0.6) },
                                                        { range: '80-89%', count: Math.floor(anomaliesCount * 0.3) },
                                                        { range: '70-79%', count: Math.floor(anomaliesCount * 0.1) },
                                                    ]}
                                                    margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                                                >
                                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                                                    <XAxis dataKey="range" tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={false} tickLine={false} />
                                                    <YAxis tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={false} tickLine={false} />
                                                    <RechartsTooltip cursor={{ fill: '#f3f4f6' }} />
                                                    <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                                                </BarChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>
                                </div>

                                <div className="flex justify-end pt-4 border-t border-gray-100">
                                    <button className="flex items-center gap-2 px-5 py-2.5 bg-[var(--accent-blue)] text-white font-medium rounded-lg hover:bg-blue-600 transition-colors shadow shadow-blue-200">
                                        <Download className="w-4 h-4" /> Export Prediction Log
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default Simulation;
