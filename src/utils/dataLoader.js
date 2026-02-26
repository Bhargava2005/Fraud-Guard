export const fetchDashboardData = async () => {
    return new Promise((resolve) => {
        // Simulate network delay
        setTimeout(() => {
            resolve({
                metrics: {
                    gat: { accuracy: 89.4, f1: 88.2, loss: 0.24, title: "GAT Performance" },
                    graphsage: { accuracy: 91.2, f1: 90.5, loss: 0.18, title: "GraphSAGE Performance" },
                    xgboost: { accuracy: 93.8, f1: 92.1, loss: 0.12, title: "XGBoost Ensemble" }
                },
                systemInsights: {
                    totalDatasetSize: "808,550 Rows",
                    uniqueUsers: "4,500+",
                    uniqueProducts: "30,000+",
                    totalReturns: "18,205",
                    globalReturnRate: "2.25%",
                    platformsIncluded: "Amazon, Flipkart, Meesho, Ajio",
                    trainingTime: "45 Mins (GNN)",
                    inferenceTime: "1.2s per 10k batch"
                },
                featureImportance: [
                    { feature: 'User Global Return Rate', score: 0.38 },
                    { feature: 'UnitPrice', score: 0.22 },
                    { feature: 'Quantity', score: 0.18 },
                    { feature: 'Platform (Meesho/Ajio)', score: 0.12 },
                    { feature: 'Distance to Hub', score: 0.10 }
                ],
                salesTrend: [
                    { month: 'Jan', sales: 120 },
                    { month: 'Feb', sales: 150 },
                    { month: 'Mar', sales: 180 },
                    { month: 'Apr', sales: 140 },
                    { month: 'May', sales: 220 },
                    { month: 'Jun', sales: 300 },
                    { month: 'Jul', sales: 280 }
                ],
                platformDistribution: [
                    { name: 'Amazon', value: 35 },
                    { name: 'Flipkart', value: 30 },
                    { name: 'Meesho', value: 20 },
                    { name: 'Ajio', value: 10 },
                    { name: 'Others', value: 5 }
                ],
                hotspots: [
                    { city: 'Delhi NCR', returnRate: 15.2 },
                    { city: 'Mumbai', returnRate: 12.8 },
                    { city: 'Bengaluru', returnRate: 10.5 },
                    { city: 'Hyderabad', returnRate: 8.9 },
                    { city: 'Chennai', returnRate: 7.4 }
                ],
                networkNodes: Array.from({ length: 40 }).map((_, i) => ({
                    id: `node${i}`,
                    type: Math.random() > 0.6 ? 'item' : 'user', // 60% users, 40% items roughly
                    risk: Math.random() // Used for coloring nodes slightly differently based on "fraud risk"
                })),
                networkLinks: Array.from({ length: 50 }).map((_, i) => ({
                    source: `node${Math.floor(Math.random() * 24)}`, // Link primarily from the first 24 (likely users)
                    target: `node${Math.floor(25 + Math.random() * 15)}` // Link to items
                }))
            });
        }, 800);
    });
};
