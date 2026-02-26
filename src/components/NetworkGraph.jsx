import React, { useEffect, useRef } from 'react';

const NetworkGraph = ({ nodes = [], links = [] }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let animationFrameId;

        // Set responsive size
        const resizeCanvas = () => {
            const parent = canvas.parentElement;
            canvas.width = parent.clientWidth;
            canvas.height = parent.clientHeight;
        };
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        const width = canvas.width;
        const height = canvas.height;

        // Initialize scattered positions dynamically instead of random to avoid cross mark
        const renderNodes = nodes.map((n, i) => {
            // Create a pleasant clustered layout rather than totally random
            const angle = (i / nodes.length) * Math.PI * 2;
            const radius = 50 + Math.random() * (Math.min(width, height) / 2.5);
            return {
                ...n,
                x: width / 2 + Math.cos(angle) * radius,
                y: height / 2 + Math.sin(angle) * radius,
                vx: 0,
                vy: 0
            };
        });

        const render = () => {
            ctx.clearRect(0, 0, width, height);

            // Gentle center gravity
            renderNodes.forEach(n => {
                const dx = (width / 2) - n.x;
                const dy = (height / 2) - n.y;
                n.vx += dx * 0.0005;
                n.vy += dy * 0.0005;
            });

            // Repulsion to space them nicely as a web
            for (let i = 0; i < renderNodes.length; i++) {
                for (let j = i + 1; j < renderNodes.length; j++) {
                    const n1 = renderNodes[i];
                    const n2 = renderNodes[j];
                    const dx = n2.x - n1.x;
                    const dy = n2.y - n1.y;
                    const dist = Math.sqrt(dx * dx + dy * dy) || 1;

                    if (dist < 100) {
                        const force = (100 - dist) / dist * 0.01;
                        n1.vx -= dx * force;
                        n1.vy -= dy * force;
                        n2.vx += dx * force;
                        n2.vy += dy * force;
                    }
                }
            }

            // Spring force for links
            links.forEach(link => {
                const source = renderNodes.find(n => n.id === link.source);
                const target = renderNodes.find(n => n.id === link.target);
                if (source && target) {
                    const dx = target.x - source.x;
                    const dy = target.y - source.y;
                    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                    const force = (dist - 150) * 0.002;
                    source.vx += dx * force;
                    source.vy += dy * force;
                    target.vx -= dx * force;
                    target.vy -= dy * force;
                }
            });

            // Draw links First
            ctx.lineWidth = 1;
            links.forEach(link => {
                const source = renderNodes.find(n => n.id === link.source);
                const target = renderNodes.find(n => n.id === link.target);
                if (source && target) {
                    ctx.beginPath();
                    ctx.moveTo(source.x, source.y);
                    ctx.lineTo(target.x, target.y);
                    ctx.strokeStyle = '#e2e8f0'; // Subtle gray for light theme
                    ctx.stroke();
                }
            });

            // Draw Nodes Last (on top)
            renderNodes.forEach(n => {
                n.vx *= 0.92;
                n.vy *= 0.92;
                n.x += n.vx;
                n.y += n.vy;

                // Bounce off walls gently
                if (n.x < 20) { n.x = 20; n.vx *= -1; }
                if (n.x > width - 20) { n.x = width - 20; n.vx *= -1; }
                if (n.y < 20) { n.y = 20; n.vy *= -1; }
                if (n.y > height - 20) { n.y = height - 20; n.vy *= -1; }

                ctx.beginPath();
                ctx.arc(n.x, n.y, n.type === 'user' ? 6 : 9, 0, Math.PI * 2);

                // Colors: User = Blue, Item = Teal
                let fillStyle = n.type === 'user' ? '#3b82f6' : '#10b981';

                // Slightly change color if risk is high (just for visual variety)
                if (n.risk > 0.8 && n.type === 'user') fillStyle = '#ef4444';

                ctx.fillStyle = fillStyle;
                ctx.fill();

                // Light border
                ctx.lineWidth = 1.5;
                ctx.strokeStyle = '#ffffff';
                ctx.stroke();
            });

            animationFrameId = requestAnimationFrame(render);
        };

        render();

        return () => {
            cancelAnimationFrame(animationFrameId);
            window.removeEventListener('resize', resizeCanvas);
        };
    }, [nodes, links]);

    return (
        <div className="w-full h-[400px] relative">
            <div className="absolute top-4 left-4 z-10 pointer-events-none">
                <h3 className="text-[var(--text-primary)] font-semibold text-lg flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[var(--accent-blue)] animate-pulse"></span>
                    User-Item Latent Connections
                </h3>
                <p className="text-[var(--text-secondary)] text-sm">Real-time force-directed topology map</p>
            </div>
            <canvas ref={canvasRef} className="w-full h-full cursor-default" />
        </div>
    );
};

export default NetworkGraph;
