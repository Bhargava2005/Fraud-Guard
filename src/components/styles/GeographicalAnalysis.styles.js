// Design tokens
export const COLORS = {
  pageBg:    "#f4f6fb",
  cardBg:    "#ffffff",
  accent:    "#646cff",
  accentDk:  "#535bf2",
  cyan:      "#61dafb",
  text:      "#213547",
  muted:     "#888888",
  border:    "#e2e8f0",
  highColor: "#dc2626", highBg: "#fef2f2", highBdr: "#fca5a5",
  modColor:  "#d97706", modBg:  "#fffbeb", modBdr:  "#fcd34d",
  safeColor: "#16a34a", safeBg: "#f0fdf4", safeBdr: "#86efac",
};

// Inline Styles Object
export const S = {
  root: {
    display: "flex", flexDirection: "column", height: "100vh",
    background: COLORS.pageBg, fontFamily: "'Outfit',sans-serif",
    color: COLORS.text, overflow: "hidden",
  },
  topBar: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    flexWrap: "wrap", gap: 12, padding: "10px 20px",
    background: COLORS.cardBg, borderBottom: `1px solid ${COLORS.border}`,
    flexShrink: 0, boxShadow: "0 1px 12px rgba(100,108,255,0.07)",
  },
  topBarLeft: { display: "flex", alignItems: "center", gap: 12 },
  menuBtn: {
    background: "none", border: `1px solid ${COLORS.border}`,
    color: COLORS.muted, padding: "6px 10px", borderRadius: 6,
    cursor: "pointer", fontSize: 18,
    transition: "border-color 300ms, color 300ms",
  },
  appTitle: { fontSize: 16, fontWeight: 600, color: COLORS.text, letterSpacing: "0.01em" },
  statsRow: { display: "flex", gap: 8, flexWrap: "wrap" },
  statChip: {
    display: "flex", flexDirection: "column", alignItems: "center",
    background: COLORS.cardBg, border: "1.5px solid",
    borderRadius: 8, padding: "6px 16px", minWidth: 76,
    boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
  },
  statVal:   { fontSize: 19, fontWeight: 700, fontFamily: "monospace" },
  statLabel: { fontSize: 10, color: COLORS.muted, letterSpacing: "0.05em", textTransform: "uppercase" },
  body:      { display: "flex", flex: 1, overflow: "hidden" },
  sidebar: {
    width: 275, flexShrink: 0,
    background: COLORS.pageBg, borderRight: `1px solid ${COLORS.border}`,
    overflowY: "auto", padding: 12,
    display: "flex", flexDirection: "column", gap: 10,
  },
  card: {
    background: COLORS.cardBg, borderRadius: 10, padding: 16,
    border: `1px solid ${COLORS.border}`,
    boxShadow: "0 1px 6px rgba(0,0,0,0.05)",
  },
  cardTitle: {
    fontSize: 11, fontWeight: 600, color: COLORS.accent,
    letterSpacing: "0.08em", textTransform: "uppercase",
    marginBottom: 12, fontFamily: "monospace",
  },
  fieldLabel: {
    display: "block", fontSize: 11, color: COLORS.muted,
    textTransform: "uppercase", letterSpacing: "0.06em",
  },
  submitBtn: {
    width: "100%",
    background: `linear-gradient(135deg,${COLORS.accent},${COLORS.accentDk})`,
    border: "none", borderRadius: 8, color: "#fff",
    padding: "11px 0", fontSize: 13, fontWeight: 600,
    cursor: "pointer", letterSpacing: "0.04em",
  },
  mapArea: { flex: 1, position: "relative", overflow: "hidden" },
  loadOverlay: {
    position: "absolute", inset: 0,
    background: "rgba(244,246,251,0.92)",
    display: "flex", alignItems: "center", justifyContent: "center",
    zIndex: 1000, backdropFilter: "blur(6px)",
  },
  loadCard: {
    background: COLORS.cardBg, border: `1px solid ${COLORS.border}`,
    borderRadius: 14, padding: "32px 40px", textAlign: "center",
    animation: "fadeUp 0.3s ease",
    boxShadow: "0 8px 40px rgba(100,108,255,0.10)",
  },
  spinner: {
    width: 40, height: 40,
    border: `3px solid ${COLORS.accent}22`,
    borderTop: `3px solid ${COLORS.accent}`,
    borderRadius: "50%", margin: "0 auto 16px",
    animation: "spin-ring 0.8s linear infinite",
  },
  readyBadge: {
    position: "absolute", bottom: 16, left: "50%",
    transform: "translateX(-50%)",
    background: "rgba(255,255,255,0.92)",
    border: `1px solid ${COLORS.cyan}66`,
    borderRadius: 20, padding: "6px 18px",
    fontSize: 12, color: "#0987a0",
    fontFamily: "monospace", fontWeight: 500,
    backdropFilter: "blur(6px)", zIndex: 500,
    animation: "fadeUp 0.4s ease",
    boxShadow: `0 2px 12px ${COLORS.cyan}33`,
    whiteSpace: "nowrap",
  },
};

// Global CSS as a template literal to be used in a <style> tag
export const globalStyles = `
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');
  *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }

  ::-webkit-scrollbar { width:5px; }
  ::-webkit-scrollbar-track { background:${COLORS.pageBg}; }
  ::-webkit-scrollbar-thumb { background:${COLORS.accent}55; border-radius:3px; }
  ::-webkit-scrollbar-thumb:hover { background:${COLORS.accent}; }

  select option { background:#fff; color:${COLORS.text}; }

  .geo-range {
    -webkit-appearance: none;
    appearance: none;
    background: transparent;
    outline: none;
    pointer-events: none;
    height: 0;
    width: 100%;
  }
  .geo-range::-webkit-slider-thumb {
    -webkit-appearance: none;
    pointer-events: all;
    width: 18px; height: 18px;
    border-radius: 50%;
    background: #fff;
    border: 2.5px solid ${COLORS.accent};
    box-shadow: 0 1px 5px rgba(0,0,0,0.18), 0 0 0 3px ${COLORS.accent}22;
    cursor: pointer;
    will-change: filter;
    transition: filter 300ms, box-shadow 200ms;
  }
  .geo-range::-webkit-slider-thumb:hover {
    filter: drop-shadow(0 0 2em ${COLORS.accent}aa);
    box-shadow: 0 1px 5px rgba(0,0,0,0.2), 0 0 0 5px ${COLORS.accent}33;
  }
  .geo-range::-moz-range-thumb {
    pointer-events: all;
    width:18px; height:18px; border-radius:50%;
    background:#fff; border:2.5px solid ${COLORS.accent};
    cursor:pointer; box-sizing:border-box;
  }

  .geo-select:focus {
    border-color:${COLORS.accent} !important;
    box-shadow:0 0 0 3px ${COLORS.accent}22 !important;
    outline:none;
  }

  .geo-submit {
    will-change:filter;
    transition:filter 300ms, transform 0.1s;
  }
  .geo-submit:hover  { filter:drop-shadow(0 0 1.5em ${COLORS.accent}aa); }
  .geo-submit:active { transform:scale(0.98); }

  .geo-menu:hover { border-color:${COLORS.accent} !important; color:${COLORS.accent} !important; }

  .geo-globe { will-change:filter; transition:filter 300ms; }
  .geo-globe:hover { filter:drop-shadow(0 0 2em ${COLORS.accent}aa); }

  @keyframes logo-spin {
    from { transform:rotate(0deg); }
    to   { transform:rotate(360deg); }
  }
  @media (prefers-reduced-motion: no-preference) {
    .geo-globe { animation:logo-spin infinite 20s linear; }
  }

  @keyframes fadeUp {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0); }
  }
  @keyframes spin-ring {
    to { transform:rotate(360deg); }
  }

  .leaflet-popup-content-wrapper {
    background:#fff !important;
    border:1px solid ${COLORS.accent}33 !important;
    border-radius:10px !important;
    box-shadow:0 4px 20px ${COLORS.accent}22 !important;
  }
  .leaflet-popup-tip { background:#fff !important; }
`;