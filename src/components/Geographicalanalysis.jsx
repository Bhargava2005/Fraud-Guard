import React, { useState, useEffect, useRef, useCallback } from "react";
import { S, COLORS as C, globalStyles } from "./styles/GeographicalAnalysis.styles";

// ─── Helpers ──────────────────────────────────────────────────────────────────
function riskColor(risk, lo, hi, showH, showM, showS) {
  if (showH && risk >= hi) return C.highColor;
  if (showM && risk >= lo && risk < hi) return C.modColor;
  if (showS && risk < lo) return C.safeColor;
  return null;
}

function isValidPin(pin) {
  return (
    pin &&
    typeof pin.lat  === "number" && isFinite(pin.lat)  &&
    typeof pin.lng  === "number" && isFinite(pin.lng)  &&
    pin.lat >= -90  && pin.lat <= 90 &&
    pin.lng >= -180 && pin.lng <= 180
  );
}

// ── Compute bounding box center + appropriate zoom from a list of pins ──
// FIX 1: replaces STATE_VIEW constant entirely
function computeMapView(pins, state, zone, district) {
  if (!pins || pins.length === 0) {
    return { center: [20.59, 78.96], zoom: 5 };   // fallback: All India
  }

  const lats = pins.map(p => p.lat);
  const lngs = pins.map(p => p.lng);

  const minLat = Math.min(...lats), maxLat = Math.max(...lats);
  const minLng = Math.min(...lngs), maxLng = Math.max(...lngs);

  const centerLat = (minLat + maxLat) / 2;
  const centerLng = (minLng + maxLng) / 2;

  // Zoom by filter granularity
  const zoom = district && district !== "All Districts" ? 10
             : zone     && zone     !== "All Zones"     ? 8
             : state    && state    !== "All India"      ? 7
             : 5;

  return { center: [centerLat, centerLng], zoom };
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function DualRangeSlider({ lo, hi, onChange }) {
  const trackBg = `linear-gradient(to right,
    ${C.safeColor} 0%, ${C.safeColor} ${lo}%,
    ${C.modColor}  ${lo}%, ${C.modColor}  ${hi}%,
    ${C.highColor} ${hi}%, ${C.highColor} 100%)`;
  const loZ = lo >= hi - 1 ? 5 : 3;
  const hiZ = lo >= hi - 1 ? 3 : 5;

  return (
    <div style={{ position: "relative", height: 34, userSelect: "none", marginTop: 10, marginBottom: 6 }}>
      <div style={{ position: "absolute", top: "50%", left: 0, right: 0, height: 6, borderRadius: 4, transform: "translateY(-50%)", background: trackBg, pointerEvents: "none", boxShadow: "inset 0 1px 3px rgba(0,0,0,0.12)" }} />
      <input type="range" min={0} max={100} value={lo} className="geo-range"
        style={{ position: "absolute", width: "100%", top: "50%", transform: "translateY(-50%)", zIndex: loZ }}
        onChange={e => onChange([Math.min(+e.target.value, hi - 1), hi])} />
      <input type="range" min={0} max={100} value={hi} className="geo-range"
        style={{ position: "absolute", width: "100%", top: "50%", transform: "translateY(-50%)", zIndex: hiZ }}
        onChange={e => onChange([lo, Math.max(+e.target.value, lo + 1)])} />
    </div>
  );
}

function Checkbox({ checked, color, label, onChange }) {
  return (
    <label onClick={onChange} style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer", userSelect: "none", marginBottom: 9 }}>
      <div style={{ width: 18, height: 18, borderRadius: 4, flexShrink: 0, border: `2px solid ${color}`, background: checked ? color : "#fff", boxShadow: checked ? `0 0 0 3px ${color}28` : "none", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.15s" }}>
        {checked && <span style={{ fontSize: 11, color: "#fff", fontWeight: 700, lineHeight: 1 }}>✓</span>}
      </div>
      <span style={{ fontSize: 13, color: C.text }}>{label}</span>
    </label>
  );
}

function GeoSelect({ label, value, options, disabled, onChange }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <label style={S.fieldLabel}>{label}</label>
      <select value={value} disabled={disabled} onChange={e => onChange(e.target.value)} className="geo-select"
        style={{ width: "100%", padding: "8px 10px", fontSize: 13, border: `1px solid ${C.border}`, borderRadius: 7, background: disabled ? "#f9fafb" : C.pageBg, color: disabled ? C.muted : C.text, outline: "none", cursor: disabled ? "not-allowed" : "pointer", transition: "border-color 200ms" }}>
        {options.map(o => <option key={o}>{o}</option>)}
      </select>
      {/* FIX 2: show hint when zone has no data for selected state */}
      {disabled && label === "Zone" && (
        <span style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", marginTop: 3, display: "block" }}>
          No zone data available for this state
        </span>
      )}
    </div>
  );
}

// ════ MAIN COMPONENT ════
export default function GeographicalAnalysis() {
  const [state, setState]       = useState("All India");
  const [zone, setZone]         = useState("All Zones");
  const [district, setDistrict] = useState("All Districts");
  const [showHigh, setShowHigh] = useState(true);
  const [showMod,  setShowMod]  = useState(false);
  const [showSafe, setShowSafe] = useState(false);
  const [pending, setPending]   = useState([30, 60]);
  const [applied, setApplied]   = useState([30, 60]);

  const [pincodes,  setPincodes]  = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadStep,  setLoadStep]  = useState("🗺 Initialising map…");
  const [mapReady,  setMapReady]  = useState(false);
  const [stats, setStats]         = useState({ total: 0, high: 0, moderate: 0, safe: 0 });
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const mapDivRef    = useRef(null);
  const leafletMap   = useRef(null);
  const markersGroup = useRef(null);
  const mapInited    = useRef(false);

  const [leafletLoaded,       setLeafletLoaded]       = useState(!!window.L);
  const [allStates,           setAllStates]           = useState([]);
  const [availableZones,      setAvailableZones]      = useState([]);
  const [availableDistricts,  setAvailableDistricts]  = useState([]);

  // FIX 2: track whether zone data actually exists for selected state
  const [zoneAvailable, setZoneAvailable] = useState(true);

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  // ── Fetch states ──
  useEffect(() => {
    fetch("http://127.0.0.1:5000/get_states")
      .then(r => r.json())
      .then(setAllStates)
      .catch(e => console.error("Error fetching states:", e));
  }, []);

  // ── Fetch zones + districts when state changes ──
  useEffect(() => {
    if (state === "All India") {
      setAvailableZones([]);
      setAvailableDistricts([]);
      setZoneAvailable(false);
      return;
    }
    fetch(`http://127.0.0.1:5000/get_geo_lists?state=${encodeURIComponent(state)}`)
      .then(r => r.json())
      .then(data => {
        const zones = data.zones || [];
        setAvailableZones(zones);
        setAvailableDistricts(data.districts || []);

        // FIX 2: if backend returns empty zones list, zone is unavailable for this state
        setZoneAvailable(zones.length > 0);

        // reset dependent selections
        setZone("All Zones");
        setDistrict("All Districts");
      })
      .catch(e => console.error("Error fetching geo lists:", e));
  }, [state]);

  // ── Leaflet injection ──
  useEffect(() => {
    if (window.L) { setLeafletLoaded(true); return; }
    if (!document.getElementById("lf-css")) {
      const link = document.createElement("link");
      link.id = "lf-css"; link.rel = "stylesheet";
      link.href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css";
      document.head.appendChild(link);
    }
    const script = document.createElement("script");
    script.id = "lf-js";
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js";
    script.onload = () => setLeafletLoaded(true);
    document.head.appendChild(script);
  }, []);

  // ── Draw markers ──
  const drawMarkers = useCallback((pins, lo, hi, sH, sM, sS) => {
    const L = window.L;
    if (!L || !markersGroup.current) return;
    markersGroup.current.clearLayers();

    let hC = 0, mC = 0, sC = 0;

    pins.forEach(pin => {
      if (!isValidPin(pin)) return;
      const col = riskColor(pin.risk_percent, lo, hi, sH, sM, sS);
      if (!col) return;

      if      (pin.risk_percent >= hi) hC++;
      else if (pin.risk_percent >= lo) mC++;
      else                             sC++;

      L.circleMarker([pin.lat, pin.lng], {
        radius: 7, color: col, fillColor: col, fillOpacity: 0.82, weight: 1.5,
      }).bindPopup(`
        <div style="font-family:monospace;font-size:12px;line-height:1.9;color:${C.text}">
          <strong style="font-size:13px;color:${C.accent}">📍 ${pin.pincode}</strong><br/>
          Risk     : <strong style="color:${col}">${pin.risk_percent.toFixed(1)}%</strong><br/>
          District : ${pin.dist}<br/>
          State    : ${pin.st}<br/>
          Division : ${pin.dv}
        </div>
      `).addTo(markersGroup.current);
    });

    setStats({ total: pins.length, high: hC, moderate: mC, safe: sC });
  }, []);

  // ── Main data loader ──
  const loadData = async (_state, _zone, _district, _lo, _hi, _sH, _sM, _sS) => {
    setIsLoading(true);
    setMapReady(false);
    setLoadStep("📦 Loading data…");

    try {
      const activeTypes = [];
      if (_sH) activeTypes.push("high");
      if (_sM) activeTypes.push("moderate");
      if (_sS) activeTypes.push("safe");

      const params = new URLSearchParams({
        state:    _state    === "All India"     ? "" : _state,
        zone:     _zone     === "All Zones"     ? "" : _zone,
        district: _district === "All Districts" ? "" : _district,
        point1:   _lo,
        point2:   _hi,
        types:    activeTypes.join(","),
      });

      setLoadStep("🌐 Fetching pincode data from server…");
      const response = await fetch(`http://127.0.0.1:5000/get_pincodes?${params}`);
      const data     = await response.json();

      setLoadStep("📐 Processing data…");
      await sleep(150);

      // Flatten { high:[...], moderate:[...], safe:[...] } → single array
      const flatPins = Object.values(data)
        .flat()
        .map(item => ({
          ...item,
          lat: parseFloat(item.lat),
          lng: parseFloat(item.lang),   // API sends "lang"
        }))
        .filter(isValidPin);

      setLoadStep("📍 Placing markers on map…");
      await sleep(150);

      // FIX 1: compute center + zoom from actual pin coordinates
      const { center, zoom } = computeMapView(flatPins, _state, _zone, _district);
      if (leafletMap.current) {
        leafletMap.current.setView(center, zoom);
      }

      setPincodes(flatPins);
      drawMarkers(flatPins, _lo, _hi, _sH, _sM, _sS);

      setLoadStep("✅ Map ready!");
      await sleep(100);
      setIsLoading(false);
      setMapReady(true);

    } catch (error) {
      console.error("Error loading pincode data:", error);
      setLoadStep("❌ Failed to load data. Check console.");
      setIsLoading(false);
    }
  };

  // ── Init Leaflet once ──
  useEffect(() => {
    if (!leafletLoaded || mapInited.current || !mapDivRef.current) return;
    const L = window.L;
    mapInited.current = true;

    leafletMap.current = L.map(mapDivRef.current, {
      center: [20.59, 78.96], zoom: 5, zoomControl: false,
    });

    L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
      { attribution: "© OSM © CartoDB", subdomains: "abcd", maxZoom: 19 }
    ).addTo(leafletMap.current);

    L.control.zoom({ position: "bottomright" }).addTo(leafletMap.current);
    markersGroup.current = L.layerGroup().addTo(leafletMap.current);

    loadData("All India", "All Zones", "All Districts", 30, 60, true, false, false);
  }, [leafletLoaded]);

  const handleStateChange = v => {
    setState(v);
    setZone("All Zones");
    setDistrict("All Districts");
  };

  const handleZoneChange = v => {
    setZone(v);
    setDistrict("All Districts");
  };

  const handleSubmit = () => {
    setApplied(pending);
    loadData(state, zone, district, pending[0], pending[1], showHigh, showMod, showSafe);
  };

  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <div style={S.root}>
      <style>{globalStyles}</style>

      <header style={S.topBar}>
        <div style={S.topBarLeft}>
          <button className="geo-menu" style={S.menuBtn} onClick={() => setSidebarOpen(o => !o)}>☰</button>
          <span style={{ fontSize: 28, lineHeight: 1, cursor: "default" }}>🌍</span>
          <div>
            <div style={S.appTitle}>Geo Risk Analytics</div>
            <div style={{ fontSize: 11, color: C.muted, fontFamily: "monospace" }}>Pincode Risk Distribution Dashboard</div>
          </div>
        </div>
        <div style={S.statsRow}>
          {[
            { label: "Total Pins", value: stats.total,    color: C.accent    },
            { label: "High Risk",  value: stats.high,     color: C.highColor },
            { label: "Moderate",   value: stats.moderate, color: C.modColor  },
            { label: "Safe",       value: stats.safe,     color: C.safeColor },
          ].map(s => (
            <div key={s.label} style={{ ...S.statChip, borderColor: `${s.color}55` }}>
              <span style={{ ...S.statVal, color: s.color }}>{s.value}</span>
              <span style={S.statLabel}>{s.label}</span>
            </div>
          ))}
        </div>
      </header>

      <div style={S.body}>
        {sidebarOpen && (
          <aside style={S.sidebar}>

            <div style={S.card}>
              <p style={S.cardTitle}>📍 Location Filters</p>
              <GeoSelect
                label="State"
                value={state}
                options={["All India", ...allStates]}
                onChange={handleStateChange}
              />
              {/* FIX 2: zone disabled when no zone data exists for selected state */}
              <GeoSelect
                label="Zone"
                value={zone}
                options={["All Zones", ...availableZones]}
                disabled={state === "All India" || !zoneAvailable}
                onChange={handleZoneChange}
              />
              <GeoSelect
                label="District"
                value={district}
                options={["All Districts", ...availableDistricts]}
                disabled={state === "All India"}
                onChange={setDistrict}
              />
            </div>

            <div style={S.card}>
              <p style={S.cardTitle}>⚠️ Risk Indicators</p>
              <Checkbox checked={showHigh} color={C.highColor} label="🔴 High Risk"     onChange={() => setShowHigh(v => !v)} />
              <Checkbox checked={showMod}  color={C.modColor}  label="🟡 Moderate Risk" onChange={() => setShowMod(v => !v)}  />
              <Checkbox checked={showSafe} color={C.safeColor} label="🟢 Safe Area"     onChange={() => setShowSafe(v => !v)} />

              <div style={{ marginTop: 14 }}>
                <p style={{ ...S.fieldLabel, marginBottom: 2 }}>Risk Thresholds %</p>
                <DualRangeSlider lo={pending[0]} hi={pending[1]} onChange={setPending} />
                <div style={{ display: "flex", flexDirection: "column", gap: 5, marginTop: 6 }}>
                  {[
                    { bg: C.safeBg, color: C.safeColor, border: C.safeBdr, text: `🟢 Safe : 0 – ${pending[0]}%`             },
                    { bg: C.modBg,  color: C.modColor,  border: C.modBdr,  text: `🟡 Mod  : ${pending[0]} – ${pending[1]}%` },
                    { bg: C.highBg, color: C.highColor, border: C.highBdr, text: `🔴 High : ${pending[1]} – 100%`           },
                  ].map(p => (
                    <span key={p.text} style={{ display: "block", fontSize: 11, fontFamily: "monospace", fontWeight: 500, padding: "4px 9px", borderRadius: 6, background: p.bg, color: p.color, border: `1px solid ${p.border}` }}>
                      {p.text}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <button className="geo-submit" style={S.submitBtn} onClick={handleSubmit}>
              🔄 Show / Update Map
            </button>

            {/* <div style={S.card}>
              <p style={S.cardTitle}>📋 Legend</p>
              {[
                { color: C.highColor, bg: C.highBg, label: `High Risk ≥ ${applied[1]}%`          },
                { color: C.modColor,  bg: C.modBg,  label: `Mod  ${applied[0]} – ${applied[1]}%` },
                { color: C.safeColor, bg: C.safeBg, label: `Safe < ${applied[0]}%`               },
              ].map(l => (
                <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 7, padding: "6px 9px", borderRadius: 7, background: l.bg, borderLeft: `3px solid ${l.color}` }}>
                  <div style={{ width: 10, height: 10, borderRadius: "50%", background: l.color, flexShrink: 0 }} />
                  <span style={{ fontSize: 12, color: "#374151", fontFamily: "monospace" }}>{l.label}</span>
                </div>
              ))}
            </div> */}

          </aside>
        )}

        <main style={S.mapArea}>
          {isLoading && (
            <div style={S.loadOverlay}>
              <div style={S.loadCard}>
                <div style={S.spinner} />
                <div style={{ fontSize: 15, fontWeight: 600, color: C.text, marginBottom: 6 }}>{loadStep}</div>
                <div style={{ fontSize: 12, color: C.muted, fontFamily: "monospace" }}>Processing geo-spatial data…</div>
              </div>
            </div>
          )}
          <div ref={mapDivRef} style={{ width: "100%", height: "100%", background: C.pageBg }} />
          {mapReady && !isLoading && (
            <div style={S.readyBadge}>✅ Map Ready · {pincodes.length} pincodes plotted</div>
          )}
        </main>
      </div>
    </div>
  );
}