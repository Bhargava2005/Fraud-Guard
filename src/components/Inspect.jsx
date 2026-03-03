import React, { useState } from 'react';
import { Search, ShieldAlert, ShieldCheck, User, Cpu, Truck, ShoppingCart, Package, Store, AlertTriangle, Info, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';

// ── Match existing app color tokens ──────────────────────────
const C = {
  pageBg:   "#f4f6fb",
  cardBg:   "#ffffff",
  accent:   "#646cff",
  accentDk: "#535bf2",
  cyan:     "#61dafb",
  text:     "#213547",
  muted:    "#888888",
  border:   "#e2e8f0",
  high:  { color: "#dc2626", bg: "#fef2f2", border: "#fca5a5" },
  mod:   { color: "#d97706", bg: "#fffbeb", border: "#fcd34d" },
  safe:  { color: "#16a34a", bg: "#f0fdf4", border: "#86efac" },
};

// ── Risk level helpers ────────────────────────────────────────
function riskLevel(pct) {
  if (pct >= 65) return 'high';
  if (pct >= 35) return 'mod';
  return 'safe';
}
function riskTheme(pct) {
  const lvl = riskLevel(pct);
  return lvl === 'high' ? C.high : lvl === 'mod' ? C.mod : C.safe;
}
function riskLabel(pct) {
  const lvl = riskLevel(pct);
  return lvl === 'high' ? 'HIGH RISK' : lvl === 'mod' ? 'MODERATE' : 'LOW RISK';
}

// ── Animated arc / donut gauge ────────────────────────────────
function Gauge({ pct, size = 160, stroke = 14 }) {
  const theme  = riskTheme(pct);
  const r      = (size - stroke) / 2;
  const circ   = 2 * Math.PI * r;
  const dash   = (pct / 100) * circ;
  const cx     = size / 2;
  const cy     = size / 2;

  return (
    <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
      {/* track */}
      <circle cx={cx} cy={cy} r={r} fill="none"
        stroke={theme.border} strokeWidth={stroke} />
      {/* fill */}
      <circle cx={cx} cy={cy} r={r} fill="none"
        stroke={theme.color} strokeWidth={stroke}
        strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round"
        style={{ transition: 'stroke-dasharray 1s cubic-bezier(.4,0,.2,1)' }}
      />
    </svg>
  );
}

// ── Tooltip on hover ──────────────────────────────────────────
function Tooltip({ children, tip }) {
  const [show, setShow] = useState(false);
  return (
    <span style={{ position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <span style={{
          position: 'absolute', bottom: '130%', left: '50%',
          transform: 'translateX(-50%)',
          background: C.text, color: '#fff',
          fontSize: 11, borderRadius: 6, padding: '5px 10px',
          whiteSpace: 'nowrap', zIndex: 999,
          boxShadow: '0 2px 12px rgba(0,0,0,0.18)',
          pointerEvents: 'none',
          animation: 'fadeUp 0.15s ease',
        }}>{tip}</span>
      )}
    </span>
  );
}

// ── Mini horizontal bar ───────────────────────────────────────
function MiniBar({ pct }) {
  const theme = riskTheme(pct);
  return (
    <div style={{
      height: 6, borderRadius: 99,
      background: theme.bg, border: `1px solid ${theme.border}`,
      overflow: 'hidden', width: '100%',
    }}>
      <div style={{
        height: '100%', width: `${pct}%`,
        background: theme.color, borderRadius: 99,
        transition: 'width 0.9s cubic-bezier(.4,0,.2,1)',
      }} />
    </div>
  );
}

// ── Sub-risk card (for the 5 smaller risks) ───────────────────
const RISK_META = {
  device_risk:    { icon: Cpu,          label: 'Device',    tip: 'Emulator, rooted, VPN, failed logins, geo anomalies' },
  logistics_risk: { icon: Truck,        label: 'Logistics', tip: 'Delivery attempts, delays, tamper route, weight mismatch' },
  order_risk:     { icon: ShoppingCart, label: 'Order',     tip: 'Order value, velocity, address mismatch, payment method' },
  product_risk:   { icon: Package,      label: 'Product',   tip: 'Counterfeit risk, fraud return rate, discount, category' },
  seller_risk:    { icon: Store,        label: 'Seller',    tip: 'Dispute rate, wrong items, negative feedback, verification' },
};

function SubRiskCard({ riskKey, pct }) {
  const meta  = RISK_META[riskKey];
  const theme = riskTheme(pct);
  const Icon  = meta.icon;
  const [open, setOpen] = useState(false);

  return (
    <div style={{
      background: C.cardBg,
      border: `1.5px solid ${open ? theme.border : C.border}`,
      borderRadius: 12,
      padding: '14px 16px',
      transition: 'border-color 0.2s, box-shadow 0.2s',
      boxShadow: open ? `0 2px 16px ${theme.color}18` : '0 1px 4px rgba(0,0,0,0.05)',
      cursor: 'pointer',
    }} onClick={() => setOpen(o => !o)}>

      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{
          width: 34, height: 34, borderRadius: 8,
          background: theme.bg, border: `1px solid ${theme.border}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
        }}>
          <Icon size={16} color={theme.color} />
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: C.text, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {meta.label}
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{
                fontSize: 11, fontWeight: 700, color: theme.color,
                background: theme.bg, border: `1px solid ${theme.border}`,
                borderRadius: 6, padding: '2px 8px',
              }}>{riskLabel(pct)}</span>
              <span style={{ fontSize: 14, fontWeight: 700, color: theme.color, fontFamily: 'monospace' }}>
                {pct.toFixed(1)}%
              </span>
              {open ? <ChevronUp size={14} color={C.muted} /> : <ChevronDown size={14} color={C.muted} />}
            </div>
          </div>
          <MiniBar pct={pct} />
        </div>
      </div>

      {/* Expanded detail */}
      {open && (
        <div style={{
          marginTop: 12, paddingTop: 12,
          borderTop: `1px dashed ${theme.border}`,
          fontSize: 12, color: C.muted, lineHeight: 1.6,
          animation: 'fadeUp 0.2s ease',
        }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6 }}>
            <Info size={13} color={C.accent} style={{ marginTop: 2, flexShrink: 0 }} />
            <span><strong style={{ color: C.text }}>Signals analysed:</strong> {meta.tip}</span>
          </div>
          <div style={{
            marginTop: 8, padding: '8px 12px',
            background: theme.bg, borderRadius: 8,
            border: `1px solid ${theme.border}`,
            color: theme.color, fontWeight: 600, fontSize: 12,
          }}>
            {pct >= 65
              ? '⚠ High probability of fraudulent activity detected in this dimension.'
              : pct >= 35
              ? '⚡ Moderate anomalies found — recommend manual review.'
              : '✓ No significant risk signals detected in this dimension.'}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main Inspect Component ────────────────────────────────────
export default function Inspect() {
  const [orderId, setOrderId]   = useState('');
  const [loading, setLoading]   = useState(false);
  const [result,  setResult]    = useState(null);
  const [error,   setError]     = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!orderId.trim()) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await fetch(`http://localhost:5000/predict/fraud-risk?order_id=${orderId.trim()}`);
      const data = await res.json();

      if (!res.ok) {
        setError(data.error || 'Something went wrong');
      } else {
        setResult(data);
      }
    } catch (err) {
      setError('Cannot reach backend. Make sure Flask is running on port 5000.');
    } finally {
      setLoading(false);
    }
  }

  const finalPct      = result ? result.final_fraud_probability : null;
  const finalTheme    = finalPct !== null ? riskTheme(finalPct) : null;
  const custPct       = result ? result.risk_scores.customer_risk : null;
  const custTheme     = custPct !== null ? riskTheme(custPct) : null;

  const subRisks = result
    ? ['device_risk','logistics_risk','order_risk','product_risk','seller_risk']
    : [];

  return (
    <div style={{
      minHeight: '100%',
      background: C.pageBg,
      padding: '32px 24px',
      fontFamily: "'Outfit', sans-serif",
      color: C.text,
    }}>

      {/* ── Page title ── */}
      <div style={{ marginBottom: 28 }}>
        <h2 style={{ fontSize: 28, fontWeight: 700, color: C.text, marginBottom: 6, letterSpacing: '-0.01em' }}>
          Order Fraud Inspector
        </h2>
        <p style={{ fontSize: 14, color: C.muted, maxWidth: 520 }}>
          Enter an Order ID to run all six risk models and view a full fraud probability breakdown.
        </p>
      </div>

      {/* ── Search bar ── */}
      <form onSubmit={handleSubmit} style={{
        display: 'flex', gap: 10, maxWidth: 480, marginBottom: 36,
      }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <Search size={16} color={C.muted} style={{
            position: 'absolute', left: 12, top: '50%',
            transform: 'translateY(-50%)', pointerEvents: 'none',
          }} />
          <input
            type="text"
            placeholder="Enter Order ID e.g. 482931"
            value={orderId}
            onChange={e => setOrderId(e.target.value)}
            style={{
              width: '100%', paddingLeft: 38, paddingRight: 14,
              paddingTop: 11, paddingBottom: 11,
              border: `1.5px solid ${C.border}`, borderRadius: 10,
              fontSize: 14, color: C.text, background: C.cardBg,
              outline: 'none', fontFamily: 'inherit',
              boxShadow: '0 1px 4px rgba(0,0,0,0.05)',
              transition: 'border-color 0.2s, box-shadow 0.2s',
            }}
            onFocus={e => {
              e.target.style.borderColor = C.accent;
              e.target.style.boxShadow = `0 0 0 3px ${C.accent}22`;
            }}
            onBlur={e => {
              e.target.style.borderColor = C.border;
              e.target.style.boxShadow = '0 1px 4px rgba(0,0,0,0.05)';
            }}
          />
        </div>
        <button type="submit" disabled={loading || !orderId.trim()}
          className="geo-submit"
          style={{
            background: `linear-gradient(135deg, ${C.accent}, ${C.accentDk})`,
            border: 'none', borderRadius: 10, color: '#fff',
            padding: '11px 22px', fontSize: 13, fontWeight: 600,
            cursor: loading || !orderId.trim() ? 'not-allowed' : 'pointer',
            opacity: !orderId.trim() ? 0.5 : 1,
            display: 'flex', alignItems: 'center', gap: 7,
            letterSpacing: '0.03em', fontFamily: 'inherit',
          }}>
          {loading
            ? <><Loader2 size={15} style={{ animation: 'spin-ring 0.8s linear infinite' }} /> Scanning…</>
            : <><Search size={15} /> Inspect</>}
        </button>
      </form>

      {/* ── Error ── */}
      {error && (
        <div style={{
          maxWidth: 560, padding: '14px 18px', borderRadius: 10,
          background: C.high.bg, border: `1.5px solid ${C.high.border}`,
          color: C.high.color, fontSize: 13, fontWeight: 500,
          display: 'flex', alignItems: 'center', gap: 8,
          marginBottom: 24, animation: 'fadeUp 0.2s ease',
        }}>
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {/* ── Results ── */}
      {result && (
        <div style={{ animation: 'fadeUp 0.35s ease' }}>

          {/* ── ID meta strip ── */}
          <div style={{
            display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 24,
          }}>
            {[
              { label: 'Order ID',    val: result.order_id },
              { label: 'Customer ID', val: result.customer_id },
              { label: 'Product ID',  val: result.product_id },
              { label: 'Seller ID',   val: result.seller_id },
            ].map(({ label, val }) => (
              <div key={label} style={{
                background: C.cardBg,
                border: `1px solid ${C.border}`,
                borderRadius: 8, padding: '6px 14px',
                fontSize: 12,
              }}>
                <span style={{ color: C.muted, marginRight: 6 }}>{label}</span>
                <span style={{ fontWeight: 700, fontFamily: 'monospace', color: C.accent }}>{val}</span>
              </div>
            ))}
          </div>

          {/* ── Top row: Final score + Customer risk ── */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
            gap: 20, marginBottom: 20,
          }}>

            {/* Final fraud probability — hero card */}
            <div style={{
              background: finalTheme.bg,
              border: `2px solid ${finalTheme.border}`,
              borderRadius: 16, padding: '28px 24px',
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              boxShadow: `0 4px 24px ${finalTheme.color}18`,
              gridColumn: 'span 1',
            }}>
              <div style={{ position: 'relative', marginBottom: 12 }}>
                <Gauge pct={finalPct} size={150} stroke={13} />
                <div style={{
                  position: 'absolute', inset: 0,
                  display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center',
                  transform: 'rotate(0deg)',
                }}>
                  {result.fraud_alert
                    ? <ShieldAlert size={22} color={finalTheme.color} />
                    : <ShieldCheck size={22} color={finalTheme.color} />}
                  <span style={{
                    fontSize: 26, fontWeight: 800,
                    color: finalTheme.color, fontFamily: 'monospace',
                    lineHeight: 1.1, marginTop: 2,
                  }}>{finalPct.toFixed(1)}%</span>
                  <span style={{ fontSize: 9, color: finalTheme.color, letterSpacing: '0.08em', fontWeight: 600 }}>
                    FRAUD PROB.
                  </span>
                </div>
              </div>

              <div style={{
                fontSize: 13, fontWeight: 700, color: finalTheme.color,
                background: '#fff', border: `1.5px solid ${finalTheme.border}`,
                borderRadius: 20, padding: '4px 16px', marginBottom: 6,
                letterSpacing: '0.06em',
              }}>{riskLabel(finalPct)}</div>

              <div style={{
                fontSize: 11, color: finalTheme.color, textAlign: 'center',
                maxWidth: 200, lineHeight: 1.5,
              }}>
                {result.fraud_alert
                  ? '🚨 Fraud alert triggered — immediate review recommended.'
                  : '✓ Transaction appears within normal risk bounds.'}
              </div>
            </div>

            {/* Customer risk — primary entity card */}
            <div style={{
              background: custTheme.bg,
              border: `2px solid ${custTheme.border}`,
              borderRadius: 16, padding: '28px 24px',
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center',
              boxShadow: `0 4px 24px ${custTheme.color}18`,
            }}>
              <div style={{
                width: 56, height: 56, borderRadius: '50%',
                background: '#fff', border: `2px solid ${custTheme.border}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                marginBottom: 14, boxShadow: `0 2px 12px ${custTheme.color}22`,
              }}>
                <User size={24} color={custTheme.color} />
              </div>

              <div style={{
                fontSize: 11, fontWeight: 600, color: custTheme.color,
                textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 4,
              }}>Customer Risk</div>

              <div style={{
                fontSize: 44, fontWeight: 800, color: custTheme.color,
                fontFamily: 'monospace', lineHeight: 1, marginBottom: 8,
              }}>{custPct.toFixed(1)}<span style={{ fontSize: 20 }}>%</span></div>

              <Tooltip tip="Based on: account age, return rate, COD %, linked accounts, VPN usage, fraud history">
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 5,
                  fontSize: 11, color: custTheme.color, cursor: 'help',
                  background: '#fff', borderRadius: 20,
                  padding: '4px 12px', border: `1px solid ${custTheme.border}`,
                }}>
                  <Info size={11} /> Hover for signals
                </div>
              </Tooltip>

              <div style={{ width: '100%', marginTop: 16 }}>
                <MiniBar pct={custPct} />
              </div>

              <div style={{
                marginTop: 10, padding: '8px 14px',
                background: '#fff', borderRadius: 8,
                border: `1px solid ${custTheme.border}`,
                fontSize: 11, color: custTheme.color, fontWeight: 600,
                textAlign: 'center', width: '100%',
              }}>
                {riskLabel(custPct)} · Customer #{result.customer_id}
              </div>
            </div>
          </div>

          {/* ── 5 sub-risk cards ── */}
          <div style={{ marginBottom: 8 }}>
            <div style={{
              fontSize: 11, fontWeight: 600, color: C.accent,
              letterSpacing: '0.08em', textTransform: 'uppercase',
              marginBottom: 12, fontFamily: 'monospace',
            }}>
              DIMENSION BREAKDOWN — click any card to expand
            </div>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: 12,
            }}>
              {subRisks.map(key => (
                <SubRiskCard key={key} riskKey={key} pct={result.risk_scores[key]} />
              ))}
            </div>
          </div>

        </div>
      )}

      {/* ── Empty state ── */}
      {!result && !loading && !error && (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', padding: '60px 0',
          color: C.muted, textAlign: 'center',
        }}>
          <div style={{
            width: 72, height: 72, borderRadius: '50%',
            background: `${C.accent}11`,
            border: `2px dashed ${C.accent}44`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            marginBottom: 16,
          }}>
            <Search size={28} color={`${C.accent}88`} />
          </div>
          <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6, color: C.text }}>
            No order inspected yet
          </div>
          <div style={{ fontSize: 13, maxWidth: 320 }}>
            Enter an Order ID above to run the full FraudGuard risk pipeline across all six dimensions.
          </div>
        </div>
      )}

    </div>
  );
}