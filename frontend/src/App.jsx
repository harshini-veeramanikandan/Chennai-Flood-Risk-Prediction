import { useState, useEffect } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { CloudRain, Droplets, AlertTriangle, TrendingUp, Database, Target, Calendar, Sparkles, ExternalLink } from 'lucide-react'
const API_BASE = 'http://localhost:8000'
const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December']

function App() {
  const [history, setHistory] = useState([])
  const [threshold, setThreshold] = useState(0)
  const [importances, setImportances] = useState([])
  const [climatology, setClimatology] = useState([])
  const [rainfallToday, setRainfallToday] = useState(20)
  const [rainfall3day, setRainfall3day] = useState(60)
  const [rainfall7day, setRainfall7day] = useState(100)
  const [month, setMonth] = useState(10)
  const [prediction, setPrediction] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch(`${API_BASE}/history`).then(r => r.json()).then(data => {
      setHistory(data.dates.map((d, i) => ({ date: d, rainfall: data.rainfall_3day_sum[i] })))
      setThreshold(data.threshold)
    })
    fetch(`${API_BASE}/feature-importance`).then(r => r.json()).then(data => setImportances(data.features))
    fetch(`${API_BASE}/climatology`).then(r => r.json()).then(data => setClimatology(data.months))
  }, [])

  useEffect(() => {
    setLoading(true)
    const payload = {
      rainfall_mm: rainfallToday,
      rainfall_3day_sum: rainfall3day,
      rainfall_7day_sum: rainfall7day,
      month: month,
    }
    const timer = setTimeout(() => {
      Promise.all([
        fetch(`${API_BASE}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).then(r => r.json()),
        fetch(`${API_BASE}/explain`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).then(r => r.json()),
      ]).then(([pred, expl]) => { setPrediction(pred); setExplanation(expl); setLoading(false) })
    }, 250)
    return () => clearTimeout(timer)
  }, [rainfallToday, rainfall3day, rainfall7day, month])

  const riskColor = prediction?.risk_label === 'HIGH' ? '#EA4335' : prediction?.risk_label === 'MODERATE' ? '#F4B400' : '#2ECC71'
  const proba = prediction ? prediction.probability * 100 : 0
  const maxClim = Math.max(...climatology.map(c => c.avg_rainfall), 1)

  return (
    <div className="min-h-screen bg-[#050D18] text-slate-200 font-sans overflow-x-hidden">

      {/* ---------- NAV ---------- */}
      <nav className="flex items-center justify-between px-8 md:px-16 py-6 border-b border-white/5 sticky top-0 bg-[#050D18]/80 backdrop-blur-md z-50">
        <div className="flex items-center gap-2.5">
          <CloudRain className="text-cyan-400" size={26} />
          <span className="font-bold text-white text-lg tracking-tight">Chennai Flood Risk</span>
        </div>
        <div className="hidden md:flex items-center gap-8 text-sm text-slate-400">
          <span>Predict</span><span>Insights</span><span>Methodology</span>
        </div>
        <a href="#" className="flex items-center gap-2 text-xs text-slate-400 border border-white/10 rounded-full px-4 py-2 hover:border-cyan-400/50 hover:text-cyan-400 transition-colors">
          <xternalLink size={14} /> View source
        </a>
      </nav>

      {/* ---------- HERO ---------- */}
      <div className="relative px-8 pt-28 pb-24 text-center max-w-4xl mx-auto"
           style={{ backgroundImage: 'radial-gradient(circle at 50% 0%, rgba(31,111,235,0.25), transparent 55%)' }}>
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-cyan-400/10 border border-cyan-400/20 text-cyan-300 text-xs font-medium mb-8">
          <Sparkles size={13} /> 30 years of rainfall data · XGBoost · Live SHAP explainability
        </div>
        <h1 className="text-6xl md:text-7xl font-extrabold text-white tracking-tighter leading-[1.05]">
          Predict floods<br />before they <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">happen.</span>
        </h1>
        <p className="text-slate-400 text-lg mt-8 max-w-xl mx-auto leading-relaxed">
          A machine learning system trained on three decades of Chennai rainfall records, forecasting next-day flood risk with explainable, transparent predictions.
        </p>
      </div>

      {/* ---------- STATS BAR ---------- */}
      <div className="max-w-6xl mx-auto px-6 -mt-4 pb-24 grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon={<Database size={18} />} value="30+" label="Years of rainfall data" />
        <StatCard icon={<Target size={18} />} value="86%" label="Flood-day recall" />
        <StatCard icon={<Calendar size={18} />} value="11,304" label="Daily records analyzed" />
        <StatCard icon={<TrendingUp size={18} />} value="62.3mm" label="Flood-risk threshold" />
      </div>

      {/* ---------- RAINFALL HISTORY ---------- */}
      <Section eyebrow="Historical record" title="Three decades of Chennai rainfall, at a glance">
        <div className="bg-white/[0.03] backdrop-blur border border-white/10 rounded-3xl p-8 shadow-2xl">
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={history}>
              <defs>
                <linearGradient id="rainGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#4FC3F7" stopOpacity={0.45} />
                  <stop offset="100%" stopColor="#4FC3F7" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="date" tick={false} axisLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, fontSize: 12 }} />
              <ReferenceLine y={threshold} stroke="#F4B400" strokeDasharray="4 4"
                             label={{ value: 'Flood threshold', fill: '#F4B400', fontSize: 11, position: 'insideTopRight' }} />
              <Area type="monotone" dataKey="rainfall" stroke="#4FC3F7" strokeWidth={2.5} fill="url(#rainGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Section>

      {/* ---------- MONSOON CALENDAR ---------- */}
      <Section eyebrow="Seasonality" title="The monsoon calendar" subtitle="Average 3-day cumulative rainfall by month, across all 30 years — the northeast monsoon (Oct–Dec) dominates Chennai's flood risk.">
        <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
          {climatology.map((c) => {
            const intensity = c.avg_rainfall / maxClim
            return (
              <div key={c.month}
                   className="rounded-2xl p-5 border border-white/10 flex flex-col items-center justify-center gap-2 transition-transform hover:scale-105"
                   style={{ background: `rgba(79,195,247,${0.06 + intensity * 0.35})` }}>
                <span className="text-xs uppercase tracking-widest text-slate-400">{c.month}</span>
                <span className="text-2xl font-bold text-white">{c.avg_rainfall}</span>
                <span className="text-[10px] text-slate-500">mm avg</span>
              </div>
            )
          })}
        </div>
      </Section>

      {/* ---------- PREDICTION ---------- */}
      <Section eyebrow="Try it live" title="Set a rainfall scenario, get an instant forecast">
        <div className="grid lg:grid-cols-5 gap-8">
          <div className="lg:col-span-3 bg-white/[0.03] backdrop-blur border border-white/10 rounded-3xl p-8 shadow-2xl space-y-7">
            <div className="flex items-center gap-2">
              <Droplets size={18} className="text-cyan-400" />
              <span className="text-sm uppercase tracking-widest text-slate-400 font-medium">Scenario inputs</span>
            </div>
            <SliderField label="Rainfall today (mm)" value={rainfallToday} max={300} onChange={setRainfallToday} />
            <SliderField label="3-day cumulative rainfall (mm)" value={rainfall3day} max={500} onChange={setRainfall3day} />
            <SliderField label="7-day cumulative rainfall (mm)" value={rainfall7day} max={700} onChange={setRainfall7day} />
            <div>
              <span className="text-sm text-slate-300 block mb-2">Month</span>
              <select value={month} onChange={e => setMonth(Number(e.target.value))}
                      className="w-full bg-slate-900 border border-white/10 rounded-xl px-4 py-3 text-sm text-slate-200">
                {MONTHS.map((m, i) => <option key={i} value={i + 1}>{m}</option>)}
              </select>
            </div>
          </div>

          <div className="lg:col-span-2 bg-white/[0.03] backdrop-blur border border-white/10 rounded-3xl p-8 shadow-2xl flex flex-col items-center justify-center">
            <span className="text-sm uppercase tracking-widest text-slate-400 font-medium self-start mb-6">Forecast</span>
            <div className="relative w-52 h-52 rounded-full flex items-center justify-center"
                 style={{ background: `conic-gradient(${riskColor} ${proba * 3.6}deg, #16233699 0deg)` }}>
              <div className="absolute w-40 h-40 bg-[#0a1421] rounded-full flex flex-col items-center justify-center">
                <span className="text-4xl font-extrabold text-white">{proba.toFixed(1)}%</span>
                <span className="text-[11px] text-slate-500 mt-1">flood probability</span>
              </div>
            </div>
            <div className="mt-6 px-5 py-2 rounded-full text-sm font-bold tracking-wide"
                 style={{ background: `${riskColor}22`, color: riskColor }}>
              {loading ? 'CALCULATING…' : `${prediction?.risk_label || '—'} RISK — TOMORROW`}
            </div>
          </div>
        </div>
      </Section>

      {/* ---------- LIVE SHAP EXPLANATION ---------- */}
      <Section eyebrow="Explainable AI" title="Why did the model say that?" subtitle="Computed fresh for your exact scenario using SHAP — not a generic chart. Positive bars push risk up, negative bars push it down.">
        <div className="bg-white/[0.03] backdrop-blur border border-white/10 rounded-3xl p-8 shadow-2xl">
          {explanation ? (
            <div className="space-y-4">
              {explanation.contributions.map((c) => {
                const isPositive = c.contribution >= 0
                const width = Math.min(Math.abs(c.contribution) * 100, 100)
                return (
                  <div key={c.feature} className="flex items-center gap-4">
                    <span className="w-44 text-sm text-slate-300 shrink-0">{c.feature.replace(/_/g, ' ')}</span>
                    <div className="flex-1 h-8 bg-slate-900/50 rounded-lg relative overflow-hidden flex items-center">
                      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700" />
                      <div className="h-6 rounded-md transition-all duration-300"
                           style={{
                             width: `${width / 2}%`,
                             marginLeft: isPositive ? '50%' : `${50 - width / 2}%`,
                             background: isPositive ? '#EA433599' : '#2ECC7199',
                           }} />
                    </div>
                    <span className="w-16 text-right text-xs font-mono text-slate-400">{c.value.toFixed(1)}</span>
                  </div>
                )
              })}
              <p className="text-xs text-slate-500 pt-2">Red = pushes flood risk up · Green = pushes flood risk down · Base rate: {(explanation.base_value * 100).toFixed(1)}%</p>
            </div>
          ) : (
            <p className="text-slate-500 text-sm">Move a slider above to see live explanation…</p>
          )}
        </div>
      </Section>

      {/* ---------- GLOBAL FEATURE IMPORTANCE ---------- */}
      <Section eyebrow="Model internals" title="What the model learned overall">
        <div className="bg-white/[0.03] backdrop-blur border border-white/10 rounded-3xl p-8 shadow-2xl space-y-5">
          {importances.map((f) => (
            <div key={f.name}>
              <div className="flex justify-between text-sm text-slate-300 mb-2">
                <span className="capitalize">{f.name.replace(/_/g, ' ')}</span>
                <span className="font-mono text-cyan-400">{(f.importance * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-500"
                     style={{ width: `${f.importance * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* ---------- METHODOLOGY / LIMITATIONS ---------- */}
      <Section eyebrow="Transparency" title="Methodology & limitations">
        <div className="grid md:grid-cols-2 gap-6">
          <InfoCard title="How the target label works">
            Flood risk is a rainfall-intensity proxy — days in the top 5% of historical 3-day cumulative rainfall are labeled high-risk. This is not confirmed ground-truth flood event data.
          </InfoCard>
          <InfoCard title="Why recall matters more here">
            The model is tuned to prioritize catching real flood-risk days over avoiding false alarms — missing a flood warning is far more costly than one extra alert.
          </InfoCard>
          <InfoCard title="Data leakage, caught and fixed">
            An earlier version of this model accidentally used same-day features to predict same-day risk, producing artificially perfect scores. The final model forecasts strictly using only past information.
          </InfoCard>
          <InfoCard title="Future scope">
            Zone-level prediction using Chennai's expanded rain-gauge network, real flood event ground-truth data, and elevation/drainage geospatial features are natural next steps.
          </InfoCard>
        </div>
      </Section>

      <footer className="text-center text-xs text-slate-600 py-12 border-t border-white/5 mt-12">
        Built with Python, XGBoost, SHAP, FastAPI, and React · Chennai rainfall data via IMD / TNDMA (1993–2023)
      </footer>
    </div>
  )
}

function Section({ eyebrow, title, subtitle, children }) {
  return (
    <div className="max-w-6xl mx-auto px-6 py-16">
      <div className="mb-10">
        <span className="text-xs uppercase tracking-widest text-cyan-400 font-semibold">{eyebrow}</span>
        <h2 className="text-3xl md:text-4xl font-bold text-white mt-2 tracking-tight">{title}</h2>
        {subtitle && <p className="text-slate-400 mt-3 max-w-2xl">{subtitle}</p>}
      </div>
      {children}
    </div>
  )
}

function StatCard({ icon, value, label }) {
  return (
    <div className="bg-white/[0.03] backdrop-blur border border-white/10 rounded-2xl p-6 flex flex-col gap-2">
      <div className="text-cyan-400">{icon}</div>
      <span className="text-2xl md:text-3xl font-extrabold text-white">{value}</span>
      <span className="text-xs text-slate-400">{label}</span>
    </div>
  )
}

function InfoCard({ title, children }) {
  return (
    <div className="bg-white/[0.03] backdrop-blur border border-white/10 rounded-2xl p-6">
      <h3 className="text-white font-semibold mb-2">{title}</h3>
      <p className="text-sm text-slate-400 leading-relaxed">{children}</p>
    </div>
  )
}

function SliderField({ label, value, max, onChange }) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-2">
        <span className="text-slate-300">{label}</span>
        <span className="text-cyan-400 font-mono text-xs">{value} mm</span>
      </div>
      <input type="range" min={0} max={max} value={value} onChange={e => onChange(Number(e.target.value))}
             className="w-full accent-cyan-400" />
    </div>
  )
}

export default App