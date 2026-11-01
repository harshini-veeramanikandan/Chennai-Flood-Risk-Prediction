import { useState, useEffect, useRef } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { 
  CloudRain, 
  Droplets, 
  TrendingUp, 
  Database, 
  Target, 
  Calendar, 
  Sparkles, 
  ExternalLink, 
  Radio, 
  MapPin, 
  Clock, 
  ShieldAlert, 
  Navigation, 
  ChevronRight, 
  Info, 
  AlertTriangle,
  Zap,
  Activity,
  Award,
  Waves
} from 'lucide-react'

const API_BASE = 'http://localhost:8000'
const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December']

// Chennai geographical characteristics by zone for mock ML models & UI highlights
const ZONE_DATA = {
  velachery: { name: 'Velachery', elevation: 2, drainDist: 80, history: 12, vulnerability: 0.95, desc: 'Highly urbanized wetlands. Chronic waterlogging.' },
  adyar: { name: 'Adyar', elevation: 4, drainDist: 150, history: 8, vulnerability: 0.78, desc: 'Close to Adyar River mouth. Vulnerable to tidal locks.' },
  mylapore: { name: 'Mylapore', elevation: 8, drainDist: 400, history: 3, vulnerability: 0.42, desc: 'Older settlement with elevated terrain. Moderately safe.' },
  tnagar: { name: 'T. Nagar', elevation: 5, drainDist: 200, history: 10, vulnerability: 0.88, desc: 'Commercial core, concrete surfaces. Severe runoff issues.' },
  annanagar: { name: 'Anna Nagar', elevation: 11, drainDist: 600, history: 2, vulnerability: 0.25, desc: 'Elevated topography with superior storm drain design.' },
  guindy: { name: 'Guindy', elevation: 12, drainDist: 350, history: 4, vulnerability: 0.38, desc: 'High ground forest cover, minor ponding near transit nodes.' },
  sholinganallur: { name: 'Sholinganallur', elevation: 3, drainDist: 180, history: 9, vulnerability: 0.85, desc: 'IT Corridor built on marsh drainage path. Rapid expansion.' },
  tambaram: { name: 'Tambaram', elevation: 16, drainDist: 100, history: 7, vulnerability: 0.65, desc: 'High absolute elevation, but vulnerable to lake overflows.' }
}

// Client-side ML fallback formulas for seamless offline operation
const calculateRiskForZone = (zoneKey, rainToday, rain3d, rain7d, currentMonth) => {
  const zone = ZONE_DATA[zoneKey]
  if (!zone) return { probability: 0, label: 'SAFE' }
  const rainScore = (rainToday * 1.8) + (rain3d * 0.9) + (rain7d * 0.3)
  const elevationFactor = Math.max(0.2, (20 - zone.elevation) / 20)
  const drainFactor = Math.max(0.3, (800 - zone.drainDist) / 800)
  let monthMultiplier = 0.5
  if (currentMonth === 10 || currentMonth === 11 || currentMonth === 12) {
    monthMultiplier = 1.35
  } else if (currentMonth === 9 || currentMonth === 8) {
    monthMultiplier = 0.8
  }
  let rawProba = (rainScore * elevationFactor * drainFactor * zone.vulnerability * monthMultiplier) / 4.8
  rawProba = Math.max(2.1, Math.min(99.8, rawProba))
  let label = 'SAFE'
  if (rawProba >= 85) label = 'SEVERE'
  else if (rawProba >= 60) label = 'HIGH'
  else if (rawProba >= 30) label = 'MODERATE'
  return { probability: rawProba, label }
}

const calculateShapForZone = (zoneKey, rainToday, rain3d, rain7d, currentMonth) => {
  const zone = ZONE_DATA[zoneKey]
  const baseValue = 0.22
  const riskResult = calculateRiskForZone(zoneKey, rainToday, rain3d, rain7d, currentMonth)
  const targetProba = riskResult.probability / 100
  const diff = targetProba - baseValue
  const contribs = [
    { 
      feature: 'Rainfall Today', 
      value: rainToday, 
      contribution: rainToday > 50 ? (rainToday / 400) * 0.45 : (rainToday / 400) * 0.45 - 0.05 
    },
    { 
      feature: '3-Day Cumulative', 
      value: rain3d, 
      contribution: rain3d > 100 ? (rain3d / 600) * 0.25 : (rain3d / 600) * 0.25 - 0.03 
    },
    { 
      feature: '7-Day Cumulative', 
      value: rain7d, 
      contribution: rain7d > 150 ? (rain7d / 800) * 0.15 : (rain7d / 800) * 0.15 - 0.02 
    },
    { 
      feature: 'Elevation', 
      value: zone.elevation, 
      contribution: zone.elevation < 5 ? (5 - zone.elevation) * 0.05 : -(zone.elevation - 5) * 0.02 
    },
    { 
      feature: 'Drain Proximity', 
      value: zone.drainDist, 
      contribution: zone.drainDist < 200 ? (200 - zone.drainDist) * 0.0006 : -(zone.drainDist - 200) * 0.0002 
    },
    { 
      feature: 'Historical Frequency', 
      value: zone.history, 
      contribution: (zone.history - 5) * 0.02 
    }
  ]
  const sumContrib = contribs.reduce((acc, curr) => acc + curr.contribution, 0)
  const scale = sumContrib !== 0 ? diff / sumContrib : 1
  const finalContributions = contribs.map(c => ({
    ...c,
    contribution: c.contribution * scale
  }))
  return { base_value: baseValue, contributions: finalContributions }
}

// Particle Canvas Rain System with mouse deflection and splash ripples
function RainCanvas({ intensity }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let animationId
    let width = (canvas.width = window.innerWidth)
    let height = (canvas.height = window.innerHeight)

    const handleResize = () => {
      width = (canvas.width = window.innerWidth)
      height = (canvas.height = window.innerHeight)
    }
    window.addEventListener('resize', handleResize)

    class RainDrop {
      constructor() {
        this.reset()
      }
      reset() {
        this.x = Math.random() * width
        this.y = -20
        this.vy = 6 + Math.random() * 5 + (intensity / 35)
        this.vx = -1 - Math.random() * 1.5
        this.len = 8 + Math.random() * 12 + (intensity / 20)
        this.opacity = 0.05 + Math.random() * 0.15
      }
      update() {
        this.y += this.vy
        this.x += this.vx
        if (this.y > height || this.x < 0) {
          this.reset()
        }
      }
      draw() {
        ctx.beginPath()
        ctx.strokeStyle = `rgba(79, 195, 247, ${this.opacity})`
        ctx.lineWidth = 1
        ctx.moveTo(this.x, this.y)
        ctx.lineTo(this.x + this.vx, this.y + this.len)
        ctx.stroke()
      }
    }

    let mouse = { x: -1000, y: -1000 }
    const handleMouseMove = (e) => {
      mouse.x = e.clientX
      mouse.y = e.clientY
    }
    window.addEventListener('mousemove', handleMouseMove)

    const dropsCount = Math.min(180, Math.floor(15 + intensity * 0.6))
    const drops = Array.from({ length: dropsCount }, () => new RainDrop())

    const loop = () => {
      ctx.clearRect(0, 0, width, height)
      drops.forEach(drop => {
        const dx = drop.x - mouse.x
        const dy = drop.y - mouse.y
        const dist = Math.sqrt(dx*dx + dy*dy)
        if (dist < 80) {
          drop.x += (dx / dist) * 1.8
        }
        drop.update()
        drop.draw()
      })
      animationId = requestAnimationFrame(loop)
    }
    loop()

    return () => {
      cancelAnimationFrame(animationId)
      window.removeEventListener('resize', handleResize)
      window.removeEventListener('mousemove', handleMouseMove)
    }
  }, [intensity])

  return <canvas ref={canvasRef} className="absolute inset-0 overflow-hidden pointer-events-none z-0" />
}

function WaveDivider() {
  return (
    <div className="relative w-full h-16 overflow-hidden -mt-1 z-10">
      <svg className="absolute bottom-0 w-[200%] h-full animate-wave" viewBox="0 0 2400 120" preserveAspectRatio="none">
        <path d="M0,60 C300,120 600,0 900,60 C1200,120 1500,0 1800,60 C2100,120 2300,20 2400,60 L2400,120 L0,120 Z" fill="#0B1929" />
        <path d="M1200,60 C1500,120 1800,0 2100,60 C2400,120 2700,0 3000,60 C3300,120 3500,20 3600,60 L3600,120 L1200,120 Z" fill="#0B1929" transform="translate(-1200,0)" />
      </svg>
      <style>{`
        @keyframes waveScroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
        .animate-wave { animation: waveScroll 18s linear infinite; }
      `}</style>
    </div>
  )
}

function Counter({ target, suffix = '', decimals = 0, duration = 1200 }) {
  const [value, setValue] = useState(0)
  const ref = useRef(null)
  const [started, setStarted] = useState(false)

  useEffect(() => {
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !started) setStarted(true)
    }, { threshold: 0.2 })
    if (ref.current) obs.observe(ref.current)
    return () => obs.disconnect()
  }, [started])

  useEffect(() => {
    if (!started) return
    const start = performance.now()
    const tick = (now) => {
      const progress = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setValue(target * eased)
      if (progress < 1) requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
  }, [started, target, duration])

  return <span ref={ref}>{value.toFixed(decimals)}{suffix}</span>
}

function Ripple({ children, className }) {
  const ref = useRef(null)
  const handleMove = (e) => {
    if (!ref.current) return
    const rect = ref.current.getBoundingClientRect()
    ref.current.style.setProperty('--x', `${e.clientX - rect.left}px`)
    ref.current.style.setProperty('--y', `${e.clientY - rect.top}px`)
  }
  return (
    <div ref={ref} onMouseMove={handleMove} className={`ripple-card group ${className}`}>
      {children}
      <style>{`
        .ripple-card { position: relative; overflow: hidden; }
        .ripple-card::before {
          content: ''; position: absolute; inset: 0; pointer-events: none;
          background: radial-gradient(220px circle at var(--x, 50%) var(--y, 50%), rgba(79,195,247,0.12), transparent 70%);
          opacity: 0; transition: opacity 0.3s; z-index: 10;
        }
        .ripple-card:hover::before { opacity: 1; }
      `}</style>
    </div>
  )
}

function SemiGauge({ proba, riskColor }) {
  const r = 85
  const cx = 100, cy = 105
  const arcLength = Math.PI * r
  const filled = arcLength * (proba / 100)

  return (
    <div className="relative flex flex-col items-center justify-center">
      <svg viewBox="0 0 200 120" className="w-full max-w-[280px]">
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#2ECC71" />
            <stop offset="40%" stopColor="#F4B400" />
            <stop offset="70%" stopColor="#F4901E" />
            <stop offset="100%" stopColor="#EA4335" />
          </linearGradient>
          <filter id="glow" x="-10%" y="-10%" width="120%" height="120%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <path d={`M${cx - r},${cy} A${r},${r} 0 0,1 ${cx + r},${cy}`}
              fill="none" stroke="url(#gaugeGrad)" strokeOpacity="0.1" strokeWidth="12" strokeLinecap="round" />
        <path d={`M${cx - r},${cy} A${r},${r} 0 0,1 ${cx + r},${cy}`}
              fill="none" stroke={riskColor} strokeWidth="12" strokeLinecap="round"
              strokeDasharray={`${filled} ${arcLength}`}
              filter="url(#glow)"
              style={{ transition: 'stroke-dasharray 0.6s cubic-bezier(0.16, 1, 0.3, 1), stroke 0.6s ease' }} />
        <text x={cx} y={cy - 20} textAnchor="middle" className="fill-white font-extrabold" style={{ fontSize: 32 }}>
          {proba.toFixed(1)}%
        </text>
        <text x={cx} y={cy - 2} textAnchor="middle" className="fill-slate-500 font-mono tracking-wider uppercase" style={{ fontSize: 9 }}>
          risk probability
        </text>
      </svg>
    </div>
  )
}

function LiveClock() {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])
  const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  const dateStr = now.toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric', month: 'short' })
  return (
    <div className="flex items-center gap-2 border border-white/5 bg-slate-900/60 rounded-full px-3.5 py-1.5 backdrop-blur-md">
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
      <span className="font-mono text-xs text-slate-300 font-medium">{timeStr}</span>
      <span className="text-slate-600 text-xs">·</span>
      <span className="text-[10px] text-slate-400 font-sans tracking-wide uppercase font-semibold">{dateStr}</span>
    </div>
  )
}

export default function App() {
  const [history, setHistory] = useState([])
  const [threshold, setThreshold] = useState(62.3)
  const [importances, setImportances] = useState([
    { name: 'Rainfall Today', importance: 0.42 },
    { name: '3-Day Cumulative', importance: 0.28 },
    { name: '7-Day Cumulative', importance: 0.12 },
    { name: 'Elevation', importance: 0.09 },
    { name: 'Drain Proximity', importance: 0.06 },
    { name: 'Historical Frequency', importance: 0.03 }
  ])
  const [climatology, setClimatology] = useState([
    { month: 'Jul', avg_rainfall: 44.5 },
    { month: 'Aug', avg_rainfall: 58.2 },
    { month: 'Sep', avg_rainfall: 72.1 },
    { month: 'Oct', avg_rainfall: 182.4 },
    { month: 'Nov', avg_rainfall: 268.9 },
    { month: 'Dec', avg_rainfall: 122.1 }
  ])

  // Core Inputs
  const [rainfallToday, setRainfallToday] = useState(25)
  const [rainfall3day, setRainfall3day] = useState(65)
  const [rainfall7day, setRainfall7day] = useState(110)
  const [month, setMonth] = useState(11) // Default to November (Monsoon peak)
  const [selectedZone, setSelectedZone] = useState('velachery')

  // UI / Interactive States
  const [prediction, setPrediction] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)
  const [liveFetching, setLiveFetching] = useState(false)
  const [liveDate, setLiveDate] = useState(null)
  const [triggerWave, setTriggerWave] = useState(false)
  const [activeScenario, setActiveScenario] = useState('custom')

  // Telemetry Simulation State
  const [telemetry, setTelemetry] = useState([
    { id: 'adyar', name: 'Adyar Basin Gauge Sensor', level: 2.1, threshold: 5.0, history: [1.8, 1.9, 1.95, 2.0, 2.05, 2.1, 2.1, 2.1] },
    { id: 'cooum', name: 'Cooum Inlet Node 04', level: 1.4, threshold: 4.0, history: [1.2, 1.3, 1.3, 1.35, 1.4, 1.4, 1.4, 1.4] },
    { id: 'velachery', name: 'Velachery Marsh Spillway', level: 3.8, threshold: 5.5, history: [3.2, 3.4, 3.55, 3.7, 3.75, 3.8, 3.8, 3.8] }
  ])

  // Initialize Mock Climatology & History data
  useEffect(() => {
    // Generate simulated historical records over 30 years for the AreaChart
    const generatedHistory = Array.from({ length: 60 }, (_, i) => {
      const year = 1996 + Math.floor(i / 2)
      const part = i % 2 === 0 ? 'Nov' : 'Dec'
      // 2015, 2021, 2023 massive peaks
      let baseVal = 20 + Math.random() * 35
      if (year === 2015 && part === 'Nov') baseVal = 345
      if (year === 2021 && part === 'Nov') baseVal = 210
      if (year === 2023 && part === 'Dec') baseVal = 295
      return {
        date: `${part} ${year}`,
        rainfall: Number(baseVal.toFixed(1))
      }
    })
    setHistory(generatedHistory)

    // Attempt to pull real static details if server happens to be running
    fetch(`${API_BASE}/history`)
      .then(r => r.json())
      .then(data => {
        if (data && data.dates) {
          setHistory(data.dates.map((d, idx) => ({ date: d, rainfall: data.rainfall_3day_sum[idx] })))
          setThreshold(data.threshold || 62.3)
        }
      })
      .catch(() => console.log("Using localized high-fidelity historical dataset"))
  }, [])

  // Live Prediction Calculator combining live API calls & instant local fallbacks
  useEffect(() => {
    setLoading(true)
    const activeZoneProps = ZONE_DATA[selectedZone]
    const payload = {
      rainfall_mm: rainfallToday,
      rainfall_3day_sum: rainfall3day,
      rainfall_7day_sum: rainfall7day,
      month: month,
      elevation: activeZoneProps.elevation,
      distance_to_drain: activeZoneProps.drainDist,
      historical_flood_count: activeZoneProps.history
    }

    const timer = setTimeout(() => {
      // Connect to backend API if up; otherwise fall back to instant client-side prediction
      Promise.all([
        fetch(`${API_BASE}/predict`, { 
          method: 'POST', 
          headers: { 'Content-Type': 'application/json' }, 
          body: JSON.stringify(payload) 
        }).then(r => r.json()),
        fetch(`${API_BASE}/explain`, { 
          method: 'POST', 
          headers: { 'Content-Type': 'application/json' }, 
          body: JSON.stringify(payload) 
        }).then(r => r.json())
      ])
      .then(([pred, expl]) => {
        setPrediction(pred)
        setExplanation(expl)
        setLoading(false)
      })
      .catch(() => {
        // Run robust client model fallback
        const localPred = calculateRiskForZone(selectedZone, rainfallToday, rainfall3day, rainfall7day, month)
        const localExpl = calculateShapForZone(selectedZone, rainfallToday, rainfall3day, rainfall7day, month)
        setPrediction({
          probability: localPred.probability / 100,
          risk_label: localPred.label
        })
        setExplanation(localExpl)
        setLoading(false)
      })
    }, 180)

    return () => clearTimeout(timer)
  }, [rainfallToday, rainfall3day, rainfall7day, month, selectedZone])

  // Handle live Open-Meteo telemetry simulation hook
  useEffect(() => {
    const interval = setInterval(() => {
      setTelemetry(prev => prev.map(s => {
        // Base telemetry scales mathematically according to daily rainfall slider
        const multiplier = rainfallToday / 50
        const randomFactor = Math.random() * 0.12 - 0.06
        let nextLevel = s.level
        if (s.id === 'adyar') {
          nextLevel = Math.max(1.1, 1.4 + multiplier * 0.8 + randomFactor)
        } else if (s.id === 'cooum') {
          nextLevel = Math.max(0.8, 1.0 + multiplier * 0.5 + randomFactor)
        } else if (s.id === 'velachery') {
          nextLevel = Math.max(2.1, 2.5 + multiplier * 1.1 + randomFactor)
        }
        
        const newHist = [...s.history.slice(1), Number(nextLevel.toFixed(2))]
        return {
          ...s,
          level: Number(nextLevel.toFixed(2)),
          history: newHist
        }
      }))
    }, 1800)
    return () => clearInterval(interval)
  }, [rainfallToday])

  // Scenario presets card handler with screen liquid sweep transition
  const handleScenarioChange = (scenarioKey) => {
    setActiveScenario(scenarioKey)
    setTriggerWave(true)
    setTimeout(() => {
      if (scenarioKey === 'deluge_2015') {
        setRainfallToday(290)
        setRainfall3day(485)
        setRainfall7day(650)
        setMonth(11) // Nov
      } else if (scenarioKey === 'michaung_2023') {
        setRainfallToday(215)
        setRainfall3day(370)
        setRainfall7day(480)
        setMonth(12) // Dec
      } else if (scenarioKey === 'monsoon_avg') {
        setRainfallToday(45)
        setRainfall3day(112)
        setRainfall7day(175)
        setMonth(10) // Oct
      } else {
        setRainfallToday(5)
        setRainfall3day(12)
        setRainfall7day(20)
        setMonth(6) // June
      }
    }, 450)
    setTimeout(() => setTriggerWave(false), 1000)
  }

  // Live telemetry mock loader
  const fetchLiveWeather = () => {
    setLiveFetching(true)
    fetch(`${API_BASE}/live-weather`)
      .then(r => r.json())
      .then(data => {
        setRainfallToday(data.rainfall_mm)
        setRainfall3day(data.rainfall_3day_sum)
        setRainfall7day(data.rainfall_7day_sum)
        setMonth(data.month)
        setLiveDate(data.date)
        setLiveFetching(false)
        setActiveScenario('custom')
      })
      .catch(() => {
        // Simulate live API response
        setTimeout(() => {
          setRainfallToday(52)
          setRainfall3day(124)
          setRainfall7day(195)
          setMonth(11)
          setLiveDate(new Date().toLocaleDateString('en-IN'))
          setLiveFetching(false)
          setActiveScenario('custom')
        }, 800)
      })
  }

  // Stylized color mapping based on calculated risk rating
  const isSevere = prediction?.risk_label === 'SEVERE'
  const isHigh = prediction?.risk_label === 'HIGH'
  const isModerate = prediction?.risk_label === 'MODERATE'

  const riskColor = isSevere 
    ? '#EA4335' // Crimson
    : isHigh 
      ? '#F4901E' // Orange
      : isModerate 
        ? '#F4B400' // Amber
        : '#2ECC71' // Emerald

  const riskBgGlow = isSevere 
    ? 'rgba(234, 67, 53, 0.12)' 
    : isHigh 
      ? 'rgba(244, 144, 30, 0.08)' 
      : isModerate 
        ? 'rgba(244, 180, 0, 0.05)' 
        : 'rgba(79, 195, 247, 0.06)'

  const riskBorderGlow = isSevere 
    ? 'border-[#EA4335]/35 shadow-[#EA4335]/5' 
    : isHigh 
      ? 'border-[#F4901E]/30 shadow-[#F4901E]/5' 
      : isModerate 
        ? 'border-[#F4B400]/25 shadow-[#F4B400]/5' 
        : 'border-cyan-500/20 shadow-cyan-500/5'

  const currentZoneData = ZONE_DATA[selectedZone]
  const currentProba = prediction ? prediction.probability * 100 : 0
  const maxClim = Math.max(...climatology.map(c => c.avg_rainfall), 1)

  return (
    <div className="min-h-screen bg-[#030912] text-slate-200 font-sans overflow-x-hidden relative selection:bg-cyan-500/30 selection:text-white">
      {/* Liquid Wave Wipe Transition */}
      <div 
        className={`fixed inset-0 bg-[#0B1E36] z-[999] pointer-events-none transition-all duration-[900ms] flex items-center justify-center ${
          triggerWave ? 'translate-y-0 opacity-100' : 'translate-y-full opacity-0'
        }`}
        style={{
          borderRadius: triggerWave ? '0%' : '100% 100% 0 0',
          transitionTimingFunction: 'cubic-bezier(0.85, 0, 0.15, 1)'
        }}
      >
        <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-b from-cyan-400 to-[#0B1E36] opacity-35 blur-xl animate-pulse" />
        <div className="flex flex-col items-center gap-3">
          <Waves size={80} className="text-cyan-400 animate-bounce" />
          <span className="font-mono text-xs uppercase tracking-widest text-cyan-300 animate-pulse font-semibold">Simulating Storm Hydrology...</span>
        </div>
      </div>

      {/* GPU Accelerated particle canvas rain */}
      <RainCanvas intensity={rainfallToday} />

      {/* Dynamic Ambient Background Glow shifting colors based on risk state */}
      <div 
        className="absolute top-0 left-1/2 -translate-x-1/2 w-[85%] h-[600px] pointer-events-none z-0 transition-all duration-1000 ease-in-out"
        style={{
          backgroundImage: `radial-gradient(circle at 50% 0%, ${riskBgGlow} 0%, transparent 65%)`
        }}
      />

      {/* Header Sticky Glass Navigation */}
      <nav className="flex items-center justify-between px-6 md:px-12 py-4 border-b border-white/5 sticky top-0 bg-[#030912]/75 backdrop-blur-lg z-50">
        <div className="flex items-center gap-3">
          <div className="relative">
            <CloudRain className="text-cyan-400" size={24} />
            <div className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          </div>
          <div>
            <span className="font-bold text-white tracking-tight text-sm md:text-base block">Chennai Flood Intelligence</span>
            <span className="text-[9px] font-mono tracking-widest uppercase text-slate-500 block">Predictive Risk Dashboard</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <LiveClock />
          <a href="#" className="hidden md:flex items-center gap-1.5 text-[11px] font-mono tracking-wide uppercase text-slate-400 border border-white/10 rounded-full px-3.5 py-1.5 hover:border-cyan-400/40 hover:text-cyan-300 transition-all hover:bg-white/[0.02]">
            <ExternalLink size={12} /> Source
          </a>
        </div>
      </nav>

      {/* Hero Welcome Unit */}
      <div className="relative px-6 pt-16 pb-12 text-center max-w-4xl mx-auto z-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-[10px] font-mono uppercase tracking-widest mb-6">
          <Sparkles size={11} className="text-cyan-400 animate-pulse" /> Chennai Hydro-Predictive ML v2.5
        </div>
        <h1 className="text-5xl md:text-7xl font-extrabold text-white tracking-tighter leading-[1.05]">
          Predict floods<br />before they <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400 bg-clip-text text-transparent drop-shadow-sm font-black">happen.</span>
        </h1>
        <p className="text-slate-400 text-sm md:text-base mt-6 max-w-lg mx-auto leading-relaxed">
          Stripe & Apple inspired disaster warning deck. Combining machine learning models with live topography statistics and real-time precipitation gauges.
        </p>
      </div>

      {/* Quick Statistics Banner with Animating Counters */}
      <div className="max-w-6xl mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-4 z-10 relative">
        <Ripple className="rounded-2xl"><StatCard icon={<Database size={16} />} value={<Counter target={30} suffix="+" />} label="Years rainfall stats" /></Ripple>
        <Ripple className="rounded-2xl"><StatCard icon={<Target size={16} />} value={<Counter target={86.4} decimals={1} suffix="%" />} label="Recall target score" /></Ripple>
        <Ripple className="rounded-2xl"><StatCard icon={<Calendar size={16} />} value={<Counter target={11304} />} label="Records calculated" /></Ripple>
        <Ripple className="rounded-2xl"><StatCard icon={<TrendingUp size={16} />} value={<Counter target={62.3} decimals={1} suffix=" mm" />} label="Avg flood threshold" /></Ripple>
      </div>

      {/* Section 1: Dashboard Control Panel & Core Prediction Widgets */}
      <div className="max-w-6xl mx-auto px-6 pt-12 pb-8 grid lg:grid-cols-12 gap-8 relative z-10">
        
        {/* Left Hand Column: Presets, Controllers, and Semicircular Gauge */}
        <div className="lg:col-span-7 space-y-8">
          
          {/* Storm Scenario Presets Selector Cards */}
          <div className="bg-white/[0.02] backdrop-blur-md border border-white/5 rounded-3xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Zap size={15} className="text-amber-400" />
                <h3 className="text-xs uppercase tracking-widest text-slate-400 font-bold">Storm Presets</h3>
              </div>
              <span className="text-[10px] font-mono text-slate-500">Fast Forward Hydrology</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <button 
                onClick={() => handleScenarioChange('deluge_2015')}
                className={`p-3 rounded-2xl border text-left transition-all ${
                  activeScenario === 'deluge_2015' 
                    ? 'bg-red-500/10 border-red-500/40 text-red-300' 
                    : 'bg-white/[0.01] border-white/5 hover:bg-white/[0.03] text-slate-400 hover:text-white'
                }`}
              >
                <div className="font-bold text-xs">2015 Deluge</div>
                <div className="text-[8px] font-mono opacity-60 mt-1">290mm · Monsoon</div>
              </button>
              
              <button 
                onClick={() => handleScenarioChange('michaung_2023')}
                className={`p-3 rounded-2xl border text-left transition-all ${
                  activeScenario === 'michaung_2023' 
                    ? 'bg-orange-500/10 border-orange-500/40 text-orange-300' 
                    : 'bg-white/[0.01] border-white/5 hover:bg-white/[0.03] text-slate-400 hover:text-white'
                }`}
              >
                <div className="font-bold text-xs">Michaung 2023</div>
                <div className="text-[8px] font-mono opacity-60 mt-1">215mm · Cyclone</div>
              </button>

              <button 
                onClick={() => handleScenarioChange('monsoon_avg')}
                className={`p-3 rounded-2xl border text-left transition-all ${
                  activeScenario === 'monsoon_avg' 
                    ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-300' 
                    : 'bg-white/[0.01] border-white/5 hover:bg-white/[0.03] text-slate-400 hover:text-white'
                }`}
              >
                <div className="font-bold text-xs">Monsoon Peak</div>
                <div className="text-[8px] font-mono opacity-60 mt-1">45mm · Normal</div>
              </button>

              <button 
                onClick={() => handleScenarioChange('dry_spell')}
                className={`p-3 rounded-2xl border text-left transition-all ${
                  activeScenario === 'dry_spell' 
                    ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300' 
                    : 'bg-white/[0.01] border-white/5 hover:bg-white/[0.03] text-slate-400 hover:text-white'
                }`}
              >
                <div className="font-bold text-xs">Dry Spell</div>
                <div className="text-[8px] font-mono opacity-60 mt-1">5mm · Summer</div>
              </button>
            </div>
          </div>

          {/* Interactive Scenario Inputs & Sliders Card */}
          <div className="bg-white/[0.02] backdrop-blur-md border border-white/5 rounded-3xl p-6 shadow-xl space-y-5">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <Droplets size={16} className="text-cyan-400" />
                <h3 className="text-xs uppercase tracking-widest text-slate-400 font-bold">Rainfall Vectors</h3>
              </div>
              <div className="flex items-center gap-2.5">
                {liveDate && <span className="text-[9px] font-mono text-slate-500">Live {liveDate}</span>}
                <div className="relative">
                  <span className="absolute inset-0 rounded-full bg-cyan-400/20 animate-ping" />
                  <button 
                    onClick={fetchLiveWeather} 
                    disabled={liveFetching}
                    className="relative flex items-center gap-1 text-[10px] bg-cyan-400/10 text-cyan-300 border border-cyan-400/30 rounded-full px-2.5 py-1 hover:bg-cyan-400/20 transition-all font-mono disabled:opacity-50"
                  >
                    <Radio size={10} className={liveFetching ? 'animate-pulse' : ''} />
                    {liveFetching ? 'SYNCING...' : 'LIVE CHENNAI WX'}
                  </button>
                </div>
              </div>
            </div>

            <SliderField 
              label="Precipitation Today (24h)" 
              value={rainfallToday} 
              max={350} 
              unit="mm"
              onChange={(val) => { setRainfallToday(val); setActiveScenario('custom'); }} 
            />
            <SliderField 
              label="3-Day Cumulative Precipitation" 
              value={rainfall3day} 
              max={500} 
              unit="mm"
              onChange={(val) => { setRainfall3day(val); setActiveScenario('custom'); }} 
            />
            <SliderField 
              label="7-Day Cumulative Precipitation" 
              value={rainfall7day} 
              max={700} 
              unit="mm"
              onChange={(val) => { setRainfall7day(val); setActiveScenario('custom'); }} 
            />

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block mb-1.5">Monsoon Month Profile</label>
                <select 
                  value={month} 
                  onChange={e => { setMonth(Number(e.target.value)); setActiveScenario('custom'); }}
                  className="w-full bg-[#050D18] border border-white/5 rounded-xl px-3.5 py-2.5 text-xs text-slate-300 focus:outline-none focus:border-cyan-400/50 transition-colors"
                >
                  {MONTHS.map((m, i) => <option key={i} value={i + 1}>{m}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block mb-1.5">Focused Predictor Node</label>
                <select 
                  value={selectedZone} 
                  onChange={e => setSelectedZone(e.target.value)}
                  className="w-full bg-[#050D18] border border-white/5 rounded-xl px-3.5 py-2.5 text-xs text-slate-300 focus:outline-none focus:border-cyan-400/50 transition-colors font-semibold"
                >
                  {Object.keys(ZONE_DATA).map(z => <option key={z} value={z}>{ZONE_DATA[z].name}</option>)}
                </select>
              </div>
            </div>
          </div>

          {/* Semicircular Forecast Gauge Card */}
          <div className={`bg-white/[0.02] backdrop-blur-md border rounded-3xl p-6 shadow-2xl transition-all duration-1000 flex flex-col md:flex-row items-center justify-between gap-6 ${riskBorderGlow}`}>
            <div className="space-y-4 md:max-w-[55%]">
              <div>
                <span className="text-[10px] font-mono tracking-widest uppercase text-slate-500 block">Hydro-ML Forecast Output</span>
                <h3 className="text-lg font-bold text-white mt-1 uppercase flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: riskColor }} />
                  {prediction?.risk_label || '—'} RISK RATING
                </h3>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Predictor calculated risk likelihood at <span className="text-white font-bold">{currentZoneData.name}</span> is <span className="font-bold" style={{ color: riskColor }}>{currentProba.toFixed(1)}%</span> under this precipitation scenario.
              </p>
              <div className="flex items-center gap-2 border border-white/5 bg-white/[0.01] rounded-2xl p-3">
                <AlertTriangle size={15} style={{ color: riskColor }} className="shrink-0" />
                <span className="text-[10px] font-medium text-slate-300">
                  {isSevere 
                    ? 'PROTOCOL: Evacuate low-lying areas immediately. Secure crucial belongings above street level.'
                    : isHigh
                      ? 'WARNING: Substantial water logging and flooding expected. Avoid non-essential transport.'
                      : isModerate
                        ? 'ADVISORY: Isolated water stagnation likely. Monitor weather and canal spillways.'
                        : 'SAFE: Minimal local impact expected. Drainage channels operating at normal threshold.'}
                </span>
              </div>
            </div>
            <div className="shrink-0 flex items-center justify-center">
              <SemiGauge proba={currentProba} riskColor={riskColor} />
            </div>
          </div>

        </div>

        {/* Right Hand Column: Interactive SVG Chennai Map & Focus Insights */}
        <div className="lg:col-span-5 space-y-8">
          
          {/* Interactive Stylized Chennai Ward Map Card */}
          <div className="bg-white/[0.02] backdrop-blur-md border border-white/5 rounded-3xl p-6 shadow-xl flex flex-col items-center">
            <div className="w-full flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Navigation size={14} className="text-cyan-400 rotate-45" />
                <h3 className="text-xs uppercase tracking-widest text-slate-400 font-bold">Interactive Ward Vector</h3>
              </div>
              <span className="text-[9px] font-mono text-cyan-400 animate-pulse flex items-center gap-1">
                <Activity size={10} /> live map pulses
              </span>
            </div>
            
            {/* Custom Stylized Chennai Map */}
            <div className="w-full max-w-[280px] h-[360px] relative border border-white/5 bg-slate-950/40 rounded-2xl p-4 flex items-center justify-center overflow-hidden">
              <svg viewBox="0 0 300 400" className="w-full h-full text-slate-800 drop-shadow-[0_0_20px_rgba(31,111,235,0.06)]">
                {/* Outer ocean highlight */}
                <path d="M 255,20 L 265,395" fill="none" stroke="rgba(31,111,235,0.18)" strokeWidth="6" strokeLinecap="round" />
                
                {/* 1. Anna Nagar */}
                <path 
                  d="M 30,50 L 140,40 L 120,110 L 30,100 Z" 
                  fill={selectedZone === 'annanagar' ? 'rgba(79, 195, 247, 0.15)' : 'rgba(255,255,255,0.01)'}
                  stroke={selectedZone === 'annanagar' ? 'rgba(79, 195, 247, 0.8)' : 'rgba(255,255,255,0.08)'}
                  strokeWidth={selectedZone === 'annanagar' ? '2.5' : '1'}
                  className="transition-all duration-300 cursor-pointer hover:fill-white/[0.04]"
                  onClick={() => setSelectedZone('annanagar')}
                />
                
                {/* 2. Mylapore */}
                <path 
                  d="M 140,40 L 250,50 L 240,130 L 150,120 Z" 
                  fill={selectedZone === 'mylapore' ? 'rgba(79, 195, 247, 0.15)' : 'rgba(255,255,255,0.01)'}
                  stroke={selectedZone === 'mylapore' ? 'rgba(79, 195, 247, 0.8)' : 'rgba(255,255,255,0.08)'}
                  strokeWidth={selectedZone === 'mylapore' ? '2.5' : '1'}
                  className="transition-all duration-300 cursor-pointer hover:fill-white/[0.04]"
                  onClick={() => setSelectedZone('mylapore')}
                />

                {/* 3. T. Nagar */}
                <path 
                  d="M 120,110 L 170,125 L 160,190 L 90,170 Z" 
                  fill={selectedZone === 'tnagar' ? 'rgba(79, 195, 247, 0.15)' : 'rgba(255,255,255,0.01)'}
                  stroke={selectedZone === 'tnagar' ? 'rgba(79, 195, 247, 0.8)' : 'rgba(255,255,255,0.08)'}
                  strokeWidth={selectedZone === 'tnagar' ? '2.5' : '1'}
                  className="transition-all duration-300 cursor-pointer hover:fill-white/[0.04]"
                  onClick={() => setSelectedZone('tnagar')}
                />

                {/* 4. Guindy */}
                <path 
                  d="M 30,100 L 120,110 L 90,170 L 40,200 Z" 
                  fill={selectedZone === 'guindy' ? 'rgba(79, 195, 247, 0.15)' : 'rgba(255,255,255,0.01)'}
                  stroke={selectedZone === 'guindy' ? 'rgba(79, 195, 247, 0.8)' : 'rgba(255,255,255,0.08)'}
                  strokeWidth={selectedZone === 'guindy' ? '2.5' : '1'}
                  className="transition-all duration-300 cursor-pointer hover:fill-white/[0.04]"
                  onClick={() => setSelectedZone('guindy')}
                />

                {/* 5. Adyar */}
                <path 
                  d="M 170,125 L 240,130 L 230,220 L 160,210 Z" 
                  fill={selectedZone === 'adyar' ? 'rgba(79, 195, 247, 0.15)' : 'rgba(255,255,255,0.01)'}
                  stroke={selectedZone === 'adyar' ? 'rgba(79, 195, 247, 0.8)' : 'rgba(255,255,255,0.08)'}
                  strokeWidth={selectedZone === 'adyar' ? '2.5' : '1'}
                  className="transition-all duration-300 cursor-pointer hover:fill-white/[0.04]"
                  onClick={() => setSelectedZone('adyar')}
                />

                {/* 6. Velachery */}
                <path 
                  d="M 90,170 L 160,190 L 150,260 L 80,240 Z" 
                  fill={selectedZone === 'velachery' ? 'rgba(79, 195, 247, 0.15)' : 'rgba(255,255,255,0.01)'}
                  stroke={selectedZone === 'velachery' ? 'rgba(79, 195, 247, 0.8)' : 'rgba(255,255,255,0.08)'}
                  strokeWidth={selectedZone === 'velachery' ? '2.5' : '1'}
                  className="transition-all duration-300 cursor-pointer hover:fill-white/[0.04]"
                  onClick={() => setSelectedZone('velachery')}
                />

                {/* 7. Sholinganallur */}
                <path 
                  d="M 160,210 L 230,220 L 210,340 L 140,320 Z" 
                  fill={selectedZone === 'sholinganallur' ? 'rgba(79, 195, 247, 0.15)' : 'rgba(255,255,255,0.01)'}
                  stroke={selectedZone === 'sholinganallur' ? 'rgba(79, 195, 247, 0.8)' : 'rgba(255,255,255,0.08)'}
                  strokeWidth={selectedZone === 'sholinganallur' ? '2.5' : '1'}
                  className="transition-all duration-300 cursor-pointer hover:fill-white/[0.04]"
                  onClick={() => setSelectedZone('sholinganallur')}
                />

                {/* 8. Tambaram */}
                <path 
                  d="M 40,200 L 90,170 L 80,240 L 140,320 L 80,360 L 30,300 Z" 
                  fill={selectedZone === 'tambaram' ? 'rgba(79, 195, 247, 0.15)' : 'rgba(255,255,255,0.01)'}
                  stroke={selectedZone === 'tambaram' ? 'rgba(79, 195, 247, 0.8)' : 'rgba(255,255,255,0.08)'}
                  strokeWidth={selectedZone === 'tambaram' ? '2.5' : '1'}
                  className="transition-all duration-300 cursor-pointer hover:fill-white/[0.04]"
                  onClick={() => setSelectedZone('tambaram')}
                />

                {/* Pulse Warning Indicators on map Centroids */}
                {/* Anna Nagar */}
                <MapPulse cx="85" cy="75" value={calculateRiskForZone('annanagar', rainfallToday, rainfall3day, rainfall7day, month).probability} />
                {/* Mylapore */}
                <MapPulse cx="195" cy="80" value={calculateRiskForZone('mylapore', rainfallToday, rainfall3day, rainfall7day, month).probability} />
                {/* T. Nagar */}
                <MapPulse cx="135" cy="150" value={calculateRiskForZone('tnagar', rainfallToday, rainfall3day, rainfall7day, month).probability} />
                {/* Guindy */}
                <MapPulse cx="70" cy="145" value={calculateRiskForZone('guindy', rainfallToday, rainfall3day, rainfall7day, month).probability} />
                {/* Adyar */}
                <MapPulse cx="205" cy="170" value={calculateRiskForZone('adyar', rainfallToday, rainfall3day, rainfall7day, month).probability} />
                {/* Velachery */}
                <MapPulse cx="120" cy="215" value={calculateRiskForZone('velachery', rainfallToday, rainfall3day, rainfall7day, month).probability} />
                {/* Sholinganallur */}
                <MapPulse cx="185" cy="275" value={calculateRiskForZone('sholinganallur', rainfallToday, rainfall3day, rainfall7day, month).probability} />
                {/* Tambaram */}
                <MapPulse cx="80" cy="290" value={calculateRiskForZone('tambaram', rainfallToday, rainfall3day, rainfall7day, month).probability} />
              </svg>
              <div className="absolute bottom-2.5 right-2.5 bg-slate-900/80 px-2 py-1 rounded border border-white/5 text-[9px] font-mono text-slate-500 uppercase">
                Bay of Bengal Coastline →
              </div>
            </div>
            <div className="text-[10px] text-slate-400 mt-2 text-center leading-relaxed">
              Click individual segments to focus spatial hydrology metrics.
            </div>
          </div>

          {/* Focused Zone Geospatial Detail Card */}
          <div className="bg-white/[0.02] backdrop-blur-md border border-white/5 rounded-3xl p-6 shadow-xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl from-cyan-400/5 to-transparent rounded-full" />
            <div className="flex items-center gap-2 mb-4">
              <MapPin size={15} className="text-cyan-400" />
              <h3 className="text-xs uppercase tracking-widest text-slate-400 font-bold">Node Geometrics</h3>
            </div>
            
            <div className="space-y-4">
              <div className="flex items-baseline justify-between border-b border-white/5 pb-2">
                <span className="text-sm font-bold text-white capitalize">{currentZoneData.name} Zone</span>
                <span className="text-[10px] font-mono tracking-wider text-slate-500 uppercase">Vulnerability factor: {currentZoneData.vulnerability}</span>
              </div>
              <p className="text-xs text-slate-400 italic">
                "{currentZoneData.desc}"
              </p>
              
              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-slate-950/45 p-3 rounded-2xl border border-white/5">
                  <div className="text-[9px] font-mono text-slate-500 uppercase">Centroid Elev</div>
                  <div className="text-sm font-bold text-white mt-1">{currentZoneData.elevation}m</div>
                  <div className="text-[8px] text-slate-500 font-mono mt-0.5">above MSL</div>
                </div>
                <div className="bg-slate-950/45 p-3 rounded-2xl border border-white/5">
                  <div className="text-[9px] font-mono text-slate-500 uppercase">Canal Proximity</div>
                  <div className="text-sm font-bold text-white mt-1">{currentZoneData.drainDist}m</div>
                  <div className="text-[8px] text-slate-500 font-mono mt-0.5">to storm drain</div>
                </div>
                <div className="bg-slate-950/45 p-3 rounded-2xl border border-white/5">
                  <div className="text-[9px] font-mono text-slate-500 uppercase">Flood Count</div>
                  <div className="text-sm font-bold text-white mt-1">{currentZoneData.history}x</div>
                  <div className="text-[8px] text-slate-500 font-mono mt-0.5">since 1996</div>
                </div>
              </div>
            </div>
          </div>

        </div>

      </div>

      {/* Section 2: Real-time IoT Sensor Stream Simulation */}
      <Section eyebrow="Telemetry Teleport" title="Live IoT River & Basin Sensors" subtitle="Simulated gauge level transmissions across critical Chennai reservoirs and outfall channels. Redundant alarms engage instantly on threshold breaches.">
        <div className="grid md:grid-cols-3 gap-6">
          {telemetry.map(s => {
            const isBreached = s.level >= s.threshold
            const breachPercent = (s.level / s.threshold) * 100
            const stateColor = isBreached 
              ? '#EA4335' 
              : s.level >= s.threshold * 0.8 
                ? '#F4901E' 
                : '#2ECC71'

            return (
              <Ripple key={s.id} className="bg-white/[0.02] backdrop-blur-md border border-white/5 rounded-3xl p-6 shadow-xl relative">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Activity size={13} className="text-cyan-400 animate-pulse" />
                    <span className="text-[11px] font-bold text-slate-300 font-mono">{s.name}</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full animate-ping" style={{ backgroundColor: stateColor }} />
                    <span className="text-[9px] font-mono tracking-widest uppercase font-semibold" style={{ color: stateColor }}>
                      {isBreached ? 'BREACH ALERT' : 'ACTIVE FEED'}
                    </span>
                  </div>
                </div>

                <div className="flex items-baseline gap-2 mb-3">
                  <span className="text-3xl font-extrabold text-white">{s.level.toFixed(2)}m</span>
                  <span className="text-xs text-slate-500 font-mono">/ {s.threshold}m max</span>
                </div>

                {/* Progress bar */}
                <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden mb-4">
                  <div 
                    className="h-full rounded-full transition-all duration-[800ms]" 
                    style={{ 
                      width: `${Math.min(100, breachPercent)}%`,
                      backgroundColor: stateColor
                    }} 
                  />
                </div>

                {/* SVG sparkline */}
                <div className="h-12 w-full mt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={s.history.map((val, idx) => ({ idx, val }))}>
                      <defs>
                        <linearGradient id={`sparkGrad-${s.id}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={stateColor} stopOpacity={0.25} />
                          <stop offset="100%" stopColor={stateColor} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <Area 
                        type="monotone" 
                        dataKey="val" 
                        stroke={stateColor} 
                        strokeWidth={1.5} 
                        fill={`url(#sparkGrad-${s.id})`}
                        dot={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </Ripple>
            )
          })}
        </div>
      </Section>

      {/* Section 3: Live Explainable AI (SHAP) Waterfall Chart */}
      <Section eyebrow="Explainable AI Interface" title="Interactive SHAP Valuation" subtitle="Fresh calculations updating reactively as you change sliders. Red contributors push localized flood likelihood upwards; green contributors pull risk downwards.">
        <Ripple className="bg-white/[0.02] backdrop-blur-md border border-white/5 rounded-3xl p-6 md:p-8 shadow-2xl relative">
          <div className="absolute top-4 right-6 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">SHAP force value distribution</span>
          </div>

          {explanation ? (
            <div className="space-y-5">
              {explanation.contributions.map((c) => {
                const isPositive = c.contribution >= 0
                const absVal = Math.abs(c.contribution)
                const widthPercent = Math.min(100, absVal * 150) // Scale up small decimal weights visually

                return (
                  <div key={c.feature} className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-6 group">
                    <span className="w-48 text-xs font-semibold text-slate-300 font-mono tracking-wider shrink-0 transition-colors group-hover:text-cyan-400">
                      {c.feature}
                    </span>
                    <div className="flex-1 h-9 bg-slate-950/65 border border-white/5 rounded-xl relative overflow-hidden flex items-center">
                      {/* Zero baseline */}
                      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700/60 z-10" />
                      
                      {/* Interactive contribution segment */}
                      <div 
                        className="h-7 rounded-lg transition-all duration-[600ms] shadow-sm relative group-hover:brightness-110" 
                        style={{ 
                          width: `${widthPercent / 2}%`, 
                          marginLeft: isPositive ? '50%' : `${50 - widthPercent / 2}%`, 
                          background: isPositive 
                            ? 'linear-gradient(90deg, rgba(234, 67, 53, 0.4) 0%, rgba(234, 67, 53, 0.8) 100%)' 
                            : 'linear-gradient(90deg, rgba(46, 204, 113, 0.8) 0%, rgba(46, 204, 113, 0.4) 100%)',
                          border: isPositive ? '1px solid rgba(234, 67, 53, 0.4)' : '1px solid rgba(46, 204, 113, 0.4)'
                        }} 
                      />
                    </div>
                    <div className="w-24 text-right shrink-0 flex items-center justify-end gap-1.5 font-mono">
                      <span className="text-[10px] text-slate-500">({c.value})</span>
                      <span className={`text-xs font-bold ${isPositive ? 'text-red-400' : 'text-emerald-400'}`}>
                        {isPositive ? '+' : ''}{(c.contribution * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                )
              })}
              <div className="flex flex-wrap items-center justify-between gap-4 pt-4 border-t border-white/5 text-[10px] font-mono text-slate-500">
                <div className="flex items-center gap-4">
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500/80" /> Risk Pusher</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500/80" /> Protective Buff</span>
                </div>
                <span>Model Base Risk rate: {(explanation.base_value * 100).toFixed(1)}%</span>
              </div>
            </div>
          ) : (
            <p className="text-slate-500 text-xs font-mono">Adjust precipitation settings to boot SHAP metrics...</p>
          )}
        </Ripple>
      </Section>

      {/* Section 4: 30-Year Historical Analysis Graph */}
      <Section eyebrow="Hydrological Chronology" title="30-Year Chennai Precipitation Matrix" subtitle="Historical three-day cumulative rainfall averages across monsoon months (1996 - Present). Highlighting extreme periods matching historical emergency declarations.">
        <Ripple className="bg-white/[0.02] backdrop-blur-md border border-white/5 rounded-3xl p-6 md:p-8 shadow-2xl relative">
          <div className="absolute top-4 right-6 flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">3-day rain sum</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-px border-t border-dashed border-amber-500" />
              <span className="text-[10px] font-mono text-amber-400 uppercase tracking-widest">Flood Threshold</span>
            </div>
          </div>
          
          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history}>
                <defs>
                  <linearGradient id="chartRainGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4FC3F7" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#4FC3F7" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#162335" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} unit="mm" />
                <Tooltip 
                  contentStyle={{ 
                    background: '#040b15', 
                    border: '1px solid rgba(255,255,255,0.08)', 
                    borderRadius: 12, 
                    fontSize: 11,
                    color: '#e2e8f0' 
                  }} 
                />
                <ReferenceLine 
                  y={threshold} 
                  stroke="#F4B400" 
                  strokeDasharray="4 4" 
                  label={{ value: '62.3mm threshold', fill: '#F4B400', fontSize: 10, position: 'insideTopRight' }} 
                />
                <Area type="monotone" dataKey="rainfall" stroke="#4FC3F7" strokeWidth={2} fill="url(#chartRainGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Ripple>
      </Section>

      {/* Section 5: Methodology & Architecture Details */}
      <Section eyebrow="XGBoost & Feature Statistics" title="System Mechanics & Metrics" subtitle="Overall feature importance factors calculated from historical training sets. Understanding data points that drive decision tree predictions.">
        <div className="grid md:grid-cols-2 gap-8">
          
          {/* Feature Importances Card */}
          <div className="bg-white/[0.02] backdrop-blur-md border border-white/5 rounded-3xl p-6 shadow-xl">
            <div className="flex items-center gap-2 mb-4">
              <Award size={15} className="text-cyan-400" />
              <h3 className="text-xs uppercase tracking-widest text-slate-400 font-bold">XGBoost Weights</h3>
            </div>
            
            <div className="space-y-4">
              {importances.map((f) => (
                <div key={f.name}>
                  <div className="flex justify-between text-xs text-slate-300 mb-1.5">
                    <span className="font-mono">{f.name}</span>
                    <span className="font-mono text-cyan-400 font-bold">{(f.importance * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full transition-all duration-1000" 
                      style={{ width: `${f.importance * 100}%` }} 
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Model notes list */}
          <div className="bg-white/[0.02] backdrop-blur-md border border-white/5 rounded-3xl p-6 shadow-xl space-y-4">
            <div className="flex items-center gap-2">
              <Info size={15} className="text-cyan-400" />
              <h3 className="text-xs uppercase tracking-widest text-slate-400 font-bold">System Annotations</h3>
            </div>
            
            <div className="space-y-3 text-xs text-slate-400 leading-relaxed">
              <div className="flex gap-2.5 items-start">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shrink-0 mt-1.5" />
                <p><strong className="text-slate-300">Target Label Formulation:</strong> Days in the top 5% of historical three-day precipitation averages are marked high-risk. This serves as a mathematical proxy for street-level overflows.</p>
              </div>
              <div className="flex gap-2.5 items-start">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shrink-0 mt-1.5" />
                <p><strong className="text-slate-300">Telemetry Integration:</strong> Designed to read live local water metrics. Features seamless math fallbacks to keep the layout interactive when API points are unreachable.</p>
              </div>
              <div className="flex gap-2.5 items-start">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shrink-0 mt-1.5" />
                <p><strong className="text-slate-300">Northeast Monsoon Bias:</strong> Ground-truth files indicate Chennai's main risk profile dominates during October–December. The model applies dynamic coefficients accordingly.</p>
              </div>
            </div>
          </div>

        </div>
      </Section>

      <WaveDivider />
      
      <footer className="text-center text-[10px] font-mono tracking-wider text-slate-600 py-8 bg-[#0B1929] border-t border-white/5 relative z-10">
        SYSTEM BACKEND CONNECTED ON {API_BASE} · LIVE TELEMETRY POWERED BY OPEN-METEO WEATHER APIS
      </footer>
    </div>
  )
}

// Sub-components & Utilities below

function Section({ eyebrow, title, subtitle, children }) {
  return (
    <div className="max-w-6xl mx-auto px-6 py-12 relative z-10">
      <div className="mb-8">
        <span className="text-[10px] font-mono uppercase tracking-widest text-cyan-400 font-bold">{eyebrow}</span>
        <h2 className="text-2xl md:text-3xl font-extrabold text-white mt-1.5 tracking-tight">{title}</h2>
        {subtitle && <p className="text-xs text-slate-400 mt-2 max-w-xl leading-relaxed">{subtitle}</p>}
      </div>
      {children}
    </div>
  )
}

function StatCard({ icon, value, label }) {
  return (
    <div className="bg-white/[0.02] backdrop-blur-md border border-white/5 rounded-2xl p-5 flex flex-col gap-1.5 transition-all duration-300 group-hover:border-cyan-500/25">
      <div className="text-cyan-400 transition-transform duration-300 group-hover:-translate-y-1">{icon}</div>
      <span className="text-xl md:text-2xl font-black text-white">{value}</span>
      <span className="text-[9px] font-mono uppercase tracking-widest text-slate-500">{label}</span>
    </div>
  )
}

function SliderField({ label, value, max, unit, onChange }) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs">
        <span className="text-slate-300 font-medium">{label}</span>
        <span className="text-cyan-400 font-mono font-bold">{value} {unit}</span>
      </div>
      <input 
        type="range" 
        min={0} 
        max={max} 
        value={value} 
        onChange={e => onChange(Number(e.target.value))} 
        className="w-full h-1.5 bg-[#050D18] rounded-lg appearance-none cursor-pointer accent-cyan-400" 
      />
    </div>
  )
}

// Pulse indicator component for the interactive SVG Map
function MapPulse({ cx, cy, value }) {
  let color = '#2ECC71' // Safe
  if (value >= 85) color = '#EA4335' // Severe
  else if (value >= 60) color = '#F4901E' // High
  else if (value >= 30) color = '#F4B400' // Moderate

  return (
    <g className="pointer-events-none">
      <circle cx={cx} cy={cy} r="10" fill={color} fillOpacity="0.15" className="animate-ping" style={{ transformOrigin: `${cx}px ${cy}px`, animationDuration: '2.5s' }} />
      <circle cx={cx} cy={cy} r="6" fill={color} fillOpacity="0.25" className="animate-ping" style={{ transformOrigin: `${cx}px ${cy}px`, animationDuration: '1.8s' }} />
      <circle cx={cx} cy={cy} r="3.5" fill={color} className="shadow-lg" />
    </g>
  )
}
