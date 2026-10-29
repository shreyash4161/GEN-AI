import { useEffect, useMemo, useState } from 'react'
import {
  Activity, ArrowDownRight, ArrowUpRight, BookOpen, Check, ChevronDown,
  CircleAlert, Cpu, Gauge, HeartPulse, Languages, LoaderCircle,
  MessageSquareText, RotateCcw, ShieldCheck, Sparkles, Zap, Database,
  FlaskConical, BarChart3, FileText, LayoutDashboard,
} from 'lucide-react'
import { AboutPage, ComparisonPage, DatasetPage, ExperimentsPage, OverviewPage, ReportPage, SyntheticLabPage } from './ResearchWorkspace'

const API = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
const SAMPLE_EXAMPLES = [
  { id: 'hate-devanagari', label: 'HATE', tone: 'coral', style: 'Devanagari Marathi', text: 'त्या काल्पनिक गटातील लोकांना गावातून हाकलून द्या.' },
  { id: 'hate-romanized', label: 'HATE', tone: 'coral', style: 'Romanized Marathi', text: 'Tya kalpanik ghatatil lokanna gavatoon hakalun dya.' },
  { id: 'offensive-devanagari', label: 'OFFENSIVE', tone: 'orange', style: 'Devanagari Marathi', text: 'तू अगदी मूर्ख आहेस; तुझं बोलणं निरर्थक आहे.' },
  { id: 'offensive-mixed', label: 'OFFENSIVE', tone: 'orange', style: 'Marathi-English mix', text: 'तू खूप rude आहेस; तुझं बोलणं काही कामाचं नाही.' },
  { id: 'profane-devanagari', label: 'PROFANE', tone: 'violet', style: 'Devanagari Marathi', text: 'अरे, हे काय फालतू काम आहे!' },
  { id: 'profane-romanized', label: 'PROFANE', tone: 'violet', style: 'Romanized Marathi', text: 'Are, he kay faltu kaam aahe!' },
  { id: 'not-devanagari', label: 'NOT ABUSIVE', tone: 'mint', style: 'Devanagari Marathi', text: 'आजचा दिवस खूप सुंदर आहे…' },
  { id: 'not-mixed', label: 'NOT ABUSIVE', tone: 'mint', style: 'Marathi-English mix', text: 'आजचा दिवस खूप सुंदर आहे, feeling peaceful!' },
]
const LABEL_STYLES = {
  hate: { tone: 'coral', short: 'Group-directed abuse' },
  offensive: { tone: 'orange', short: 'Individual-directed insult' },
  profane: { tone: 'violet', short: 'Profanity / swear words' },
  not: { tone: 'mint', short: 'No abuse detected' },
}

function pct(value, digits = 1) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : '—'
}

function App() {
  const [models, setModels] = useState([])
  const [labels, setLabels] = useState([])
  const [modelId, setModelId] = useState('')
  const [device, setDevice] = useState('checking')
  const [backendError, setBackendError] = useState('')
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  const selectedModel = useMemo(() => models.find((model) => model.id === modelId) || models[0], [models, modelId])

  useEffect(() => {
    let alive = true
    async function loadStatus() {
      try {
        const response = await fetch(`${API}/api/models`)
        if (!response.ok) throw new Error(`API returned ${response.status}`)
        const data = await response.json()
        if (!alive) return
        setModels(data.models || [])
        setLabels(data.labels || [])
        setModelId((current) => current || data.models?.[0]?.id || '')
        setDevice(data.device || 'cpu')
        setBackendError(data.models?.length ? '' : 'No completed model checkpoints were found.')
      } catch (exception) {
        if (!alive) return
        setBackendError('The inference API is not running. Start the backend in the project folder.')
        setDevice('offline')
      }
    }
    loadStatus()
    return () => { alive = false }
  }, [])

  async function analyze(event) {
    event.preventDefault()
    setError('')
    setResult(null)
    if (!text.trim()) {
      setError('Add Marathi, Romanized Marathi, or code-mixed text before analyzing.')
      return
    }
    setLoading(true)
    try {
      const response = await fetch(`${API}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim(), model_id: modelId || undefined }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || `Analysis failed (${response.status})`)
      setResult(data)
      setDevice(data.device || device)
    } catch (exception) {
      setError(exception.message || 'Unable to reach the local inference API.')
    } finally {
      setLoading(false)
    }
  }

  function reset() {
    setText('')
    setResult(null)
    setError('')
  }

  const classes = labels.length ? labels : [
    { id: 'hate', title: 'Hate speech', description: 'Abuse aimed at a group or community.' },
    { id: 'offensive', title: 'Offensive language', description: 'Insults or threats aimed at an individual.' },
    { id: 'profane', title: 'Profane language', description: 'Swear words or cursing language.' },
    { id: 'not', title: 'Not abusive', description: 'No clear abusive or profane content.' },
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#home" aria-label="Marathi NLP Lab home">
          <span className="brand-mark"><span>म</span></span>
          <span className="brand-copy"><strong>maha<span>lab</span></strong><small>MARATHI TEXT INTELLIGENCE</small></span>
        </a>

        <div className="sidebar-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Workspace navigation">
          <button title="Overview" className={activeTab === 'overview' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('overview')}>
            <LayoutDashboard size={17} /> <span>Overview</span><span className="nav-dot" />
          </button>
          <button title="Text classifier" className={activeTab === 'classifier' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('classifier')}>
            <MessageSquareText size={17} /> <span>Text classifier</span><span className="nav-dot" />
          </button>
          <button title="Synthetic data lab" className={activeTab === 'synthetic' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('synthetic')}>
            <Sparkles size={17} /> <span>Synthetic lab</span>
          </button>
          <button title="Model comparison" className={activeTab === 'comparison' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('comparison')}>
            <BarChart3 size={17} /> <span>Model comparison</span>
          </button>
          <button title="Experiments" className={activeTab === 'experiments' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('experiments')}>
            <FlaskConical size={17} /> <span>Experiments</span>
          </button>
          <button title="Dataset explorer" className={activeTab === 'dataset' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('dataset')}>
            <Database size={17} /> <span>Dataset</span>
          </button>
          <button title="Research report" className={activeTab === 'report' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('report')}>
            <FileText size={17} /> <span>Report</span>
          </button>
          <button title="About this project" className={activeTab === 'about' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('about')}>
            <BookOpen size={17} /> <span>About & scope</span>
          </button>
        </nav>

        <div className="side-model-card">
          <div className="side-card-top"><span className="pulse-dot" /> MODEL STATUS</div>
          {backendError ? <strong className="side-model-name">Needs attention</strong> : <strong className="side-model-name">{selectedModel?.name || 'Loading model…'}</strong>}
          <div className="side-card-foot"><Cpu size={13} /> {device === 'checking' ? 'Connecting…' : device === 'offline' ? 'API offline' : `${String(device).toUpperCase()} inference`}</div>
        </div>

        <div className="sidebar-bottom">
          <div className="sidebar-divider" />
          <div className="privacy-note"><ShieldCheck size={15} /><span>Runs locally on your machine.<br />Text is not sent to a cloud API.</span></div>
          <div className="sidebar-version">RESEARCH PROTOTYPE <span>v1.0</span></div>
        </div>
      </aside>

      <main className="main-content" id="home">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><strong>{({overview:'Overview',classifier:'Classifier',synthetic:'Synthetic lab',comparison:'Model comparison',experiments:'Experiments',dataset:'Dataset explorer',report:'Research report',about:'About & scope'})[activeTab]}</strong></div>
          <div className="topbar-right"><span className="api-status"><span className={device === 'checking' || backendError || !models.length ? 'status-light offline' : 'status-light'} />{device === 'checking' ? 'Connecting…' : backendError ? 'API disconnected' : models.length ? 'Inference API ready' : 'No model available'}</span><span className="topbar-divider" /><span className="topbar-locale"><Languages size={15} /> मराठी</span></div>
        </header>

        {activeTab === 'classifier' ? (
          <>
            <section className="page-intro">
              <div className="intro-kicker"><Sparkles size={14} /> MARATHI NLP · FOUR-CLASS DETECTION</div>
              <div className="intro-title-row"><div><h1>Read between<br className="mobile-break" /> the <em>words.</em></h1><p>Explore how a fine-tuned language model reads Marathi social text, from group-directed hate to everyday language.</p></div><div className="intro-stamp"><span>आवाज</span><small>LANGUAGE<br />MATTERS</small></div></div>
            </section>

            <section className="workspace-grid">
              <div className="analyzer-card">
                <div className="card-heading"><div><span className="section-index">01 / INPUT</span><h2>Analyze a text</h2></div><span className="input-limit">{text.length}<span> / 2,000</span></span></div>
                {backendError && <div className="inline-alert"><CircleAlert size={17} /><div><strong>Model connection needed</strong><span>{backendError}</span></div></div>}
                <form onSubmit={analyze}>
                  <label className="field-label" htmlFor="marathi-text">MARATHI · ROMANIZED · CODE-MIXED</label>
                  <div className="textarea-shell"><textarea id="marathi-text" value={text} onChange={(event) => setText(event.target.value.slice(0, 2000))} onKeyDown={(event) => { if (event.ctrlKey && event.key === 'Enter') event.currentTarget.form?.requestSubmit() }} placeholder="तुमचा मराठी मजकूर इथे लिहा किंवा पेस्ट करा…" maxLength={2000} rows={7} /><div className="textarea-footer"><span><Languages size={13} /> Devanagari and Latin text supported</span><span>Ctrl ↵ to analyze</span></div></div>
                  <details className="sample-library">
                    <summary><span><strong>TRY A SAMPLE</strong><small>8 examples · all four classes</small></span><ArrowDownRight size={14} /></summary>
                    <div className="sample-grid">{SAMPLE_EXAMPLES.map((sample) => <button className="sample-option" type="button" key={sample.id} onClick={() => { setText(sample.text); setResult(null); setError('') }}><span className={`sample-option-label ${sample.tone}`}>{sample.label}</span><span className="sample-option-style">{sample.style}</span><span className="sample-option-text">{sample.text}</span></button>)}</div>
                    <p className="sample-disclaimer">Illustrative examples only. The selected checkpoint produces the actual prediction.</p>
                  </details>
                  {error && <div className="form-error"><CircleAlert size={15} />{error}</div>}
                  <div className="form-actions"><button className="primary-button" type="submit" disabled={loading || !!backendError || !models.length}>{loading ? <><LoaderCircle size={17} className="spin" /> Running classifier…</> : <><Sparkles size={16} /> Analyze text <ArrowUpRight size={15} /></>}</button><button className="reset-button" type="button" onClick={reset} disabled={loading}><RotateCcw size={15} /> Clear</button></div>
                </form>
                <div className="model-picker"><div className="picker-label"><span>ACTIVE CHECKPOINT</span><span className="picker-best">VALIDATION-SELECTED</span></div><div className="select-wrap"><Cpu size={16} /><select aria-label="Select trained model" value={modelId} onChange={(event) => { setModelId(event.target.value); setResult(null) }} disabled={!models.length}>{models.map((model) => <option value={model.id} key={model.id}>{model.name} · {model.id}</option>)}</select><ChevronDown size={15} /></div>{selectedModel && <div className="selected-model-meta"><span>{selectedModel.train_rows.toLocaleString()} training examples</span><span>Val Macro-F1 {pct(selectedModel.validation_macro_f1, 3)}</span></div>}</div>
              </div>

              <div className="result-card" aria-live="polite">
                <div className="card-heading result-heading"><div><span className="section-index">02 / PREDICTION</span><h2>Model reading</h2></div>{result ? <span className="result-live"><span /> LIVE</span> : <span className="result-idle"><span className="idle-icon"><Zap size={13} /></span> AWAITING TEXT</span>}</div>
                {result ? <Prediction result={result} /> : <div className="empty-result"><div className="empty-orbit orbit-one" /><div className="empty-orbit orbit-two" /><div className="empty-icon"><HeartPulse size={25} strokeWidth={1.5} /></div><span className="empty-eyebrow">YOUR RESULT WILL APPEAR HERE</span><p>The model returns a four-way prediction with the probability for every class—not just a hate / not-hate label.</p><div className="empty-mini-labels"><span>HATE</span><i /><span>OFFENSIVE</span><i /><span>PROFANE</span><i /><span>NOT</span></div></div>}
                {result && <div className="result-footnote"><CircleAlert size={14} /><span>Probability is a model score, not a certainty. Review context before drawing conclusions.</span></div>}
              </div>
            </section>

            <section className="class-section"><div className="section-title-row"><div><span className="section-index">THE LABELS</span><h2>Four distinct kinds of language.</h2></div><span className="paper-source"><BookOpen size={14} /> Based on L3Cube-MahaHate</span></div><div className="class-grid">{classes.map((item, index) => <article className={`class-card ${LABEL_STYLES[item.id]?.tone || 'mint'}`} key={item.id}><div className="class-card-top"><span className="class-number">0{index + 1}</span><span className="class-code">{item.id.toUpperCase()}</span></div><h3>{item.title}</h3><p>{item.description}</p><div className="class-accent" /></article>)}</div></section>

            <section className="baseline-strip"><div className="baseline-label"><Gauge size={17} /><div><strong>Research baseline</strong><span>Held-out results from the saved run</span></div></div><div className="baseline-metric"><span>TEST MACRO-F1</span><strong>{pct(selectedModel?.test_macro_f1, 3)}</strong></div><div className="baseline-metric"><span>TEST ACCURACY</span><strong>{pct(selectedModel?.test_accuracy, 2)}</strong></div><div className="baseline-note">One 5% training subset and one seed. Results describe this run only.</div></section>
          </>
        ) : activeTab === 'overview' ? <OverviewPage />
          : activeTab === 'synthetic' ? <SyntheticLabPage models={models} />
            : activeTab === 'comparison' ? <ComparisonPage />
              : activeTab === 'experiments' ? <ExperimentsPage />
                : activeTab === 'dataset' ? <DatasetPage />
                  : activeTab === 'report' ? <ReportPage />
                    : activeTab === 'about' ? <AboutPage /> : (
          <ResearchNotes models={models} />
        )}

        <footer className="app-footer"><span>MAHALAB <span className="footer-dot">·</span> A MARATHI NLP RESEARCH INTERFACE</span><span>Built for exploration, not consequential decisions.</span></footer>
      </main>
    </div>
  )
}

function Prediction({ result }) {
  const style = LABEL_STYLES[result.predicted_label] || LABEL_STYLES.not
  return <div className="prediction-content"><div className={`prediction-banner ${style.tone}`}><span className="prediction-symbol">{result.predicted_label === 'not' ? <Check size={19} /> : <ShieldCheck size={19} />}</span><div className="prediction-main"><span>PREDICTED CLASS</span><h3>{result.predicted_title}</h3><p>{result.description}</p></div><div className="confidence"><strong>{pct(result.confidence)}</strong><span>MODEL SCORE</span></div></div><div className="probability-title"><span>CLASS PROBABILITIES</span><span>0—100%</span></div><div className="probability-list">{result.probabilities.map((item) => <div className="prob-row" key={item.label}><div className="prob-row-top"><span>{item.title}</span><strong>{pct(item.probability)}</strong></div><div className="prob-track"><div className={`prob-fill ${LABEL_STYLES[item.label]?.tone || 'mint'}`} style={{ width: `${Math.max(1, item.probability * 100)}%` }} /></div></div>)}</div><div className="inference-meta"><span><Cpu size={13} /> {result.model.name}</span><span><Zap size={12} /> {String(result.device).toUpperCase()}</span>{result.truncated && <span className="truncated-warning">Input was shortened to 128 model tokens</span>}</div></div>
}

function ResearchNotes({ models }) {
  return <section className="research-page"><div className="intro-kicker"><BookOpen size={14} /> PAPER-ALIGNED CONTEXT</div><h1>Research<br /><em>notes.</em></h1><p className="research-lede">The interface follows the four-class Marathi hate-speech task described by Velankar et al. in the L3Cube-MahaHate paper.</p><div className="research-paper-card"><div className="paper-mark">L3</div><div><span className="section-index">REFERENCE PAPER · 2022</span><h2>L3Cube-MahaHate: A Tweet-based Marathi Hate Speech Detection Dataset and BERT models</h2><p>Velankar, Patil, Gore, Salunke & Joshi · arXiv:2203.13778</p></div></div><div className="research-columns"><article className="research-info"><span className="section-index">WHAT THE PAPER REPORTS</span><h2>Four-class task</h2><p>The paper manually annotates Marathi tweets as hate, offensive, profane, or not. It reports balanced official splits: 21,500 train, 1,500 validation, and 2,000 test.</p><p>Its 2022 experiments report Macro-F1 0.803 for MahaRoBERTa and 0.802 for MahaBERT on the four-class task. Those are paper-reported figures from its original setup, not a direct comparison with the local run.</p></article><article className="research-info"><span className="section-index">THIS LOCAL IMPLEMENTATION</span><h2>Fine-tuned checkpoints</h2>{models.map((model) => <div className="run-row" key={model.id}><span className="run-dot" /><div><strong>{model.name}</strong><small>{model.train_rows.toLocaleString()} rows · validation Macro-F1 {pct(model.validation_macro_f1, 3)}</small></div><span className="run-score">{pct(model.test_macro_f1, 3)}</span></div>)}<p>Local runs use the same four label definitions and official dataset split, but a different training configuration. Test values are shown for reporting only and do not determine the selected checkpoint.</p></article></div><div className="method-note"><ShieldCheck size={17} /><p><strong>Research integrity.</strong> The test split is not used for checkpoint selection. Local results are a low-resource baseline, not a replication of the paper and not evidence that synthetic-data augmentation improves performance.</p></div></section>
}

export default App
