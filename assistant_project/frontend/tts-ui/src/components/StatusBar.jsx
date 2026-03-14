import './StatusBar.css'

export default function StatusBar({ apiOnline }) {
  return (
    <footer className="status-bar">
      <div className="status-inner">
        <div className={`status-dot ${apiOnline === null ? 'checking' : apiOnline ? 'online' : 'offline'}`} />
        <span className="status-text">
          {apiOnline === null ? 'Connecting to API…' : apiOnline ? 'API Online — TTS Ready' : 'API Offline — Start python api_server.py'}
        </span>
        <span className="status-sep">·</span>
        <span className="status-text muted">localhost:8000</span>
        <span className="status-sep">·</span>
        <span className="status-text muted">Kokoro · MMS · Indic TTS</span>
      </div>
    </footer>
  )
}
