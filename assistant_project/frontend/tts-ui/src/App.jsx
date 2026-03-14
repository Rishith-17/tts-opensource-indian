import { useState, useEffect } from 'react'
import Header from './components/Header'
import TTSPanel from './components/TTSPanel'
import HistoryPanel from './components/HistoryPanel'
import StatusBar from './components/StatusBar'
import './App.css'

const API = 'http://localhost:8000'

export default function App() {
  const [apiOnline, setApiOnline] = useState(null)
  const [history, setHistory] = useState([])

  // health check on mount
  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => setApiOnline(d.tts_loaded))
      .catch(() => setApiOnline(false))
  }, [])

  function addToHistory(entry) {
    setHistory(prev => [entry, ...prev].slice(0, 20))
  }

  return (
    <div className="app">
      <Header apiOnline={apiOnline} />
      <main className="main">
        <TTSPanel api={API} onResult={addToHistory} />
        <HistoryPanel history={history} onClear={() => setHistory([])} />
      </main>
      <StatusBar apiOnline={apiOnline} />
    </div>
  )
}
