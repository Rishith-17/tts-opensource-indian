import { Clock, Trash2, Play } from 'lucide-react'
import './HistoryPanel.css'

const LANG_NAMES = { kn:'Kannada', hi:'Hindi', en:'English', ta:'Tamil', te:'Telugu' }

export default function HistoryPanel({ history, onClear }) {
  if (history.length === 0) {
    return (
      <aside className="history-panel empty">
        <div className="history-header">
          <Clock size={14} /> History
        </div>
        <div className="empty-state">
          <div className="empty-icon">🎙️</div>
          <p>Your generated audio will appear here</p>
        </div>
      </aside>
    )
  }

  return (
    <aside className="history-panel">
      <div className="history-header">
        <span><Clock size={14} /> History ({history.length})</span>
        <button className="clear-btn" onClick={onClear}>
          <Trash2 size={13} /> Clear
        </button>
      </div>

      <div className="history-list">
        {history.map((item, i) => (
          <div key={i} className="history-item">
            <div className="history-meta">
              <span className="history-lang">{LANG_NAMES[item.lang]}</span>
              <span className="history-speaker">{item.speaker}</span>
              <span className="history-time">
                {item.time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
            <p className="history-text">{item.text.slice(0, 80)}{item.text.length > 80 ? '…' : ''}</p>
            <audio src={item.url} controls className="history-audio" />
          </div>
        ))}
      </div>
    </aside>
  )
}
