import { Mic2, Waves } from 'lucide-react'
import './Header.css'

export default function Header({ apiOnline }) {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="logo">
          <div className="logo-icon">
            <Waves size={22} />
          </div>
          <div>
            <span className="logo-title">VoiceStudio</span>
            <span className="logo-sub">Multilingual AI TTS</span>
          </div>
        </div>
        <div className="header-badges">
          <span className="badge">Kannada</span>
          <span className="badge">Hindi</span>
          <span className="badge">Tamil</span>
          <span className="badge">Telugu</span>
          <span className="badge accent">English</span>
        </div>
      </div>
    </header>
  )
}
