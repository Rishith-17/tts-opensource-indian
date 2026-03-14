import { useState, useRef } from 'react'
import { Volume2, Loader2, Sparkles, ChevronDown } from 'lucide-react'
import Waveform from './Waveform'
import './TTSPanel.css'

const LANGUAGES = [
  { code: 'kn', name: 'Kannada', flag: '🇮🇳', script: 'ಕನ್ನಡ' },
  { code: 'hi', name: 'Hindi',   flag: '🇮🇳', script: 'हिंदी' },
  { code: 'en', name: 'English', flag: '🇬🇧', script: 'English' },
  { code: 'ta', name: 'Tamil',   flag: '🇮🇳', script: 'தமிழ்' },
  { code: 'te', name: 'Telugu',  flag: '🇮🇳', script: 'తెలుగు' },
]

const SPEAKERS = [
  { value: 'female',      label: '👩 Female',            desc: 'Natural female voice' },
  { value: 'male',        label: '👨 Male',              desc: 'Natural male voice' },
  { value: 'female_slow', label: '👩 Female — Clear',    desc: 'Slower, clearer pronunciation' },
  { value: 'male_slow',   label: '👨 Male — Clear',      desc: 'Slower, clearer pronunciation' },
]

const SAMPLES = {
  kn: 'ನಮಸ್ಕಾರ, ನಾನು ನಿಮ್ಮ ಬಹುಭಾಷಾ ಧ್ವನಿ ಸಹಾಯಕ.',
  hi: 'नमस्ते, मैं आपका बहुभाषी वॉयस असिस्टेंट हूं।',
  en: 'Hello, I am your multilingual voice assistant.',
  ta: 'வணக்கம், நான் உங்கள் பன்மொழி குரல் உதவியாளர்.',
  te: 'నమస్కారం, నేను మీ బహుభాషా వాయిస్ అసిస్టెంట్.',
}

export default function TTSPanel({ api, onResult }) {
  const [text, setText] = useState('')
  const [lang, setLang] = useState('kn')
  const [speaker, setSpeaker] = useState('female')
  const [loading, setLoading] = useState(false)
  const [audioUrl, setAudioUrl] = useState(null)
  const [error, setError] = useState(null)
  const [playing, setPlaying] = useState(false)
  const audioRef = useRef(null)

  async function handleGenerate() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setAudioUrl(null)

    try {
      const res = await fetch(`${api}/tts/base64`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language: lang, speaker }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'TTS failed')
      }

      const data = await res.json()
      const blob = b64toBlob(data.audio_base64, 'audio/wav')
      const url = URL.createObjectURL(blob)
      setAudioUrl(url)

      onResult({ text, lang, speaker, url, time: new Date() })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function b64toBlob(b64, mime) {
    const bytes = atob(b64)
    const arr = new Uint8Array(bytes.length)
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i)
    return new Blob([arr], { type: mime })
  }

  function handlePlay() {
    if (audioRef.current) {
      audioRef.current.play()
      setPlaying(true)
    }
  }

  const selectedLang = LANGUAGES.find(l => l.code === lang)
  const charCount = text.length

  return (
    <div className="tts-panel">
      {/* Language selector */}
      <div className="section-label">Language</div>
      <div className="lang-grid">
        {LANGUAGES.map(l => (
          <button
            key={l.code}
            className={`lang-btn ${lang === l.code ? 'active' : ''}`}
            onClick={() => { setLang(l.code); setText(SAMPLES[l.code]) }}
          >
            <span className="lang-flag">{l.flag}</span>
            <span className="lang-name">{l.name}</span>
            <span className="lang-script">{l.script}</span>
          </button>
        ))}
      </div>

      {/* Text input */}
      <div className="section-label" style={{ marginTop: 24 }}>
        Text Input
        <span className="char-count">{charCount} chars</span>
      </div>
      <div className="textarea-wrap">
        <textarea
          className="tts-textarea"
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder={`Type in ${selectedLang?.name}…`}
          rows={5}
          maxLength={1000}
        />
        <button
          className="sample-btn"
          onClick={() => setText(SAMPLES[lang])}
          title="Load sample text"
        >
          <Sparkles size={14} /> Sample
        </button>
      </div>

      {/* Speaker selector */}
      <div className="section-label" style={{ marginTop: 20 }}>Voice</div>
      <div className="speaker-grid">
        {SPEAKERS.map(s => (
          <button
            key={s.value}
            className={`speaker-btn ${speaker === s.value ? 'active' : ''}`}
            onClick={() => setSpeaker(s.value)}
          >
            <span className="speaker-label">{s.label}</span>
            <span className="speaker-desc">{s.desc}</span>
          </button>
        ))}
      </div>

      {/* Generate button */}
      <button
        className={`generate-btn ${loading ? 'loading' : ''}`}
        onClick={handleGenerate}
        disabled={loading || !text.trim()}
      >
        {loading ? (
          <><Loader2 size={18} className="spin" /> Generating…</>
        ) : (
          <><Volume2 size={18} /> Generate Speech</>
        )}
      </button>

      {/* Error */}
      {error && <div className="error-box">⚠ {error}</div>}

      {/* Audio player */}
      {audioUrl && (
        <div className="audio-card">
          <Waveform playing={playing} />
          <audio
            ref={audioRef}
            src={audioUrl}
            controls
            autoPlay
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
            className="audio-native"
          />
          <a className="download-btn" href={audioUrl} download={`tts_${lang}.wav`}>
            ↓ Download WAV
          </a>
        </div>
      )}
    </div>
  )
}
