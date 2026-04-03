import { useState, useRef, useEffect, useCallback } from 'react'
import { Bot, User, Loader2, Volume2, PhoneOff, Phone, Send, RotateCcw } from 'lucide-react'
import './App.css'

const API        = import.meta.env.VITE_API_URL || 'http://localhost:8001'
const SESSION_ID = 'cc_' + Math.random().toString(36).slice(2, 8)

const LANGS = [
  { code:'kn', name:'ಕನ್ನಡ',   label:'Kannada', bcp:'kn-IN' },
  { code:'hi', name:'हिंदी',    label:'Hindi',   bcp:'hi-IN' },
  { code:'ta', name:'தமிழ்',   label:'Tamil',   bcp:'ta-IN' },
  { code:'te', name:'తెలుగు',  label:'Telugu',  bcp:'te-IN' },
  { code:'en', name:'English', label:'English', bcp:'en-US' },
]

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition

export default function App() {
  const [messages, setMessages]     = useState([])
  const [status, setStatus]         = useState('idle')
  const [online, setOnline]         = useState(null)
  const [callActive, setCallActive] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [inputText, setInputText]   = useState('')
  const [lang, setLang]             = useState('en')

  const recRef      = useRef(null)
  const audioRef    = useRef(null)
  const bottomRef   = useRef(null)
  const activeRef   = useRef(false)
  const statusRef   = useRef('idle')
  const langRef     = useRef('en')
  const watchRef    = useRef(null)

  useEffect(() => { statusRef.current = status }, [status])
  useEffect(() => { langRef.current = lang }, [lang])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:'smooth' }) }, [messages, transcript])

  useEffect(() => {
    fetch(`${API}/health`).then(r=>r.json()).then(()=>setOnline(true)).catch(()=>setOnline(false))
    // No auto-greeting — wait for human to speak/type first
  }, [])

  function addMsg(role, text, language, audio=null) {
    setMessages(m => [...m, { id: Date.now()+Math.random(), role, text, language, audio, time: new Date() }])
  }

  // ── Language switch — clears memory ──────────────────────────────────────
  async function switchLang(code) {
    if (code === lang) return
    setLang(code)
    langRef.current = code
    await fetch(`${API}/session/${SESSION_ID}/switch`, { method:'POST' }).catch(()=>{})
    setMessages([])
    // Wait for human to speak first
  }

  // ── Send to agent ─────────────────────────────────────────────────────────
  const sendToAgent = useCallback(async (text) => {
    if (!text.trim()) { if (activeRef.current) startListening(); return }
    const isGreeting = text === '__greeting__'
    if (!isGreeting) addMsg('user', text, langRef.current)
    setStatus('thinking'); statusRef.current = 'thinking'
    setTranscript('')
    recRef.current?.stop()
    clearWatch()

    try {
      const res  = await fetch(`${API}/chat`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ message:text, session_id:SESSION_ID, voice:'male', language:langRef.current }),
      })
      const data = await res.json()
      const url  = data.audio_base64 ? b64url(data.audio_base64) : null
      addMsg('agent', data.agent_response, data.language, url)

      if (url) {
        setStatus('speaking'); statusRef.current = 'speaking'
        const a = new Audio(url)
        audioRef.current = a
        a.onended = () => { if (activeRef.current) { setTimeout(startListening, 600) } else setStatus('idle') }
        a.play()
      } else {
        if (activeRef.current) startListening(); else setStatus('idle')
      }
    } catch {
      addMsg('agent', 'Sorry, connection error. Please try again.', 'en')
      if (activeRef.current) startListening(); else setStatus('idle')
    }
  }, [])

  // ── Speech recognition ────────────────────────────────────────────────────
  function startListening() {
    if (!SpeechRecognition) { setStatus('listening'); return }
    const rec = new SpeechRecognition()
    rec.continuous = false; rec.interimResults = true
    rec.lang = LANGS.find(l=>l.code===langRef.current)?.bcp || 'en-US'
    rec.onstart  = () => { setStatus('listening'); resetWatch() }
    rec.onresult = e => {
      resetWatch()
      let interim='', final=''
      for (let i=e.resultIndex; i<e.results.length; i++) {
        const t = e.results[i][0].transcript
        if (e.results[i].isFinal) final+=t; else interim+=t
      }
      setTranscript(interim||final)
      if (final) { rec.stop(); sendToAgent(final) }
    }
    rec.onerror = e => {
      if (activeRef.current && ['no-speech','network','aborted'].includes(e.error))
        setTimeout(startListening, 400)
    }
    rec.onend = () => {
      clearWatch()
      if (activeRef.current && !['thinking','speaking'].includes(statusRef.current))
        setTimeout(startListening, 300)
    }
    recRef.current = rec
    rec.start()
  }

  function resetWatch() {
    clearWatch()
    watchRef.current = setTimeout(() => {
      if (activeRef.current && statusRef.current==='listening') {
        recRef.current?.stop(); setTimeout(startListening, 200)
      }
    }, 5000)
  }
  function clearWatch() { if (watchRef.current) { clearTimeout(watchRef.current); watchRef.current=null } }

  function startCall()  { activeRef.current=true;  setCallActive(true);  startListening() }
  function endCall()    { activeRef.current=false; setCallActive(false); clearWatch(); recRef.current?.stop(); audioRef.current?.pause(); setStatus('idle'); setTranscript('') }

  function b64url(b64) {
    const b=atob(b64), a=new Uint8Array(b.length)
    for(let i=0;i<b.length;i++) a[i]=b.charCodeAt(i)
    return URL.createObjectURL(new Blob([a],{type:'audio/wav'}))
  }

  const statusText = { idle:'Ready', listening:'Listening...', thinking:'Thinking...', speaking:'Speaking...' }
  const statusColor = { idle:'#64748b', listening:'#10b981', thinking:'#f59e0b', speaking:'#6366f1' }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className={`bot-avatar ${status==='speaking'?'speaking':''}`}><Bot size={20}/></div>
          <div>
            <div className="brand-name">Customer Care AI</div>
            <div className="brand-status" style={{color: statusColor[status]}}>
              <span className="status-dot" style={{background: statusColor[status]}}/>
              {statusText[status]}
            </div>
          </div>
        </div>
        <div className="lang-bar">
          {LANGS.map(l => (
            <button key={l.code} className={`lang-btn ${lang===l.code?'active':''}`} onClick={()=>switchLang(l.code)}>
              <span className="lang-native">{l.name}</span>
              <span className="lang-label">{l.label}</span>
            </button>
          ))}
        </div>
      </header>

      {/* Messages */}
      <div className="chat">
        {messages.map(m => (
          <div key={m.id} className={`row ${m.role}`}>
            <div className={`avatar ${m.role}`}>{m.role==='agent'?<Bot size={14}/>:<User size={14}/>}</div>
            <div className="msg-wrap">
              <div className={`bubble ${m.role}`}>{m.text}</div>
              <div className="meta">
                <span>{m.time.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}</span>
                {m.audio && (
                  <button className="replay" onClick={()=>new Audio(m.audio).play()}>
                    <Volume2 size={11}/> replay
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}

        {transcript && (
          <div className="row user">
            <div className="avatar user"><User size={14}/></div>
            <div className="msg-wrap">
              <div className="bubble user live">{transcript}<span className="cur">|</span></div>
            </div>
          </div>
        )}

        {status==='thinking' && (
          <div className="row agent">
            <div className="avatar agent"><Bot size={14}/></div>
            <div className="bubble agent typing"><span/><span/><span/></div>
          </div>
        )}
        <div ref={bottomRef}/>
      </div>

      {/* Input */}
      <div className="input-area">
        <div className="text-row">
          <input className="input" value={inputText} onChange={e=>setInputText(e.target.value)}
            onKeyDown={e=>{if(e.key==='Enter'&&inputText.trim()){sendToAgent(inputText);setInputText('')}}}
            placeholder={`Message in ${LANGS.find(l=>l.code===lang)?.label}...`}
            disabled={status==='thinking'}
          />
          <button className="send-btn" onClick={()=>{if(inputText.trim()){sendToAgent(inputText);setInputText('')}}} disabled={!inputText.trim()||status==='thinking'}>
            <Send size={16}/>
          </button>
        </div>
        <div className="voice-row">
          <button className={`call-btn ${callActive?'end':''}`} onClick={callActive?endCall:startCall}>
            {callActive ? <><PhoneOff size={18}/> End Call</> : <><Phone size={18}/> Voice Call</>}
          </button>
          {status==='listening' && <div className="wave"><span/><span/><span/><span/><span/></div>}
          {status==='speaking'  && <div className="wave speaking"><span/><span/><span/><span/><span/></div>}
          {status==='thinking'  && <Loader2 size={16} className="spin" style={{color:'#f59e0b'}}/>}
          <button className="reset-btn" onClick={()=>switchLang(lang)} title="Clear conversation">
            <RotateCcw size={14}/>
          </button>
        </div>
      </div>
    </div>
  )
}
