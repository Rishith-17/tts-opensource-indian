import { useEffect, useRef } from 'react'
import './Waveform.css'

export default function Waveform({ playing }) {
  const bars = Array.from({ length: 32 })

  return (
    <div className={`waveform ${playing ? 'playing' : ''}`}>
      {bars.map((_, i) => (
        <div
          key={i}
          className="bar"
          style={{ animationDelay: `${(i * 0.04).toFixed(2)}s` }}
        />
      ))}
    </div>
  )
}
