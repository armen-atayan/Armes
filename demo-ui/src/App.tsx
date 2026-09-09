import { FormEvent, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { initialCallState, reduceCallEvent, statusLabel } from './callState'
import { extractPhone } from './phone'
import type { CallEvent, CallHistoryItem, TaskInput } from './types'

const API = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''
const ACTIVE_SESSION = 't2.activeSession'
const ACTIVE_CONTACT = 't2.activeContact'

function wsUrl(sessionId: string, seq: number) {
  const root = API ? new URL(API, window.location.href) : new URL(window.location.href)
  root.protocol = root.protocol === 'https:' ? 'wss:' : 'ws:'
  root.pathname = `${root.pathname.replace(/\/$/, '')}/api/calls/${encodeURIComponent(sessionId)}/events`
  root.search = seq ? `?after=${seq}` : ''
  return root.toString()
}

function useElapsed(active: boolean, startedAt?: number) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (!active || !startedAt) return
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [active, startedAt])
  const seconds = startedAt ? Math.max(0, Math.floor((now - startedAt) / 1000)) : 0
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
}

function StatusIcon({ status }: { status: string }) {
  return <span className={`status-dot ${status}`} aria-hidden="true"><span /></span>
}

function commandEnter(event: React.KeyboardEvent<HTMLTextAreaElement>, submit: () => void) {
  if (event.key !== 'Enter' || !event.metaKey || event.nativeEvent.isComposing) return
  event.preventDefault()
  submit()
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => localStorage.getItem(ACTIVE_SESSION) ?? '')
  const [contact, setContact] = useState(() => localStorage.getItem(ACTIVE_CONTACT) ?? 'Собеседник')
  const [state, dispatch] = useReducer(reduceCallEvent, initialCallState)
  const [startedAt, setStartedAt] = useState<number>()
  const [connection, setConnection] = useState<'online' | 'reconnecting'>('online')
  const [formError, setFormError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [custom, setCustom] = useState(false)
  const [customAnswer, setCustomAnswer] = useState('')
  const [hangingUp, setHangingUp] = useState(false)
  const [hangupError, setHangupError] = useState('')
  const currentSession = useRef(sessionId)
  currentSession.current = sessionId
  const [answering, setAnswering] = useState(false)
  const [liveInstruction, setLiveInstruction] = useState('')
  const [sendingInstruction, setSendingInstruction] = useState(false)
  const [followUpBusy, setFollowUpBusy] = useState<'continue' | 'cancel' | ''>('')
  const [followUpPromptOpen, setFollowUpPromptOpen] = useState(false)
  const [followUpInstruction, setFollowUpInstruction] = useState('')
  const [showComposer, setShowComposer] = useState(false)
  const [history, setHistory] = useState<CallHistoryItem[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [draft, setDraft] = useState('')
  const [pendingTask, setPendingTask] = useState('')
  const [pendingPhone, setPendingPhone] = useState('')
  const [chatStep, setChatStep] = useState<'task' | 'phone' | 'confirm'>('task')
  const [recordingVoice, setRecordingVoice] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const recorderRef = useRef<MediaRecorder | undefined>(undefined)
  const chunksRef = useRef<Blob[]>([])
  const scrollRef = useRef<HTMLDivElement>(null)
  const lastSeq = useRef(0)
  const preserveThreadOnNextSession = useRef(false)
  const elapsed = useElapsed(Boolean(sessionId && !state.ended), startedAt)

  useEffect(() => { lastSeq.current = state.lastSeq }, [state.lastSeq])
  useEffect(() => {
    if (sessionId || showComposer) return
    setHistoryLoading(true)
    fetch(`${API}/api/calls`)
      .then(response => { if (!response.ok) throw new Error(); return response.json() })
      .then(body => setHistory(Array.isArray(body.calls) ? body.calls : []))
      .catch(() => setFormError('Не удалось загрузить диалоги'))
      .finally(() => setHistoryLoading(false))
  }, [sessionId, showComposer])
  useEffect(() => {
    const node = scrollRef.current
    if (!node) return
    if (typeof node.scrollTo === 'function') node.scrollTo({ top: node.scrollHeight, behavior: 'smooth' })
    else node.scrollTop = node.scrollHeight
  }, [state.messages, state.question])

  useEffect(() => {
    if (preserveThreadOnNextSession.current) preserveThreadOnNextSession.current = false
    else dispatch({ session_id: sessionId || 'reset', seq: 0, timestamp: Date.now() / 1000, type: 'session.reset', payload: {} })
    lastSeq.current = 0
    if (!sessionId) return
    let disposed = false
    let socket: WebSocket | undefined
    let reconnectTimer: number | undefined
    const ingest = (event: CallEvent) => dispatch(event)

    async function hydrate() {
      try {
        const response = await fetch(`${API}/api/calls/${encodeURIComponent(sessionId)}`)
        if (!response.ok) throw new Error()
        const snapshot = await response.json()
        if (disposed) return
        setStartedAt((current) => current ?? (snapshot.created_at ? Number(snapshot.created_at) * 1000 : Date.now()))
        const events: CallEvent[] = Array.isArray(snapshot.events) ? snapshot.events : []
        events.sort((a, b) => a.seq - b.seq).forEach(ingest)
        if (!events.length && snapshot.status) ingest({ session_id: sessionId, seq: 0.5, timestamp: Date.now() / 1000, type: 'call.state', payload: { status: snapshot.status } })
      } catch { if (!disposed) setConnection('reconnecting') }
    }

    function connect() {
      if (disposed) return
      socket = new WebSocket(wsUrl(sessionId, lastSeq.current))
      socket.onopen = () => setConnection('online')
      socket.onmessage = (message) => {
        try {
          const parsed = JSON.parse(message.data) as CallEvent | { events?: CallEvent[] }
          if ('events' in parsed) parsed.events?.forEach(ingest)
          else if ('type' in parsed && 'seq' in parsed) ingest(parsed)
        } catch { /* Ignore malformed/untrusted stream messages. */ }
      }
      socket.onclose = () => {
        if (disposed || state.ended) return
        setConnection('reconnecting')
        reconnectTimer = window.setTimeout(connect, 1500)
      }
    }
    void hydrate().finally(connect)
    return () => { disposed = true; socket?.close(); if (reconnectTimer) clearTimeout(reconnectTimer) }
  }, [sessionId])

  async function hangup() {
    if (hangingUp || state.ended) return
    const target = sessionId
    setHangingUp(true); setHangupError('')
    try {
      const response = await fetch(`${API}/api/calls/${encodeURIComponent(target)}/hangup`, { method: 'POST' })
      if (!response.ok) throw new Error()
      const snapshot = await response.json()
      if (currentSession.current === target) snapshot.events.forEach((event: CallEvent) => dispatch(event))
    } catch {
      if (currentSession.current === target) setHangupError('Не удалось завершить звонок. Попробуйте ещё раз.')
    } finally { setHangingUp(false) }
  }

  async function submitCall(task: TaskInput) {
    setFormError(''); setSubmitting(true)
    try {
      const response = await fetch(`${API}/api/calls`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(task) })
      if (!response.ok) throw new Error()
      const call = await response.json()
      localStorage.setItem(ACTIVE_SESSION, call.session_id)
      localStorage.setItem(ACTIVE_CONTACT, task.contact_name)
      setContact(task.contact_name); setStartedAt(Date.now()); setSessionId(call.session_id)
      dispatch({ session_id: call.session_id, seq: 0.25, timestamp: Date.now() / 1000, type: 'call.dialing', payload: {} })
    } catch { setFormError('Не удалось начать звонок. Попробуйте ещё раз.') }
    finally { setSubmitting(false) }
  }

  function sendChatMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (chatStep === 'task') {
      const message = draft.trim()
      if (!message) return
      const suppliedPhone = extractPhone(message)
      setPendingTask(message); setDraft('')
      if (suppliedPhone) {
        setPendingPhone(suppliedPhone); setChatStep('confirm')
      } else {
        setChatStep('phone')
      }
      return
    }
    if (chatStep === 'phone') {
      const phone = draft.trim().replace(/[\s()-]/g, '')
      if (!new RegExp('^\\+[1-9]\\d{7,14}$').test(phone)) return setFormError('Введите номер в формате +7…')
      setPendingPhone(phone); setDraft(''); setFormError(''); setChatStep('confirm')
    }
  }

  function contactFromTask(task: string) {
    return contactFromTaskText(task)
  }

  async function startVoice() {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') return setFormError('Запись голоса недоступна в этом браузере')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder; chunksRef.current = []
      recorder.ondataavailable = event => { if (event.data.size) chunksRef.current.push(event.data) }
      recorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop()); setRecordingVoice(false); setTranscribing(true)
        try {
          const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
          const response = await fetch(`${API}/api/transcribe`, { method: 'POST', headers: { 'Content-Type': blob.type }, body: blob })
          if (!response.ok) throw new Error()
          const body = await response.json(); setDraft(String(body.text ?? ''))
        } catch { setFormError('Не удалось распознать голосовое') }
        finally { setTranscribing(false) }
      }
      recorder.start(); setRecordingVoice(true)
    } catch { setFormError('Нет доступа к микрофону') }
  }

  function stopVoice() { recorderRef.current?.stop() }

  async function answer(value: string) {
    if (!state.question || answering || !value.trim()) return
    setAnswering(true)
    try {
      const requestId = state.question.requestId
      const response = await fetch(`${API}/api/calls/${encodeURIComponent(sessionId)}/owner-response`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request_id: requestId, response: value.trim(), answer: value.trim() }),
      })
      if (!response.ok) throw new Error()
      dispatch({ session_id: sessionId, seq: state.lastSeq + 0.1, timestamp: Date.now() / 1000, type: 'owner.answer', payload: { request_id: requestId, answer: value.trim() } })
      setCustom(false); setCustomAnswer('')
    } catch { /* Keep the decision card visible for a safe retry. */ }
    finally { setAnswering(false) }
  }

  async function sendLiveInstruction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const text = liveInstruction.trim()
    if (!text || sendingInstruction || state.ended) return
    setSendingInstruction(true)
    try {
      const response = await fetch(`${API}/api/calls/${encodeURIComponent(sessionId)}/instructions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
      })
      if (!response.ok) throw new Error()
      setLiveInstruction('')
    } catch { setFormError('Не удалось передать поручение в текущий звонок') }
    finally { setSendingInstruction(false) }
  }

  async function startFollowUp(action: 'continue' | 'cancel', instruction = '') {
    if (!sessionId || followUpBusy) return
    setFollowUpBusy(action); setFormError('')
    try {
      const response = await fetch(`${API}/api/calls/${encodeURIComponent(sessionId)}/follow-up`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(instruction ? { action, instruction } : { action }),
      })
      if (!response.ok) throw new Error()
      const call = await response.json()
      localStorage.setItem(ACTIVE_SESSION, call.session_id)
      localStorage.setItem(ACTIVE_CONTACT, contact)
      preserveThreadOnNextSession.current = true
      dispatch({ session_id: call.session_id, seq: 0, timestamp: Date.now() / 1000, type: 'session.followup', payload: { action } })
      lastSeq.current = 0
      setStartedAt(Date.now()); setSessionId(call.session_id)
    } catch { setFormError(action === 'cancel' ? 'Не удалось начать звонок для отмены' : 'Не удалось перезвонить') }
    finally { setFollowUpBusy('') }
  }

  function openCall(call: CallHistoryItem) {
    localStorage.setItem(ACTIVE_SESSION, call.session_id)
    localStorage.setItem(ACTIVE_CONTACT, call.phone_number || call.contact_name)
    setContact(call.phone_number || call.contact_name); setStartedAt(call.created_at * 1000); setShowComposer(false)
    setSessionId(call.session_id)
  }

  function reset() {
    localStorage.removeItem(ACTIVE_SESSION); localStorage.removeItem(ACTIVE_CONTACT)
    setSessionId(''); setContact('Собеседник'); setStartedAt(undefined); setShowComposer(false); window.location.hash = ''
  }

  const status = useMemo(() => connection === 'reconnecting' ? 'Восстанавливаем связь' : statusLabel(state.status), [connection, state.status])

  return <main className="stage">
    <section className="phone" aria-label="Личный ассистент">
      <div className="phone-buttons left-one"/><div className="phone-buttons left-two"/><div className="phone-buttons right-one"/>
      <div className="screen">
        <div className="island" aria-hidden="true" />
        <div className="status-bar"><span>9:41</span><span className="system-icons">▮▮▮ ◉ ▰</span></div>
        {!sessionId && !showComposer ? <HistoryView calls={history} loading={historyLoading} onNew={() => setShowComposer(true)} onOpen={openCall}/> : !sessionId ? <ChatComposer step={chatStep} draft={draft} setDraft={setDraft} task={pendingTask} phone={pendingPhone} error={formError} busy={submitting} recording={recordingVoice} transcribing={transcribing} onSubmit={sendChatMessage} onVoice={recordingVoice ? stopVoice : startVoice} onBack={() => setShowComposer(false)} onCall={() => submitCall({ contact_name: contactFromTask(pendingTask), phone_number: pendingPhone, task: pendingTask, details: '' })}/> :
          <CallView hangup={hangup} hangingUp={hangingUp} hangupError={hangupError} contact={contact} status={status} elapsed={elapsed} state={state} custom={custom} setCustom={setCustom} customAnswer={customAnswer} setCustomAnswer={setCustomAnswer} answer={answer} dismissQuestion={() => dispatch({ type: 'question.dismiss' })} answering={answering} reset={reset} scrollRef={scrollRef} liveInstruction={liveInstruction} setLiveInstruction={setLiveInstruction} sendLiveInstruction={sendLiveInstruction} sendingInstruction={sendingInstruction} startFollowUp={startFollowUp} followUpBusy={followUpBusy} followUpPromptOpen={followUpPromptOpen} setFollowUpPromptOpen={setFollowUpPromptOpen} followUpInstruction={followUpInstruction} setFollowUpInstruction={setFollowUpInstruction}/>
        }
        <div className="home-indicator" aria-hidden="true" />
      </div>
    </section>
  </main>
}

function ChatComposer({ step, draft, setDraft, task, phone, error, busy, recording, transcribing, onSubmit, onVoice, onBack, onCall }: any) {
  const placeholder = step === 'task' ? 'Напишите поручение…' : 'Введите номер +7…'
  return <div className="page chat-compose-page"><header className="chat-top"><button className="icon-button back-button" onClick={onBack} aria-label="Назад">‹</button><div><strong>Новый диалог</strong><small>готов к звонку</small></div></header>
    <div className="setup-chat">
      <article className="setup-message assistant"><p>Что нужно поручить?</p><small>Опишите звонок одним сообщением или запишите голосовое.</small></article>
      {task && <article className="setup-message user"><p>{task}</p></article>}
      {step !== 'task' && <article className="setup-message assistant"><p>На какой номер позвонить?</p><small>Номер нужен каждый раз — контакты не сохраняются.</small></article>}
      {phone && <article className="setup-message user"><p>{phone}</p></article>}
      {step === 'confirm' && <article className="call-preview"><small>Поручение готово</small><strong>{contactFromTaskText(task)}</strong><p>{task}</p><span>{phone}</span><button className="primary" disabled={busy} onClick={onCall} aria-label={busy ? 'Начинаем' : 'Позвонить'}>{busy ? 'Начинаем…' : 'Позвонить'}</button></article>}
      {error && <p className="form-error" role="alert">{error}</p>}
    </div>
    {step !== 'confirm' && <form className="chat-input" onSubmit={onSubmit}><button type="button" className={`mic-button ${recording ? 'recording' : ''}`} onClick={onVoice} aria-label={recording ? 'Остановить запись' : 'Записать голосовое'}>{recording ? '■' : '●'}</button>{step === 'phone' ? <input aria-label="Номер телефона" inputMode="tel" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder={placeholder}/> : <textarea aria-label="Сообщение" rows={1} value={draft} onKeyDown={(event) => commandEnter(event, () => event.currentTarget.form?.requestSubmit())} onChange={(e) => { setDraft(e.target.value); e.target.style.height = 'auto'; e.target.style.height = `${Math.min(e.target.scrollHeight, 132)}px` }} placeholder={transcribing ? 'Распознаём…' : placeholder}/>}<button className="send-button" disabled={!draft.trim() || transcribing} aria-label={step === 'phone' ? 'Продолжить' : 'Отправить'}>↑</button></form>}
  </div>
}

function contactFromTaskText(task: string) {
  return task.match(/(?:позвони|набери|свяжись с)\s+([А-ЯЁA-Z][а-яёa-z-]+)/iu)?.[1] ?? 'Собеседник'
}

function HistoryView({ calls, loading, onNew, onOpen }: { calls: CallHistoryItem[]; loading: boolean; onNew: () => void; onOpen: (call: CallHistoryItem) => void }) {
  return <div className="page history-page"><header className="history-header"><div><small>Звонки</small><h1>Диалоги</h1></div><button className="new-chat-button" onClick={onNew} aria-label="Новый диалог">＋</button></header>
    <div className="history-list">{loading ? <p className="history-empty">Загружаем…</p> : calls.length ? calls.map(call => <button className="history-item" key={call.session_id} onClick={() => onOpen(call)} aria-label={`${call.phone_number || call.contact_name}: ${call.task}`}><div><strong>{call.phone_number || call.contact_name}</strong><time>{new Date(call.created_at * 1000).toLocaleString('ru-RU', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}</time></div><p>{call.task || 'Звонок без описания'}</p><span>{statusLabel(call.status)} <b>›</b></span></button>) : <div className="history-empty"><strong>Диалогов пока нет</strong><span>Нажмите плюс, чтобы создать первый звонок</span></div>}</div>
  </div>
}

function CallView({ hangup, hangingUp, hangupError, contact, status, elapsed, state, custom, setCustom, customAnswer, setCustomAnswer, answer, dismissQuestion, answering, reset, scrollRef, liveInstruction, setLiveInstruction, sendLiveInstruction, sendingInstruction, startFollowUp, followUpBusy, followUpPromptOpen, setFollowUpPromptOpen, followUpInstruction, setFollowUpInstruction }: any) {
  return <div className="page call-page">
    <header className="call-header"><button className="icon-button" onClick={reset} aria-label="Новая задача">‹</button><div><h1>{contact}</h1><p><StatusIcon status={state.status}/>{status}</p></div><div className="call-controls"><time>{elapsed}</time>{!state.ended && ['connected', 'listening', 'thinking', 'speaking', 'waiting_owner'].includes(state.status) && <button type="button" className="hangup-button" onClick={hangup} disabled={hangingUp} aria-busy={hangingUp}>{hangingUp ? 'Завершаем…' : 'Положить трубку'}</button>}</div></header>
    {hangupError && !state.ended && <p className="form-error" role="alert">{hangupError}</p>}
    <div className="conversation" ref={scrollRef} aria-live="polite">
      <div className="call-start"><span>Звонок начат</span></div>
      {state.archivedCalls.map((call: any, index: number) => <div className="archived-call" key={`archived-${index}`}>
        {call.messages.map((message: any) => message.role === 'owner' ? <div className="owner-answer" key={message.id}><small>Решение Армена</small><strong>{message.text}</strong></div> :
          <article key={`${message.role}-${message.id}`} className={`message ${message.role}`}><small>{message.role === 'caller' ? contact : 'Ассистент Армена'}</small><p>{message.text}</p>{message.interrupted && <em>Фраза прервана</em>}</article>)}
        <section className="result-card previous-result"><small>Предыдущий звонок завершён · {call.outcome.outcome}</small>{call.outcome.summary && <p>{call.outcome.summary}</p>}{call.outcome.nextStep && <><hr/><small>Следующий шаг</small><strong>{call.outcome.nextStep}</strong></>}{call.recording && <audio controls preload="metadata" src={call.recording}/>}</section>
      </div>)}
      {state.threadNotice && <div className="call-start continuation"><span>{state.threadNotice}</span></div>}
      {state.messages.map((message: any) => message.role === 'owner' ? <div className="owner-answer" key={message.id}><small>Решение Армена</small><strong>{message.text}</strong></div> :
        <article key={`${message.role}-${message.id}`} className={`message ${message.role} ${message.partial ? 'partial' : ''}`}>
          <small>{message.role === 'caller' ? contact : 'Ассистент Армена'}</small><p>{message.text}{message.partial && <i className="typing"/>}</p>{message.interrupted && <em>Фраза прервана</em>}
        </article>)}
      {state.question && <section className="decision-card"><div className="decision-icon">?</div><small>Нужно ваше решение</small><h2>{state.question.question}</h2>{state.question.context && <p>{state.question.context}</p>}
        {!custom && state.question.options.length > 0 ? <div className="decision-actions dynamic">
          {state.question.options.map((option: string) => <button key={option} disabled={answering} onClick={() => answer(option)}>{option}</button>)}
          <button disabled={answering} onClick={() => setCustom(true)}>Другое</button>
          <button disabled={answering} onClick={dismissQuestion}>Промолчать</button>
        </div> :
        <div className="custom-answer"><label htmlFor="custom-answer">Ваш ответ</label><textarea id="custom-answer" autoFocus value={customAnswer} onKeyDown={(event) => commandEnter(event, () => { if (customAnswer.trim() && !answering) answer(customAnswer) })} onChange={(e) => setCustomAnswer(e.target.value)} placeholder="Напишите, что ответить…"/><div>{state.question.options.length > 0 && <button onClick={() => setCustom(false)}>Назад</button>}<button className="primary" disabled={!customAnswer.trim() || answering} onClick={() => answer(customAnswer)}>Отправить</button></div></div>}
      </section>}
      {state.outcome && <section className="result-card chat-result"><small>Звонок завершён · {state.outcome.outcome}</small>{state.outcome.summary && <p>{state.outcome.summary}</p>}{state.outcome.nextStep && <><hr/><small>Следующий шаг</small><strong>{state.outcome.nextStep}</strong></>}{state.recording && <audio controls preload="metadata" src={state.recording}/>}<div className="follow-up-actions">{!followUpPromptOpen ? <button className="primary" disabled={Boolean(followUpBusy)} onClick={() => setFollowUpPromptOpen(true)}>Перезвонить</button> : <div className="follow-up-prompt"><label htmlFor="follow-up-question">Что спросить?</label><textarea id="follow-up-question" autoFocus value={followUpInstruction} onKeyDown={(event) => commandEnter(event, () => { if (followUpInstruction.trim() && !followUpBusy) startFollowUp('continue', followUpInstruction.trim()) })} onChange={(event) => setFollowUpInstruction(event.target.value)} placeholder="Введите уточнение для агента…"/><div><button type="button" onClick={() => { setFollowUpPromptOpen(false); setFollowUpInstruction('') }}>Назад</button><button type="button" className="primary" disabled={!followUpInstruction.trim() || Boolean(followUpBusy)} onClick={() => startFollowUp('continue', followUpInstruction.trim())}>{followUpBusy === 'continue' ? 'Перезваниваем…' : 'Позвонить с уточнением'}</button></div></div>}{state.outcome.outcome.trim().toLowerCase() === 'agreed' && <button className="primary black-action" disabled={Boolean(followUpBusy)} onClick={() => startFollowUp('cancel')}>{followUpBusy === 'cancel' ? 'Звоним для отмены…' : 'Отменить договорённость'}</button>}</div></section>}
      {state.error && <div className="call-error"><strong>Звонок не состоялся</strong><span>{state.error}</span><button onClick={reset}>Новая задача</button></div>}
    </div>
    {!state.ended && <form className="live-instruction" onSubmit={sendLiveInstruction}><textarea rows={1} aria-label="Новое поручение" value={liveInstruction} onKeyDown={(event) => commandEnter(event, () => event.currentTarget.form?.requestSubmit())} onChange={(event) => setLiveInstruction(event.target.value)} placeholder="Добавить поручение в этот звонок…"/><button disabled={!liveInstruction.trim() || sendingInstruction} aria-label="Отправить поручение">↑</button></form>}
    <footer className="privacy-note"><span>●</span> Разговор отображается в реальном времени</footer>
  </div>
}

function ResultView({ contact, outcome, recording, onReset }: any) {
  return <div className="page result-page"><header><div className="result-check">✓</div><small>Звонок завершён</small><h1>{outcome.outcome}</h1><p>{contact}</p></header>
    <section className="result-card"><small>Итог</small>{outcome.summary && <p>{outcome.summary}</p>}{outcome.nextStep && <><hr/><small>Следующий шаг</small><strong>{outcome.nextStep}</strong></>}</section>
    {recording && <section className="recording"><div><span className="play-icon">▶</span><div><strong>Запись разговора</strong><small>Можно прослушать ещё раз</small></div></div><audio controls preload="metadata" src={recording}/></section>}
    <button className="primary" onClick={onReset}>Новая задача</button>
  </div>
}
