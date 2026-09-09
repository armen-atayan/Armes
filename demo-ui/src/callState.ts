import type { CallEvent, CallOutcome, Message, OwnerQuestion } from './types'

export type CallState = {
  status: string
  messages: Message[]
  question?: OwnerQuestion
  outcome?: CallOutcome
  recording?: string
  lastSeq: number
  ended: boolean
  error?: string
  archivedCalls: Array<{ messages: Message[]; outcome: CallOutcome; recording?: string }>
  threadNotice?: string
}

export const initialCallState: CallState = { status: 'preparing', messages: [], archivedCalls: [], lastSeq: 0, ended: false }

const text = (payload: Record<string, unknown>, key = 'text') => typeof payload[key] === 'string' ? payload[key].trim() : ''
const rawText = (payload: Record<string, unknown>, key = 'text') => typeof payload[key] === 'string' ? payload[key] as string : ''
const turnId = (payload: Record<string, unknown>, fallback: string) => String(payload.utterance_id ?? payload.item_id ?? payload.message_id ?? fallback)
const statusMap: Record<string, string> = {
  preparing: 'Готовим звонок', dialing: 'Набираем номер', ringing: 'Ждём ответа', connected: 'На связи',
  listening: 'Слушаем', thinking: 'Обдумываем ответ', speaking: 'Говорим', waiting_owner: 'Ждём решения Армена',
  completed: 'Звонок завершён', failed: 'Не удалось позвонить',
}

function upsert(messages: Message[], next: Message, append = false) {
  const index = messages.findIndex((item) => item.id === next.id && item.role === next.role)
  if (index < 0) return [...messages, next]
  const copy = [...messages]
  copy[index] = { ...copy[index], ...next, text: append ? copy[index].text + next.text : next.text }
  return copy
}

const comparableAssistantText = (value: string) => value.toLocaleLowerCase('ru-RU').replace(/[!?.,…\s]+/g, '')

function reconcileAssistantFinal(messages: Message[], next: Message) {
  const exact = messages.findIndex((item) => item.id === next.id && item.role === 'assistant')
  if (exact >= 0) return upsert(messages, next)
  const normalized = comparableAssistantText(next.text)
  const previous = messages.length - 1
  if (previous < 0 || messages[previous].role !== 'assistant' || comparableAssistantText(messages[previous].text) !== normalized) {
    return [...messages, next]
  }
  const copy = [...messages]
  copy[previous] = next
  return copy
}

export function reduceCallEvent(state: CallState, event: CallEvent): CallState {
  if (event.type === 'session.reset') return initialCallState
  if (event.type === 'session.followup') return {
    ...initialCallState,
    status: 'dialing',
    archivedCalls: state.outcome
      ? [...state.archivedCalls, { messages: state.messages, outcome: state.outcome, recording: state.recording }]
      : state.archivedCalls,
    threadNotice: event.payload?.action === 'cancel'
      ? 'Перезваниваем, чтобы отменить договорённость'
      : 'Перезваниваем и продолжаем разговор',
  }
  if (!Number.isFinite(event.seq) || event.seq <= state.lastSeq) return state
  const payload = event.payload ?? {}
  const base = { ...state, lastSeq: event.seq }
  switch (event.type) {
    case 'call.created': return { ...base, status: 'preparing' }
    case 'call.dialing': return { ...base, status: 'dialing' }
    case 'call.connected': return { ...base, status: 'connected' }
    case 'call.state': return { ...base, status: String(payload.state ?? payload.status ?? state.status) }
    case 'transcript.caller.partial': {
      const value = text(payload); if (!value) return base
      return { ...base, messages: upsert(state.messages, { id: turnId(payload, `caller-${event.seq}`), role: 'caller', text: value, partial: true }) }
    }
    case 'transcript.caller.final': {
      const value = text(payload); if (!value) return base
      return { ...base, messages: upsert(state.messages, { id: turnId(payload, `caller-${event.seq}`), role: 'caller', text: value, partial: false }) }
    }
    case 'transcript.assistant.delta': {
      const value = rawText(payload) || rawText(payload, 'delta'); if (!value.trim()) return base
      return { ...base, messages: upsert(state.messages, { id: turnId(payload, 'assistant-active'), role: 'assistant', text: value, partial: true }, true) }
    }
    case 'transcript.assistant.final': {
      const value = text(payload); if (!value) return base
      const id = turnId(payload, 'assistant-active')
      return { ...base, messages: reconcileAssistantFinal(state.messages, { id, role: 'assistant', text: value, partial: false }) }
    }
    case 'transcript.assistant.interrupted': {
      const id = turnId(payload, 'assistant-active')
      return { ...base, messages: state.messages.map((item) => item.id === id ? { ...item, partial: false, interrupted: true } : item) }
    }
    case 'owner.question': {
      const options = Array.isArray(payload.options)
        ? payload.options.filter((item): item is string => typeof item === 'string' && Boolean(item.trim())).slice(0, 4)
        : []
      return { ...base, status: 'waiting_owner', question: {
        requestId: String(payload.request_id ?? ''), question: text(payload, 'question'),
        context: text(payload, 'context'), options,
      } }
    }
    case 'owner.instruction': {
      const instruction = text(payload)
      const instructionId = String(payload.instruction_id ?? event.seq)
      return { ...base, messages: instruction ? upsert(state.messages, { id: `instruction-${instructionId}`, role: 'owner', text: instruction }) : state.messages }
    }
    case 'owner.answer': {
      const answer = text(payload, 'answer') || text(payload, 'response')
      const requestId = String(payload.request_id ?? '')
      const id = requestId ? `owner-${requestId}` : `owner-${event.seq}`
      return { ...base, question: undefined, messages: answer ? upsert(state.messages, { id, role: 'owner', text: answer }) : state.messages }
    }
    case 'call.outcome': return { ...base, outcome: { outcome: text(payload, 'outcome') || text(payload, 'title') || 'Звонок завершён', summary: text(payload, 'summary'), nextStep: text(payload, 'next_step') } }
    case 'recording.ready': return { ...base, recording: text(payload, 'url') || `/api/calls/${event.session_id}/recording` }
    case 'call.ended': return { ...base, status: 'completed', ended: true }
    case 'call.failed': return { ...base, status: 'failed', ended: true, error: text(payload, 'message') || text(payload, 'error') || 'Попробуйте ещё раз' }
    default: return base
  }
}

export const statusLabel = (status: string) => statusMap[status] ?? 'Звонок продолжается'
