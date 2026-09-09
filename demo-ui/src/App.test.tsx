import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

class MockSocket {
  static instance: MockSocket
  onopen: (() => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: (() => void) | null = null
  readyState = 1
  constructor(public url: string) { MockSocket.instance = this }
  close() {}
  emit(data: unknown) { this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent) }
}

const createdCall = { session_id: 'demo_1', room_name: 'room_1' }
const event = (seq: number, type: string, payload: Record<string, unknown>) => ({ session_id: 'demo_1', room_name: 'room_1', seq, timestamp: 1, type, payload })

describe('T2 personal assistant UI', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.stubGlobal('WebSocket', MockSocket)
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/calls')) return new Response(JSON.stringify(createdCall), { status: 200 })
      if (url.includes('/instructions')) return new Response(JSON.stringify({ instruction_id: 'i1', status: 'queued' }), { status: 202 })
      if (url.includes('/owner-response')) return new Response('{}', { status: 200 })
      return new Response(JSON.stringify({ ...createdCall, status: 'dialing', events: [] }), { status: 200 })
    }))
  })
  afterEach(() => vi.unstubAllGlobals())

  it('opens on a phone-number-first conversation list with a plus button', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ calls: [
      { session_id: 'demo_old', contact_name: 'Артур', phone_number: '+799****4567', task: 'Узнать про машину', status: 'ended', created_at: 1700000000 },
    ] }), { status: 200 })))
    render(<App />)
    expect(await screen.findByText('+799****4567')).toBeInTheDocument()
    expect(screen.getByText('Узнать про машину')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Новый диалог' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'История' })).not.toBeInTheDocument()
  })

  it('opens a new task composer from the plus button', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ calls: [] }), { status: 200 })))
    render(<App />)
    await userEvent.click(await screen.findByRole('button', { name: 'Новый диалог' }))
    expect(screen.getByText('Что нужно поручить?')).toBeInTheDocument()
    expect(screen.getByLabelText('Сообщение')).toBeInTheDocument()
  })

  it('uses an auto-growing multiline field for a long task', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Новый диалог' }))
    const composer = screen.getByLabelText('Сообщение')
    expect(composer.tagName).toBe('TEXTAREA')
    Object.defineProperty(composer, 'scrollHeight', { configurable: true, value: 96 })
    fireEvent.change(composer, { target: { value: 'Длинное поручение\nсо второй строкой\nи третьей строкой' } })
    expect(composer).toHaveStyle({ height: '96px' })
  })

  it('asks for the phone number when the task does not contain one', async () => {
    render(<App />)
    await userEvent.click(screen.getByRole('button', { name: 'Новый диалог' }))
    expect(screen.getByText('Что нужно поручить?')).toBeInTheDocument()
    await userEvent.type(screen.getByLabelText('Сообщение'), 'Позвони Артуру и узнай, когда он продаст машину')
    await userEvent.click(screen.getByRole('button', { name: 'Отправить' }))
    expect(screen.getByText('На какой номер позвонить?')).toBeInTheDocument()
    await userEvent.type(screen.getByLabelText('Номер телефона'), '+79991234567')
    await userEvent.click(screen.getByRole('button', { name: 'Продолжить' }))
    expect(screen.getAllByText('Позвони Артуру и узнай, когда он продаст машину')).toHaveLength(2)
    await userEvent.click(screen.getByRole('button', { name: 'Позвонить' }))
    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/calls', expect.objectContaining({
      method: 'POST', body: expect.stringContaining('+79991234567'),
    })))
    expect(await screen.findByText('Набираем номер')).toBeInTheDocument()
  })

  it('sends the new task with Command Enter', async () => {
    render(<App />)
    await userEvent.click(screen.getByRole('button', { name: 'Новый диалог' }))
    const composer = screen.getByLabelText('Сообщение')
    await userEvent.type(composer, 'Позвони Артуру')
    fireEvent.keyDown(composer, { key: 'Enter', metaKey: true })
    expect(await screen.findByText('На какой номер позвонить?')).toBeInTheDocument()
  })

  it('shows call history and reopens a previous conversation', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/calls') && !init?.method) return new Response(JSON.stringify({ calls: [
        { session_id: 'demo_old', contact_name: 'Артур', task: 'Узнать про машину', status: 'ended', created_at: 1700000000 },
      ] }), { status: 200 })
      if (url.includes('/api/calls/demo_old')) return new Response(JSON.stringify({
        session_id: 'demo_old', room_name: 'room_old', status: 'ended', created_at: 1700000000,
        request: { contact_name: 'Артур' },
        events: [
          { session_id: 'demo_old', room_name: 'room_old', seq: 1, timestamp: 1, type: 'transcript.caller.final', payload: { text: 'Алло' } },
          { session_id: 'demo_old', room_name: 'room_old', seq: 2, timestamp: 2, type: 'call.outcome', payload: { outcome: 'Договорились' } },
          { session_id: 'demo_old', room_name: 'room_old', seq: 3, timestamp: 3, type: 'call.ended', payload: {} },
        ],
      }), { status: 200 })
      return new Response('{}', { status: 200 })
    }))
    render(<App />)

    expect(await screen.findByText('Узнать про машину')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Артур/ }))

    expect(await screen.findByText(/Звонок завершён ·\s*Договорились/)).toBeInTheDocument()
    expect(localStorage.getItem('t2.activeSession')).toBe('demo_old')
  })

  it('updates partial transcript in place and appends assistant deltas', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    render(<App />)
    await screen.findByText('Набираем номер')
    MockSocket.instance.emit(event(1, 'transcript.caller.partial', { utterance_id: 'u1', text: 'Я могу' }))
    MockSocket.instance.emit(event(2, 'transcript.caller.partial', { utterance_id: 'u1', text: 'Я могу завтра' }))
    await waitFor(() => expect(screen.getByText((_, node) => node?.textContent === 'Я могу завтра')).toBeInTheDocument())
    expect(screen.queryByText('Я могу')).not.toBeInTheDocument()
    MockSocket.instance.emit(event(3, 'transcript.caller.final', { utterance_id: 'u1', text: 'Я могу завтра утром' }))
    MockSocket.instance.emit(event(4, 'transcript.assistant.delta', { utterance_id: 'a1', text: 'Отлично, ' }))
    MockSocket.instance.emit(event(5, 'transcript.assistant.delta', { utterance_id: 'a1', text: 'договорились.' }))
    await waitFor(() => expect(screen.getByText('Я могу завтра утром')).toBeInTheDocument())
    await waitFor(() => expect(screen.getByText((_, node) => node?.textContent === 'Отлично, договорились.')).toBeInTheDocument())
  })

  it('reconciles the streamed assistant turn with its persisted conversation item', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    render(<App />)
    await screen.findByText('Набираем номер')
    MockSocket.instance.emit(event(4, 'transcript.assistant.delta', { utterance_id: 'stream-1', text: 'Есть стол!!!?' }))
    MockSocket.instance.emit(event(5, 'transcript.assistant.final', { utterance_id: 'stream-1', text: 'Есть стол!!!?' }))
    MockSocket.instance.emit(event(6, 'transcript.assistant.final', { utterance_id: 'item-1', text: 'Есть стол?' }))
    await waitFor(() => expect(screen.getAllByText(/Есть стол/)).toHaveLength(1))
    expect(screen.getByText('Есть стол?')).toBeInTheDocument()
  })

  it('submits owner decisions once and supports custom text', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    render(<App />)
    await screen.findByText('Набираем номер')
    MockSocket.instance.emit(event(5, 'owner.question', { request_id: 'r1', question: 'Подтвердить встречу на 10:00?', context: 'Завтра' }))
    expect(await screen.findByText('Нужно ваше решение')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Другое' }))
    await userEvent.type(screen.getByLabelText('Ваш ответ'), 'Предложить 11:00')
    await userEvent.click(screen.getByRole('button', { name: 'Отправить' }))
    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/calls/demo_1/owner-response', expect.objectContaining({ method: 'POST' })))
    expect(screen.getByText('Предложить 11:00')).toBeInTheDocument()
  })

  it('sends a custom owner answer with Command Enter', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    render(<App />)
    await screen.findByText('Набираем номер')
    MockSocket.instance.emit(event(5, 'owner.question', { request_id: 'r1', question: 'Что ответить?', context: '' }))
    await userEvent.click(await screen.findByRole('button', { name: 'Другое' }))
    const answer = screen.getByLabelText('Ваш ответ')
    await userEvent.type(answer, 'Предложить 11:00')
    fireEvent.keyDown(answer, { key: 'Enter', metaKey: true })
    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/calls/demo_1/owner-response', expect.objectContaining({ body: expect.stringContaining('Предложить 11:00') })))
  })

  it('renders dynamic owner answer options and submits the selected text', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    render(<App />)
    await screen.findByText('Набираем номер')
    MockSocket.instance.emit(event(5, 'owner.question', {
      request_id: 'r-choice', question: 'В каком зале бронировать?', context: '',
      options: ['Обычный зал', 'VIP-зал'],
    }))
    expect(await screen.findByRole('button', { name: 'Обычный зал' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'VIP-зал' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Подтвердить' })).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'VIP-зал' }))
    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/calls/demo_1/owner-response', expect.objectContaining({
      method: 'POST', body: expect.stringContaining('VIP-зал'),
    })))
  })

  it('adds a new owner instruction while the same call is active', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    render(<App />)
    await screen.findByText('Набираем номер')

    await userEvent.type(screen.getByLabelText('Новое поручение'), 'Узнай, стол у окна или нет?')
    await userEvent.click(screen.getByRole('button', { name: 'Отправить поручение' }))

    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/calls/demo_1/instructions', expect.objectContaining({
      method: 'POST', body: expect.stringContaining('Узнай, стол у окна или нет?'),
    })))
  })

  it('sends a live instruction with Command Enter', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    render(<App />)
    await screen.findByText('Набираем номер')
    const instruction = screen.getByLabelText('Новое поручение')
    await userEvent.type(instruction, 'Уточни адрес')
    fireEvent.keyDown(instruction, { key: 'Enter', metaKey: true })
    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/calls/demo_1/instructions', expect.objectContaining({ body: expect.stringContaining('Уточни адрес') })))
  })

  it('deduplicates optimistic and persisted owner answers by request id', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    render(<App />)
    await screen.findByText('Набираем номер')
    MockSocket.instance.emit(event(5, 'owner.question', { request_id: 'r1', question: 'Подтвердить?', context: '' }))
    await userEvent.click(await screen.findByRole('button', { name: 'Подтвердить' }))
    MockSocket.instance.emit(event(6, 'owner.answer', { request_id: 'r1', response: 'Да' }))
    MockSocket.instance.emit(event(7, 'owner.answer', { request_id: 'r1', answer: 'Да', action: 'instruct' }))
    await waitFor(() => expect(screen.getAllByText('Да')).toHaveLength(1))
  })

  it('keeps the transcript visible when opening a completed call from history', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/calls') && !init?.method) return new Response(JSON.stringify({ calls: [
        { session_id: 'demo_old', contact_name: 'Артур', task: 'Узнать про машину', status: 'ended', created_at: 1700000000 },
      ] }), { status: 200 })
      if (url.includes('/api/calls/demo_old')) return new Response(JSON.stringify({
        session_id: 'demo_old', room_name: 'room_old', status: 'ended', created_at: 1700000000,
        events: [
          event(1, 'transcript.caller.final', { utterance_id: 'u1', text: 'Алло' }),
          event(2, 'transcript.assistant.final', { utterance_id: 'a1', text: 'Добрый вечер' }),
          event(3, 'call.outcome', { outcome: 'Agreed', summary: 'Столик забронирован', next_step: 'Прийти к 20:00' }),
          event(4, 'call.ended', {}),
        ],
      }), { status: 200 })
      return new Response('{}', { status: 200 })
    }))
    render(<App />)
    await userEvent.click(await screen.findByRole('button', { name: /Артур/ }))
    expect(await screen.findByText('Алло')).toBeInTheDocument()
    expect(screen.getByText('Добрый вечер')).toBeInTheDocument()
    expect(screen.getByText('Столик забронирован')).toBeInTheDocument()
  })

  it('renders final outcome and recording', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    render(<App />)
    await screen.findByText('Набираем номер')
    MockSocket.instance.emit(event(8, 'call.outcome', { outcome: 'agreed', summary: 'Анна подтвердила завтра в 11:00', next_step: 'Добавить встречу в календарь' }))
    MockSocket.instance.emit(event(9, 'recording.ready', { url: '/api/calls/demo_1/recording' }))
    MockSocket.instance.emit(event(10, 'call.ended', {}))
    expect(await screen.findByText(/Звонок завершён ·\s*agreed/)).toBeInTheDocument()
    expect(screen.getByText('Добавить встречу в календарь')).toBeInTheDocument()
    expect(document.querySelector('audio')).toHaveAttribute('src', '/api/calls/demo_1/recording')
    expect(screen.queryByText('Новая задача')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Перезвонить' })).toHaveClass('primary')
    expect(screen.getByRole('button', { name: 'Отменить договорённость' })).toHaveClass('primary', 'black-action')
    expect(screen.getByRole('button', { name: 'Отменить договорённость' })).toHaveClass('black-action')
  })

  it('does not offer cancellation when the call ended without an agreement', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    render(<App />)
    await screen.findByText('Набираем номер')
    MockSocket.instance.emit(event(8, 'call.outcome', { outcome: 'incomplete', summary: 'Договорённость не достигнута' }))
    MockSocket.instance.emit(event(9, 'call.ended', {}))

    expect(await screen.findByText(/Звонок завершён ·\s*incomplete/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Перезвонить' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Отменить договорённость' })).not.toBeInTheDocument()
  })

  it.each([
    ['Отменить договорённость', 'cancel'],
  ])('starts a new contextual call from the completed result with %s', async (label, action) => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/calls/demo_1/follow-up')) return new Response(JSON.stringify({ session_id: 'demo_follow', room_name: 'room_follow' }), { status: 201 })
      if (url.includes('/api/calls/demo_follow')) return new Response(JSON.stringify({ session_id: 'demo_follow', room_name: 'room_follow', status: 'dialing', events: [] }), { status: 200 })
      return new Response(JSON.stringify({ ...createdCall, status: 'ended', request: { contact_name: 'Ресторан' }, events: [
        event(7, 'transcript.assistant.final', { utterance_id: 'a7', text: 'До свидания' }),
        event(8, 'call.outcome', { outcome: 'agreed', summary: 'Стол забронирован' }),
        event(8.5, 'recording.ready', { url: '/api/calls/demo_1/recording' }),
        event(9, 'call.ended', {}),
      ] }), { status: 200 })
    }))
    render(<App />)

    await userEvent.click(await screen.findByRole('button', { name: label }))

    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/calls/demo_1/follow-up', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ action }),
    })))
    expect(localStorage.getItem('t2.activeSession')).toBe('demo_follow')
    expect(await screen.findByText('Набираем номер')).toBeInTheDocument()
    expect(screen.getByText('Стол забронирован')).toBeInTheDocument()
    const previousResult = screen.getByText('Стол забронирован')
    const previousRecording = document.querySelector('audio[src="/api/calls/demo_1/recording"]')
    const continuation = screen.getByText(action === 'continue' ? 'Перезваниваем и продолжаем разговор' : 'Перезваниваем, чтобы отменить договорённость')
    expect(previousRecording).toBeInTheDocument()
    expect(previousResult.compareDocumentPosition(continuation) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    MockSocket.instance.emit({ ...event(1, 'transcript.caller.final', { utterance_id: 'follow-u1', text: 'Снова здравствуйте' }), session_id: 'demo_follow' })
    const followUpMessage = await screen.findByText('Снова здравствуйте')
    expect(continuation.compareDocumentPosition(followUpMessage) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(previousResult.compareDocumentPosition(followUpMessage) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('asks what to clarify before calling back and sends it with the follow-up', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/calls/demo_1/follow-up')) return new Response(JSON.stringify({ session_id: 'demo_follow', room_name: 'room_follow' }), { status: 201 })
      if (url.includes('/api/calls/demo_follow')) return new Response(JSON.stringify({ session_id: 'demo_follow', room_name: 'room_follow', status: 'dialing', events: [] }), { status: 200 })
      return new Response(JSON.stringify({ ...createdCall, status: 'ended', request: { contact_name: 'Ресторан' }, events: [
        event(8, 'call.outcome', { outcome: 'incomplete', summary: 'Не уточнили наличие столика у окна' }),
        event(9, 'call.ended', {}),
      ] }), { status: 200 })
    }))
    render(<App />)

    await userEvent.click(await screen.findByRole('button', { name: 'Перезвонить' }))
    const question = screen.getByLabelText('Что спросить?')
    const callbackButton = screen.getByRole('button', { name: 'Позвонить с уточнением' })
    expect(callbackButton).toHaveClass('primary')
    expect(fetch).not.toHaveBeenCalledWith('/api/calls/demo_1/follow-up', expect.anything())
    await userEvent.type(question, 'Уточнить, есть ли столик у окна')
    fireEvent.keyDown(question, { key: 'Enter', metaKey: true })

    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/calls/demo_1/follow-up', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ action: 'continue', instruction: 'Уточнить, есть ли столик у окна' }),
    })))
    expect(localStorage.getItem('t2.activeSession')).toBe('demo_follow')
  })

  it('ignores duplicate and unsafe event types', async () => {
    localStorage.setItem('t2.activeSession', 'demo_1')
    render(<App />)
    await screen.findByText('Набираем номер')
    MockSocket.instance.emit(event(2, 'transcript.caller.final', { utterance_id: 'u1', text: 'Здравствуйте' }))
    MockSocket.instance.emit(event(2, 'transcript.caller.final', { utterance_id: 'u2', text: 'ДУБЛЬ' }))
    MockSocket.instance.emit(event(3, 'internal.prompt', { text: 'СЕКРЕТ' }))
    await waitFor(() => expect(screen.getAllByText('Здравствуйте')).toHaveLength(1))
    expect(screen.queryByText('ДУБЛЬ')).not.toBeInTheDocument()
    expect(screen.queryByText('СЕКРЕТ')).not.toBeInTheDocument()
  })
})
