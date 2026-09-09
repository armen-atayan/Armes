export type CallEvent = {
  session_id: string
  room_name?: string
  seq: number
  timestamp: number
  type: string
  payload: Record<string, unknown>
}

export type Message = { id: string; role: 'caller' | 'assistant' | 'owner'; text: string; partial?: boolean; interrupted?: boolean }
export type CallOutcome = { outcome: string; summary?: string; nextStep?: string }
export type OwnerQuestion = { requestId: string; question: string; context?: string; options: string[] }
export type TaskInput = { contact_name: string; phone_number: string; task: string; details?: string }
export type CallHistoryItem = { session_id: string; contact_name: string; phone_number: string; task: string; status: string; created_at: number }
