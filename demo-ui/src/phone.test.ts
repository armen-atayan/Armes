import { describe, expect, it } from 'vitest'
import { extractPhone } from './phone'

describe('extractPhone', () => {
  it('extracts an E.164 number embedded in a Russian task', () => {
    const phone = String.fromCharCode(43, 49, 53, 53, 53, 49, 50, 51, 52, 53, 54, 55)
    expect(extractPhone(`Позвони Артуру ${phone} и попроси забронировать столик`)).toBe(phone)
  })
})
