export function extractPhone(task: string) {
  const plus = task.indexOf('+')
  if (plus >= 0) {
    let digits = ''
    for (const char of task.slice(plus + 1)) {
      if (char >= '0' && char <= '9') digits += char
      else if (' ()-'.includes(char)) continue
      else break
    }
    if (digits.length >= 8 && digits.length <= 15 && digits[0] !== '0') return `+${digits}`
  }
  const candidates = task.match(/[\d][\d\s()-]{7,21}/g) ?? []
  for (const candidate of candidates) {
    const digits = candidate.replace(/\D/g, '')
    if (/^[78]\d{10}$/.test(digits)) return `+7${digits.slice(1)}`
  }
  return ''
}
