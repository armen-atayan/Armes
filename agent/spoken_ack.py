"""Low-latency, lexical normalization of Russian spoken acknowledgements."""

from collections.abc import AsyncIterable, AsyncIterator
from unicodedata import category

_TARGETS = ("понял", "понятно")
_BOUNDARIES = ".!?…\n\r\u2028\u2029"
_LEADING = "\"'«»„“”‘’()[]{}—–-"

_PERSONAL_ASSISTANT_REPLACEMENTS = (
    ("смотрите... а ", "Подскажите, пожалуйста, "),
    ("смотрите… а ", "Подскажите, пожалуйста, "),
    ("смотрите, а ", "Подскажите, пожалуйста, "),
    ("смотрите, ", "Подскажите, пожалуйста, "),
    ("смотрите ", "Подскажите, пожалуйста, "),
    ("это возможно сделать", "Получится сделать"),
    ("это возможно", "Получится"),
    ("это допустимо", "Можно"),
    ("я поняла", "я понял"),
    ("я уточнила", "я уточнил"),
    ("я посмотрела", "я посмотрел"),
    ("я проверила", "я проверил"),
    ("я записала", "я записал"),
    ("я узнала", "я узнал"),
    ("я сказала", "я сказал"),
    ("я готова", "я готов"),
    ("я рада", "я рад"),
    ("поняла", "понял"),
)


async def normalize_personal_assistant_text(text: AsyncIterable[str]) -> AsyncIterator[str]:
    """Apply bounded, chunk-independent wording guards for Armen's male assistant."""
    pending = ""
    sentence_prefix = ""
    replacements = tuple((source.lower(), target) for source, target in _PERSONAL_ASSISTANT_REPLACEMENTS)

    def rendered(source: str, target: str) -> str:
        if source and source[0].isupper():
            return target[0].upper() + target[1:]
        return target

    async for chunk in normalize_acknowledgements(text):
        for char in chunk:
            pending += char
            while pending:
                lowered = pending.lower()
                prefixes = [(source, target) for source, target in replacements if source.startswith(lowered)]
                if prefixes:
                    break
                exact = [(source, target) for source, target in replacements if lowered.startswith(source)]
                if exact:
                    source, target = max(exact, key=lambda item: len(item[0]))
                    original = pending[:len(source)]
                    transformed = rendered(original, target)
                    sentence_prefix = (sentence_prefix + transformed)[-16:]
                    yield transformed
                    pending = pending[len(source):]
                    continue
                outgoing = pending[0]
                if outgoing == "?" and sentence_prefix.lstrip(" \t\n\r\"'«»„“”‘’()[]{}—–-").lower().startswith("можно "):
                    outgoing = "."
                yield outgoing
                if outgoing in ".!?…\n\r":
                    sentence_prefix = ""
                elif len(sentence_prefix) < 16:
                    sentence_prefix += outgoing
                pending = pending[1:]

    while pending:
        lowered = pending.lower()
        exact = [(source, target) for source, target in replacements if lowered.startswith(source)]
        if exact:
            source, target = max(exact, key=lambda item: len(item[0]))
            original = pending[:len(source)]
            transformed = rendered(original, target)
            sentence_prefix = (sentence_prefix + transformed)[-16:]
            yield transformed
            pending = pending[len(source):]
        else:
            outgoing = pending[0]
            if outgoing == "?" and sentence_prefix.lstrip(" \t\n\r\"'«»„“”‘’()[]{}—–-").lower().startswith("можно "):
                outgoing = "."
            yield outgoing
            if outgoing in ".!?…\n\r":
                sentence_prefix = ""
            elif len(sentence_prefix) < 16:
                sentence_prefix += outgoing
            pending = pending[1:]


async def normalize_acknowledgements(text: AsyncIterable[str]) -> AsyncIterator[str]:
    """Replace initial standalone Понял/Понятно, preserving text around them.

    Initial means start of stream or after sentence punctuation/newlines,
    allowing intervening whitespace, quotation marks and dialogue dashes.
    Capitalization follows the target's first letter (including all-caps input).
    An immediately following '?' preserves the original question.

    Across chunks only a possible target prefix is retained (at most seven
    characters); the next character decides whether to replace it. Whitespace
    and unrelated text stream immediately, without waiting for a sentence or
    the end of the turn. Output chunk boundaries need not match the input.
    This is a lexical filter, not a sentence parser: it deliberately does not
    look beyond the first delimiter for a later question mark.
    """
    pending = ""
    initial = True
    async for chunk in text:
        output = []
        for char in chunk:
            if initial and (pending or char.lower() == "п"):
                candidate = pending + char
                if any(word.startswith(candidate.lower()) for word in _TARGETS):
                    pending = candidate
                    continue
                word_continues = (
                    char.isalnum() or char in "_-‐‑" or category(char).startswith("M")
                )
                if pending.lower() in _TARGETS and not word_continues and char != "?":
                    output.append("Хорошо" if pending[0].isupper() else "хорошо")
                else:
                    output.append(pending)
                pending = ""
                initial = False
            output.append(char)
            if char in _BOUNDARIES:
                initial = True
            elif not (char.isspace() or char in _LEADING):
                initial = False
        # Coalesce only the available input chunk, never await more text to flush.
        if output:
            yield "".join(output)
    if pending:
        if pending.lower() in _TARGETS:
            yield "Хорошо" if pending[0].isupper() else "хорошо"
        else:
            yield pending
