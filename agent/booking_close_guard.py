"""Conservative Russian booking-closure checks; not a general intent classifier.

Unknown/garbled speech is not acceptance. Applied only to explicitly scoped
personal-assistant booking calls, never to other personas or tasks.
"""
import re

_QUESTION = re.compile(r'\?|\b(?:сколько|какое|какой|какая|какие|когда|куда|откуда|кто|зачем|почему|подскажите|уточните|повторите|подождите)\b', re.I)
_NEGATIVE = re.compile(r'\b(?:не|нет|нельзя|отмен[а-яё]*)\b', re.I)
_SHORT_YES = re.compile(r'(?:(?:да|хорошо|конечно|можем|согласен|согласна|согласны|давайте)[\s,!.…]*)+', re.I)
_BOOKED = re.compile(r'\b(?:подтверждаю|подтверждаем|подтвержден[аоы]?|записал[аи]?|записываю|записаны|записан|записана|забронировал[аи]?|забронирован[аоы]?|бронируем|оформил[аи]?)\b', re.I)
_NATURAL_YES = re.compile(
    r'^(?:в\s+принципе\s+)?да(?:[\s,]+(?:да|могу|можем|получится|все\s+нормально|нормально|спасибо))*[\s.!…]*$',
    re.I,
)
_HANGUP_REQUEST = re.compile(
    r'(?:\bдо\s+свидания\b|\b(?:положи(?:те)?|клади(?:те)?|повесь(?:те)?)\b.{0,24}\bтрубк|'
    r'\bтрубк\w*\b.{0,24}\b(?:положи(?:те)?|клади(?:те)?|повесь(?:те)?)\b)',
    re.I,
)


def callee_requested_hangup(text: str) -> bool:
    return bool(_HANGUP_REQUEST.search(text.strip().replace('ё', 'е')))


def booking_close_reason(text: str, outcome: str) -> str | None:
    text = text.strip().replace('ё', 'е')
    if _QUESTION.search(text):
        return ('Не завершай звонок: собеседник задал вопрос. Ответь на него по поручению; '
                'вопрос о количестве гостей не является подтверждением брони.')
    if outcome == 'agreed' and (not text or _NEGATIVE.search(text) or
            not (_SHORT_YES.fullmatch(text) or _NATURAL_YES.fullmatch(text) or _BOOKED.search(text))):
        return ('Не завершай звонок: в последней реплике нет явного согласия ресторана на бронь. '
                'Не считай неразборчивый текст согласием. Ответь на открытый вопрос; '
                'при необходимости попроси повторить. Согласие Армена в Telegram не заменяет ответ ресторана.')
    return None
