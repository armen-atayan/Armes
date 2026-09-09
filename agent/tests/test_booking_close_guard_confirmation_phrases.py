import pytest

from booking_close_guard import booking_close_reason


@pytest.mark.parametrize(
    "text",
    [
        "Готово.",
        "Оформлено.",
        "Запись готова.",
        "Запись оформлена.",
        "Ваша запись создана.",
    ],
)
def test_clear_booking_confirmation_phrases_are_accepted(text):
    assert booking_close_reason(text, "agreed") is None


def test_guard_error_is_organization_neutral():
    reason = booking_close_reason("Сообщим позже", "agreed")
    assert "организации" in reason
    assert "ресторана" not in reason