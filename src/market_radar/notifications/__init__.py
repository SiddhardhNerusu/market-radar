"""macOS notifications when high-score signals fire."""
from .macos import send_notification
from .notifier import NotificationStats, dispatch_pending
from .realtime import Notifier, AlertCandidate, SendResult

__all__ = [
    "send_notification", "dispatch_pending", "NotificationStats",
    "Notifier", "AlertCandidate", "SendResult",
]
