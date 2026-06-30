import threading

_cancellation_registry: dict[str, threading.Event] = {}

def register_request(response_id: str) -> threading.Event:
    event = threading.Event()
    _cancellation_registry[response_id] = event
    return event

def cancel_request(response_id: str) -> bool:
    if response_id in _cancellation_registry:
        _cancellation_registry[response_id].set()
        return True
    return False

def is_cancelled(response_id: str) -> bool:
    if response_id in _cancellation_registry:
        return _cancellation_registry[response_id].is_set()
    return False

def cleanup_request(response_id: str) -> None:
    _cancellation_registry.pop(response_id, None)
