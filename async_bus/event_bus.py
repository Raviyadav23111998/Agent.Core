import asyncio
import traceback
from typing import Callable, Dict, List


class EventBus:
    """Asynchronous event bus for reactive agent wakeups.
    
    Design (matching Agent):
    - Subscribers are isolated: one failing handler never crashes others
    - Events are logged with type and payload size for debugging
    - Supports unsubscribe for clean shutdown
    """
    
    _MAX_EVENT_LOG = 200

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_log: List[dict] = []
        
    def subscribe(self, event_type: str, callback: Callable):
        """Register a callback for an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        
    def unsubscribe(self, event_type: str, callback: Callable):
        """Remove a specific callback."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                cb for cb in self._subscribers[event_type] if cb != callback
            ]
        
    async def publish(self, event_type: str, payload: dict):
        """Publish an event to all subscribers with error isolation.
        
        Each subscriber runs in its own try/except — a failing email handler
        won't prevent the agent loop from receiving its event.
        """
        self._event_log.append({
            "type": event_type,
            "payload_keys": list(payload.keys()),
        })
        if len(self._event_log) > self._MAX_EVENT_LOG:
            del self._event_log[:-self._MAX_EVENT_LOG]
        
        if event_type not in self._subscribers:
            return
            
        results = []
        for callback in self._subscribers[event_type]:
            try:
                result = callback(payload)
                if asyncio.iscoroutine(result):
                    result = await result
                results.append(result)
            except Exception as e:
                print(f"[EventBus] Handler error on '{event_type}': {e}")
                traceback.print_exc()
        
        return results

    def get_event_log(self, last_n: int = 20) -> list:
        """Return recent events for debugging."""
        return self._event_log[-last_n:]
