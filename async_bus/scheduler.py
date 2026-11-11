import asyncio
import uuid
from typing import Dict
from .event_bus import EventBus


class AgentScheduler:
    """Timer and scheduling system for async agent wakeups.
    
    Design (matching Agent):
    - Max 10 concurrent timers to prevent resource exhaustion
    - Graceful cancel_all for clean shutdown
    - Timer IDs are short UUIDs for human readability
    """
    
    MAX_CONCURRENT_TIMERS = 10

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.active_timers: Dict[str, asyncio.Task] = {}
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def bind_loop(self):
        """Capture the running event loop (call once from async context).

        set_timer is a tool callback and runs in a worker thread via
        asyncio.to_thread — there is no running loop THERE, so timers must be
        scheduled onto the main loop captured here.
        """
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    async def _timer_worker(self, timer_id: str, duration_seconds: int, prompt: str):
        """Sleep in background, then publish wakeup event."""
        try:
            await asyncio.sleep(duration_seconds)
            if timer_id in self.active_timers:
                await self.event_bus.publish('timer_wakeup', {
                    'timer_id': timer_id,
                    'prompt': prompt
                })
        except asyncio.CancelledError:
            pass
        finally:
            self.active_timers.pop(timer_id, None)

    def set_timer(self, duration_seconds: int, prompt: str) -> str:
        """Set a one-shot timer. Returns timer ID."""
        if len(self.active_timers) >= self.MAX_CONCURRENT_TIMERS:
            return f"Error: Maximum {self.MAX_CONCURRENT_TIMERS} concurrent timers reached. Cancel one first."
        
        timer_id = str(uuid.uuid4())[:8]
        coro = self._timer_worker(timer_id, duration_seconds, prompt)
        try:
            loop = asyncio.get_running_loop()
            self._loop = loop
            task = loop.create_task(coro)
        except RuntimeError:
            loop = self._loop
            if loop is None or not loop.is_running():
                coro.close()
                return ("Error: no running event loop available to schedule "
                        "the timer.")
            task = asyncio.run_coroutine_threadsafe(coro, loop)
        self.active_timers[timer_id] = task

        return f"Timer '{timer_id}' set for {duration_seconds}s. Wakeup prompt: '{prompt}'"
        
    def cancel_timer(self, timer_id: str) -> str:
        """Cancel a specific timer."""
        if timer_id in self.active_timers:
            self.active_timers[timer_id].cancel()
            del self.active_timers[timer_id]
            return f"Timer '{timer_id}' cancelled."
        return f"Timer '{timer_id}' not found."
    
    def cancel_all(self) -> str:
        """Cancel all active timers — used during shutdown."""
        count = len(self.active_timers)
        for timer_id, task in list(self.active_timers.items()):
            task.cancel()
        self.active_timers.clear()
        return f"Cancelled {count} timers."
    
    def list_timers(self) -> str:
        """List all active timers."""
        if not self.active_timers:
            return "No active timers."
        lines = [f"- {tid}" for tid in self.active_timers]
        return "\n".join(lines)
