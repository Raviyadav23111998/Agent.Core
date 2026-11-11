import os
import asyncio
import uuid
from typing import Dict, Optional
from async_bus.event_bus import EventBus


class BackgroundTask:
    """Represents a single background process with monitoring."""
    
    def __init__(self, task_id: str, command: str, cwd: str, process: asyncio.subprocess.Process):
        self.task_id = task_id
        self.command = command
        self.cwd = cwd
        self.process = process
        self.stdout_buffer: list = []
        self.stderr_buffer: list = []
        self.return_code: Optional[int] = None
        self.completed = False


class BackgroundTaskManager:
    """Manages background processes with async monitoring and event-driven completion.
    
    Architecture (identical to Agent):
    - Long-running commands spawn as asyncio subprocesses
    - stdout/stderr are captured asynchronously
    - EventBus publishes 'task_completed' when process exits
    - Agent can check status, send input, or kill tasks by ID
    - Max 5 concurrent background tasks
    """
    
    MAX_CONCURRENT_TASKS = 5
    MAX_OUTPUT_BYTES = 50_000
    
    def __init__(self, event_bus: EventBus, workspace_root: str):
        self.event_bus = event_bus
        self.workspace_root = os.path.abspath(workspace_root)
        self.tasks: Dict[str, BackgroundTask] = {}
    
    async def spawn(self, command: str, cwd: str) -> str:
        """Spawn a background process and return its task ID."""
        if len(self.tasks) >= self.MAX_CONCURRENT_TASKS:
            return f"Error: Max {self.MAX_CONCURRENT_TASKS} concurrent tasks. Kill one first."
        
        abs_cwd = os.path.abspath(cwd)
        if not os.path.isdir(abs_cwd):
            return f"Error: Directory not found: {abs_cwd}"
        
        task_id = str(uuid.uuid4())[:8]
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=abs_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
                env={**os.environ, "PAGER": "cat", "GIT_PAGER": "cat"},
            )
        except Exception as e:
            return f"Error spawning process: {e}"
        
        task = BackgroundTask(task_id, command, abs_cwd, process)
        self.tasks[task_id] = task
        
        asyncio.create_task(self._monitor_task(task))
        
        return f"Background task '{task_id}' started: {command}"
    
    async def _monitor_task(self, task: BackgroundTask):
        """Monitor a background task's output and publish completion event."""
        try:
            stdout, stderr = await task.process.communicate()
            
            task.return_code = task.process.returncode
            task.completed = True
            
            if stdout:
                output = stdout.decode('utf-8', errors='replace')
                if len(output) > self.MAX_OUTPUT_BYTES:
                    output = output[:self.MAX_OUTPUT_BYTES] + f"\n...(truncated at {self.MAX_OUTPUT_BYTES} bytes)"
                task.stdout_buffer = [output]
            
            if stderr:
                err_output = stderr.decode('utf-8', errors='replace')
                if len(err_output) > self.MAX_OUTPUT_BYTES:
                    err_output = err_output[:self.MAX_OUTPUT_BYTES] + "\n...(truncated)"
                task.stderr_buffer = [err_output]
            
            await self.event_bus.publish('task_completed', {
                'task_id': task.task_id,
                'command': task.command,
                'return_code': task.return_code,
                'stdout': ''.join(task.stdout_buffer),
                'stderr': ''.join(task.stderr_buffer),
            })
            
        except asyncio.CancelledError:
            task.completed = True
            task.return_code = -1
    
    def get_status(self, task_id: str) -> str:
        """Get status of a background task."""
        if task_id not in self.tasks:
            return f"Task '{task_id}' not found."
        
        task = self.tasks[task_id]
        if task.completed:
            stdout = ''.join(task.stdout_buffer)
            stderr = ''.join(task.stderr_buffer)
            result = f"Task '{task_id}' completed (exit code: {task.return_code})\n"
            if stdout:
                result += f"STDOUT:\n{stdout}\n"
            if stderr:
                result += f"STDERR:\n{stderr}"
            return result
        else:
            return f"Task '{task_id}' is still running: {task.command}"
    
    def kill_task(self, task_id: str) -> str:
        """Kill a background task."""
        if task_id not in self.tasks:
            return f"Task '{task_id}' not found."
        
        task = self.tasks[task_id]
        if task.completed:
            return f"Task '{task_id}' already completed."
        
        try:
            task.process.terminate()
            return f"Task '{task_id}' terminated."
        except ProcessLookupError:
            return f"Task '{task_id}' process already exited."
    
    async def send_input(self, task_id: str, input_text: str) -> str:
        """Send stdin input to a running task."""
        if task_id not in self.tasks:
            return f"Task '{task_id}' not found."
        
        task = self.tasks[task_id]
        if task.completed:
            return f"Task '{task_id}' already completed."
        
        if task.process.stdin:
            task.process.stdin.write((input_text + "\n").encode())
            await task.process.stdin.drain()
            return f"Input sent to task '{task_id}'."
        
        return f"Task '{task_id}' has no stdin."
    
    def list_tasks(self) -> str:
        """List all background tasks."""
        if not self.tasks:
            return "No background tasks."
        
        lines = []
        for tid, task in self.tasks.items():
            status = "completed" if task.completed else "running"
            lines.append(f"  {tid}: [{status}] {task.command}")
        return "\n".join(lines)
    
    def cleanup_completed(self):
        """Remove completed tasks from tracking."""
        completed = [tid for tid, t in self.tasks.items() if t.completed]
        for tid in completed:
            del self.tasks[tid]
