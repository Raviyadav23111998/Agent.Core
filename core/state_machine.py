from enum import Enum
from typing import Dict, Set

class AgentState(Enum):
    """Discrete states of the agent's execution loop.
    
    Mirrors Agent's internal state machine:
    - Strict transition graph prevents invalid state jumps
    - Each state maps to exactly one phase of the Observe-Reason-Act cycle
    """
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    PLANNING = "PLANNING"
    REASONING = "REASONING"
    VALIDATING = "VALIDATING"
    EXECUTING = "EXECUTING"
    PRUNING = "PRUNING"
    ERROR_RECOVERY = "ERROR_RECOVERY"
    RESPONDING = "RESPONDING"


_VALID_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
    AgentState.IDLE:           {AgentState.OBSERVING},
    AgentState.OBSERVING:      {AgentState.PLANNING, AgentState.REASONING},
    AgentState.PLANNING:       {AgentState.REASONING, AgentState.PRUNING},
    AgentState.REASONING:      {AgentState.VALIDATING},
    AgentState.VALIDATING:     {AgentState.EXECUTING, AgentState.RESPONDING, AgentState.ERROR_RECOVERY, AgentState.REASONING},
    AgentState.EXECUTING:      {AgentState.REASONING, AgentState.PRUNING, AgentState.ERROR_RECOVERY, AgentState.RESPONDING},
    AgentState.PRUNING:        {AgentState.REASONING, AgentState.RESPONDING},
    AgentState.ERROR_RECOVERY: {AgentState.REASONING},
    AgentState.RESPONDING:     {AgentState.IDLE},
}


class StateMachine:
    """Manages transitions and current state of the Agent Loop.
    
    Validates every transition against the allowed graph.
    Invalid transitions raise ValueError immediately — fail loud, fail fast.
    """
    
    def __init__(self):
        self.current_state = AgentState.IDLE
        self._transition_count = 0
        
    def transition_to(self, new_state: AgentState):
        """Transition to a new state with validation."""
        allowed = _VALID_TRANSITIONS.get(self.current_state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid state transition: {self.current_state.value} → {new_state.value}. "
                f"Allowed: {{{', '.join(s.value for s in allowed)}}}"
            )
        self.current_state = new_state
        self._transition_count += 1
        
    def is_in_state(self, state: AgentState) -> bool:
        return self.current_state == state
    
    def reset(self):
        """Reset to IDLE after a complete turn."""
        self.current_state = AgentState.IDLE
        self._transition_count = 0
