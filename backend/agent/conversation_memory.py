from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    user_intent: str
    task: str
    active_file: str | None
    files_referenced: list[str] = field(default_factory=list)

class ConversationMemoryStore:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.sessions: dict[str, list[ConversationTurn]] = {}

    def add_turn(self, project_path: str, turn: ConversationTurn) -> None:
        if not project_path:
            return
        if project_path not in self.sessions:
            self.sessions[project_path] = []
        self.sessions[project_path].append(turn)
        if len(self.sessions[project_path]) > self.max_turns:
            self.sessions[project_path].pop(0)

    def get_context(self, project_path: str) -> list[ConversationTurn]:
        if not project_path:
            return []
        return self.sessions.get(project_path, [])

    def resolve_references(self, project_path: str, instruction: str) -> list[str]:
        """Resolve conversational references like 'the previous file' or 'both'."""
        history = self.get_context(project_path)
        if not history:
            return []
            
        lower_inst = instruction.lower()
        resolved_files = []
        
        # Flatten recently referenced files (newest first)
        recent_files = []
        for turn in reversed(history):
            for f in turn.files_referenced:
                if f not in recent_files:
                    recent_files.append(f)
            if turn.active_file and turn.active_file not in recent_files:
                recent_files.append(turn.active_file)
                
        if not recent_files:
            return []

        if "previous file" in lower_inst or "that file" in lower_inst or "the file" in lower_inst:
            resolved_files.append(recent_files[0])
            
        if "the first file" in lower_inst and len(recent_files) >= 2:
            resolved_files.append(recent_files[1]) # oldest of the last two
            
        if "the second file" in lower_inst and len(recent_files) >= 1:
            resolved_files.append(recent_files[0])

        if "both" in lower_inst and len(recent_files) >= 2:
            resolved_files.extend(recent_files[:2])
            
        return list(dict.fromkeys(resolved_files))

# Global memory singleton for the current server process
memory_store = ConversationMemoryStore()
