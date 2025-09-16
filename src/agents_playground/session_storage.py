"""
Session storage utility for persisting chat sessions to disk.
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from agents_playground.logger import agent_logger


class SessionStorage:
    """Handle persistent storage of chat sessions."""
    
    def __init__(self, storage_dir: str = None):
        """Initialize session storage with specified directory."""
        if storage_dir is None:
            # Use a .sessions directory in the project root
            project_root = Path(__file__).parent.parent.parent
            storage_dir = project_root / ".sessions"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.sessions_file = self.storage_dir / "chat_sessions.json"
        
    def _serialize_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize session data for JSON storage."""
        serialized = session.copy()
        
        # Convert datetime objects to ISO strings
        if 'created_at' in serialized and isinstance(serialized['created_at'], datetime):
            serialized['created_at'] = serialized['created_at'].isoformat()
        if 'last_updated' in serialized and isinstance(serialized['last_updated'], datetime):
            serialized['last_updated'] = serialized['last_updated'].isoformat()
            
        return serialized
    
    def _deserialize_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize session data from JSON storage."""
        deserialized = session_data.copy()
        
        # Convert ISO strings back to datetime objects
        if 'created_at' in deserialized and isinstance(deserialized['created_at'], str):
            try:
                deserialized['created_at'] = datetime.fromisoformat(deserialized['created_at'])
            except ValueError:
                deserialized['created_at'] = datetime.now()
                
        if 'last_updated' in deserialized and isinstance(deserialized['last_updated'], str):
            try:
                deserialized['last_updated'] = datetime.fromisoformat(deserialized['last_updated'])
            except ValueError:
                deserialized['last_updated'] = datetime.now()
                
        return deserialized
    
    def save_sessions(self, sessions: List[Dict[str, Any]]) -> bool:
        """Save all sessions to storage."""
        try:
            serialized_sessions = [self._serialize_session(session) for session in sessions]
            
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(serialized_sessions, f, indent=2, ensure_ascii=False)
            
            agent_logger.info(f"Saved {len(sessions)} sessions to storage")
            return True
            
        except Exception as e:
            agent_logger.log_error(e, {"context": "session_storage_save"})
            return False
    
    def load_sessions(self) -> List[Dict[str, Any]]:
        """Load all sessions from storage."""
        try:
            if not self.sessions_file.exists():
                agent_logger.info("No existing sessions file found, starting fresh")
                return []
            
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                sessions_data = json.load(f)
            
            sessions = [self._deserialize_session(session) for session in sessions_data]
            agent_logger.info(f"Loaded {len(sessions)} sessions from storage")
            return sessions
            
        except json.JSONDecodeError as e:
            agent_logger.log_error(e, {"context": "session_storage_load_json_decode"})
            # Backup corrupted file and start fresh
            self._backup_corrupted_file()
            return []
            
        except Exception as e:
            agent_logger.log_error(e, {"context": "session_storage_load"})
            return []
    
    def _backup_corrupted_file(self):
        """Backup corrupted sessions file."""
        try:
            if self.sessions_file.exists():
                backup_file = self.storage_dir / f"chat_sessions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.sessions_file.rename(backup_file)
                agent_logger.info(f"Backed up corrupted sessions file to {backup_file}")
        except Exception as e:
            agent_logger.log_error(e, {"context": "session_storage_backup"})
    
    def save_single_session(self, session: Dict[str, Any]) -> bool:
        """Save or update a single session."""
        try:
            sessions = self.load_sessions()
            
            # Find and update existing session or add new one
            session_updated = False
            for i, existing_session in enumerate(sessions):
                if existing_session.get('id') == session.get('id'):
                    sessions[i] = session
                    session_updated = True
                    break
            
            if not session_updated:
                sessions.append(session)
            
            return self.save_sessions(sessions)
            
        except Exception as e:
            agent_logger.log_error(e, {"context": "session_storage_save_single"})
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session."""
        try:
            sessions = self.load_sessions()
            original_count = len(sessions)
            
            sessions = [session for session in sessions if session.get('id') != session_id]
            
            if len(sessions) < original_count:
                self.save_sessions(sessions)
                agent_logger.info(f"Deleted session {session_id}")
                return True
            else:
                agent_logger.info(f"Session {session_id} not found for deletion")
                return False
                
        except Exception as e:
            agent_logger.log_error(e, {"context": "session_storage_delete"})
            return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage."""
        try:
            sessions = self.load_sessions()
            file_size = self.sessions_file.stat().st_size if self.sessions_file.exists() else 0
            
            return {
                "storage_dir": str(self.storage_dir),
                "sessions_file": str(self.sessions_file),
                "session_count": len(sessions),
                "file_size_bytes": file_size,
                "last_modified": datetime.fromtimestamp(
                    self.sessions_file.stat().st_mtime
                ).isoformat() if self.sessions_file.exists() else None
            }
            
        except Exception as e:
            agent_logger.log_error(e, {"context": "session_storage_info"})
            return {"error": str(e)}


# Global session storage instance
session_storage = SessionStorage()