import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """
    AgentConfig
    """