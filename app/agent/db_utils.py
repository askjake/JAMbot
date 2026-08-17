from langgraph.checkpoint.base import BaseCheckpointSaver

checkpointer = {}
def get_checkpointer() -> BaseCheckpointSaver:
    """Return an implementation of BaseCheckpointSaver for graphs to store the state"""
    return checkpointer["checkpointer"]