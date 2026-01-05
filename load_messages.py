
from types import SimpleNamespace
from langgraph_setup import graph

def get_msg_role(msg):
    cls = msg.__class__.__name__.lower()
    if "human" in cls: return "human"
    if "ai" in cls: return "ai"
    return "unknown"


def load_thread_messages(thread_id: str, limit: int = 50):
    """Each message gets its own checkpoint timestamp"""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoints = list(graph.checkpointer.list(config, limit=20))  # Recent only
    
    all_msgs = []
    for cp in sorted(checkpoints, key=lambda x: x.checkpoint["v"]):
        ts = cp.checkpoint["ts"]
        messages = cp.checkpoint.get('channel_values', {}).get("messages", [])
        
        # Only NEW messages since last checkpoint
        for msg in messages[-3:]:  # Last few per checkpoint
            if isinstance(msg, dict):
                msg = SimpleNamespace(**msg)
            
            content = str(getattr(msg, "content", "")).strip()
            if content and not getattr(msg, "tool_call_id", None):
                msg_id = getattr(msg, 'id', '')
                all_msgs.append({
                    "id": msg_id,
                    "content": content,
                    "role": get_msg_role(msg),
                    "time": ts  # Each msg gets its checkpoint's time
                })
    
    # Dedupe + limit
    seen = set()
    unique = []
    for msg in all_msgs:
        if msg["id"] not in seen:
            seen.add(msg["id"])
            unique.append(msg)
            if len(unique) >= limit:
                break

    return sorted(unique, key=lambda x: x["time"])