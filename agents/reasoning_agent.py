# backend/agents/reasoning_agent.py

def reasoning_agent(context: str) -> str:
    # Clean garbage spaces, keep tables intact
    lines = [line.strip() for line in context.split("\n") if line.strip()]
    return "\n".join(lines)
