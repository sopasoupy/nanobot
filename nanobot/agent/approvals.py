"""Helpers for tool approval gating."""

from __future__ import annotations

import json
import re
from typing import Any

from nanobot.config.schema import ToolApprovalConfig


_APPROVAL_RE = re.compile(r"^\s*(approve|deny)\s+([A-Za-z0-9_\-:.]+)\s*[.!?]*\s*$", re.IGNORECASE)
_BARE_TOKEN_RE = re.compile(r"^\s*(A\d+)\s*[.!?]*\s*$", re.IGNORECASE)


def allocate_approval_token(metadata: dict[str, Any]) -> str:
    """
    Allocate a short, user-friendly approval token (A1, A2, ...) scoped to a session.

    The long internal tool call IDs can be cumbersome to type; this provides a stable
    short token while a single approval is pending.
    """
    current = metadata.get("approval_counter", 0)
    try:
        current_int = int(current)
    except (TypeError, ValueError):
        current_int = 0
    current_int += 1
    metadata["approval_counter"] = current_int
    return f"A{current_int}"


def requires_approval(tool_name: str, cfg: ToolApprovalConfig) -> bool:
    """Check if a tool requires explicit approval."""
    return tool_name in cfg.required


def format_args_preview(args: dict[str, Any], detail: str, max_len: int = 400) -> str:
    """Format a tool argument preview based on detail level."""
    if detail == "minimal":
        return ""

    try:
        rendered = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except TypeError:
        rendered = str(args)

    if detail == "summary" and len(rendered) > max_len:
        return rendered[:max_len] + "..."
    return rendered


def make_approval_prompt(approval_id: str, tool_name: str, args_preview: str) -> str:
    """Build the user-facing approval prompt."""
    parts = [f"Approval needed to run tool {tool_name} (ref: {approval_id})."]
    if args_preview:
        parts.append(f"Args: {args_preview}")
    parts.append(f"Reply with: {approval_id} / approve {approval_id} / deny {approval_id}")
    return "\n".join(parts)


def parse_approval_message(text: str) -> tuple[str, str] | None:
    """
    Parse approval commands.

    Accepted forms:
    - 'approve <ref>' or 'deny <ref>'
    - bare 'A<n>' (treated as 'approve A<n>')
    """
    stripped = text.strip()
    bare = _BARE_TOKEN_RE.match(stripped)
    if bare:
        return "approve", bare.group(1).lower()

    match = _APPROVAL_RE.match(stripped)
    if not match:
        return None
    return match.group(1).lower(), match.group(2).lower()
