"""Agent loop: the core processing engine."""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.approvals import (
    requires_approval,
    format_args_preview,
    allocate_approval_token,
    make_approval_prompt,
    parse_approval_message,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        approval_config: "ToolApprovalConfig | None" = None,
        session_manager: SessionManager | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.config.schema import ToolApprovalConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.approval_config = approval_config or ToolApprovalConfig()
        
        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            approval_config=self.approval_config,
        )
        
        self._running = False
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
    
    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
        
        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)
        pending = session.metadata.get("pending_tool_approval")
        if pending:
            return await self._handle_pending_approval(msg, session, pending)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)
        
        # Build initial messages (use get_history for LLM-formatted messages)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        
        # Agent loop
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)  # Must be JSON string
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                
                # Execute tools (with optional approval gate)
                for tool_call in response.tool_calls:
                    if requires_approval(tool_call.name, self.approval_config):
                        args_preview = format_args_preview(
                            tool_call.arguments, self.approval_config.detail
                        )
                        approval_ref = allocate_approval_token(session.metadata)
                        approval_prompt = make_approval_prompt(
                            approval_ref, tool_call.name, args_preview
                        )
                        session.metadata["pending_tool_approval"] = {
                            "approval_ref": approval_ref,
                            "approval_ref_norm": approval_ref.lower(),
                            "tool_call_id": tool_call.id,
                            "tool_call_id_norm": tool_call.id.lower(),
                            "tool_name": tool_call.name,
                            "arguments": tool_call.arguments,
                            "assistant_content": response.content or "",
                            "approval_prompt": approval_prompt,
                            "origin": {"type": "agent"},
                        }
                        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
                        session.add_message("assistant", approval_prompt)
                        self.sessions.save(session)
                        return OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=approval_prompt,
                        )
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # No tool calls, we're done
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        # Log response preview
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        if msg.sender_id == "subagent-approval":
            return await self._handle_subagent_approval_request(msg)
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(origin_channel, origin_chat_id)
        
        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        
        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                
                for tool_call in response.tool_calls:
                    if requires_approval(tool_call.name, self.approval_config):
                        args_preview = format_args_preview(
                            tool_call.arguments, self.approval_config.detail
                        )
                        approval_ref = allocate_approval_token(session.metadata)
                        approval_prompt = make_approval_prompt(
                            approval_ref, tool_call.name, args_preview
                        )
                        session.metadata["pending_tool_approval"] = {
                            "approval_ref": approval_ref,
                            "approval_ref_norm": approval_ref.lower(),
                            "tool_call_id": tool_call.id,
                            "tool_call_id_norm": tool_call.id.lower(),
                            "tool_name": tool_call.name,
                            "arguments": tool_call.arguments,
                            "assistant_content": response.content or "",
                            "approval_prompt": approval_prompt,
                            "origin": {"type": "agent"},
                        }
                        session.add_message("user", msg.content)
                        session.add_message("assistant", approval_prompt)
                        self.sessions.save(session)
                        return OutboundMessage(
                            channel=origin_channel,
                            chat_id=origin_chat_id,
                            content=approval_prompt,
                        )
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "Background task completed."
        
        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier.
            channel: Source channel (for context).
            chat_id: Source chat ID (for context).
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg)
        return response.content if response else ""

    async def _handle_pending_approval(
        self,
        msg: InboundMessage,
        session: "Session",
        pending: dict[str, Any],
    ) -> OutboundMessage:
        action = parse_approval_message(msg.content)
        approval_ref = pending.get("approval_ref") or "A?"
        approval_ref_norm = pending.get("approval_ref_norm") or str(approval_ref).lower()
        allowed_refs = {approval_ref_norm}
        if pending.get("tool_call_id_norm"):
            allowed_refs.add(pending["tool_call_id_norm"])
        if pending.get("subagent_approval_id_norm"):
            allowed_refs.add(pending["subagent_approval_id_norm"])

        if not action or action[1] not in allowed_refs:
            prompt = pending.get("approval_prompt") or make_approval_prompt(
                approval_ref, pending.get("tool_name", "tool"), ""
            )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=prompt,
            )
        decision = action[0]
        origin = pending.get("origin", {"type": "agent"})
        approval_prompt = pending.get("approval_prompt")
        if approval_prompt and session.messages and session.messages[-1].get("content") == approval_prompt:
            session.messages.pop()
        session.metadata.pop("pending_tool_approval", None)
        self.sessions.save(session)
        if origin.get("type") == "subagent":
            subagent_approval_id = pending.get("subagent_approval_id") or ""
            self.subagents.resolve_approval(subagent_approval_id, decision == "approve")
            if decision == "approve":
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Approval granted. The background task will continue.",
                )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Approval denied. The background task was canceled.",
            )
        if decision == "deny":
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Approval denied. Tool execution canceled.",
            )
        return await self._execute_approved_tool(msg, session, pending)

    async def _execute_approved_tool(
        self,
        msg: InboundMessage,
        session: "Session",
        pending: dict[str, Any],
    ) -> OutboundMessage:
        system_prompt = self.context.build_system_prompt()
        system_prompt += f"\n\n## Current Session\nChannel: {msg.channel}\nChat ID: {msg.chat_id}"
        messages = [{"role": "system", "content": system_prompt}]
        history = session.get_history()
        approval_prompt = pending.get("approval_prompt")
        if approval_prompt and history and history[-1].get("content") == approval_prompt:
            history = history[:-1]
        messages.extend(history)
        tool_call = {
            "id": pending["tool_call_id"],
            "type": "function",
            "function": {
                "name": pending["tool_name"],
                "arguments": json.dumps(pending["arguments"]),
            },
        }
        messages = self.context.add_assistant_message(
            messages, pending.get("assistant_content", ""), [tool_call]
        )
        result = await self.tools.execute(pending["tool_name"], pending["arguments"])
        messages = self.context.add_tool_result(
            messages, pending["tool_call_id"], pending["tool_name"], result
        )
        iteration = 0
        final_content = None
        while iteration < self.max_iterations:
            iteration += 1
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
            )
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )
                for tool_call in response.tool_calls:
                    if requires_approval(tool_call.name, self.approval_config):
                        args_preview = format_args_preview(
                            tool_call.arguments, self.approval_config.detail
                        )
                        approval_ref = allocate_approval_token(session.metadata)
                        approval_prompt = make_approval_prompt(
                            approval_ref, tool_call.name, args_preview
                        )
                        session.metadata["pending_tool_approval"] = {
                            "approval_ref": approval_ref,
                            "approval_ref_norm": approval_ref.lower(),
                            "tool_call_id": tool_call.id,
                            "tool_call_id_norm": tool_call.id.lower(),
                            "tool_name": tool_call.name,
                            "arguments": tool_call.arguments,
                            "assistant_content": response.content or "",
                            "approval_prompt": approval_prompt,
                            "origin": {"type": "agent"},
                        }
                        session.add_message("assistant", approval_prompt)
                        self.sessions.save(session)
                        return OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=approval_prompt,
                        )
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
        )

    async def _handle_subagent_approval_request(self, msg: InboundMessage) -> OutboundMessage:
        import json as _json
        payload = _json.loads(msg.content)
        subagent_approval_id = payload["approval_id"]
        tool_name = payload["tool_name"]
        args_preview = payload.get("args_preview", "")
        session = self.sessions.get_or_create(payload["session_key"])
        approval_ref = allocate_approval_token(session.metadata)
        approval_prompt = make_approval_prompt(approval_ref, tool_name, args_preview)
        if session.metadata.get("pending_tool_approval"):
            return OutboundMessage(
                channel=payload["channel"],
                chat_id=payload["chat_id"],
                content="Another approval is already pending. Please respond to it first.",
            )
        session.metadata["pending_tool_approval"] = {
            "approval_ref": approval_ref,
            "approval_ref_norm": approval_ref.lower(),
            "subagent_approval_id": subagent_approval_id,
            "subagent_approval_id_norm": subagent_approval_id.lower(),
            "tool_name": tool_name,
            "arguments": payload.get("arguments", {}),
            "assistant_content": "",
            "approval_prompt": approval_prompt,
            "origin": {"type": "subagent"},
        }
        session.add_message("assistant", approval_prompt)
        self.sessions.save(session)
        return OutboundMessage(
            channel=payload["channel"],
            chat_id=payload["chat_id"],
            content=approval_prompt,
        )
