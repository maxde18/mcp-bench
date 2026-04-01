"""MCP Connection Manager.

Manages the async lifecycle of connections to one or more MCP servers.
Used by both the planning runner (tool discovery only) and the full
benchmark runner (tool discovery + execution).
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from mcp_infra.server_manager_persistent import PersistentMultiServerManager

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Async context manager for MCP server connections.

    Connects to all configured servers on entry, exposes discovered tool
    schemas via ``all_tools``, and tears down connections on exit.

    Attributes:
        server_configs: List of server configuration dicts.
        filter_problematic_tools: If True, filters known-problematic tools.
        server_manager: The active PersistentMultiServerManager (set on entry).
        all_tools: Dict of all discovered tool schemas (set on entry).

    Example::

        async with ConnectionManager(server_configs) as conn:
            # Planning: use conn.all_tools for prompt construction
            # Execution: use conn.server_manager.call_tool(...)
    """

    def __init__(
        self,
        server_configs: List[Dict[str, Any]],
        filter_problematic_tools: bool = False,
        server_manager: Optional[PersistentMultiServerManager] = None,
    ) -> None:
        self.server_configs = server_configs
        self.filter_problematic_tools = filter_problematic_tools
        self._injected_server_manager = server_manager
        self.server_manager: Optional[PersistentMultiServerManager] = None
        self.all_tools: Optional[Dict[str, Any]] = None

    async def __aenter__(self) -> "ConnectionManager":
        self.server_manager = self._injected_server_manager or PersistentMultiServerManager(
            self.server_configs,
            self.filter_problematic_tools,
        )
        self.all_tools = await self.server_manager.connect_all_servers()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> bool:
        if self.server_manager:
            try:
                await self.server_manager.close_all_connections()
            except asyncio.CancelledError:
                logger.debug("Ignoring CancelledError during connection cleanup")
            except Exception as e:
                logger.error(f"Error during connection cleanup: {e}")
            finally:
                self.server_manager = None
                self.all_tools = None
        return False
