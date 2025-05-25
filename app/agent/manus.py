import json
from typing import Any, Optional

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.anp_tool import ANPTool
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_saver import FileSaver
from app.tool.python_execute import PythonExecute


class Manus(ToolCallAgent):
    """
    A versatile general-purpose agent that uses planning to solve various tasks.

    This agent extends PlanningAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 2000
    max_steps: int = 20

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            FileSaver(),
            ANPTool(),
            Terminate(),
        )
    )

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        if not self._is_special_tool(name):
            return
        else:
            await self.available_tools.get_tool(BrowserUseTool().name).cleanup()
            await super()._handle_special_tool(name, result, **kwargs)

    async def get_browser_state(self) -> Optional[dict]:
        """Get the current browser state for context in next steps."""
        browser_tool = self.available_tools.get_tool(BrowserUseTool().name)
        if not browser_tool:
            return None

        try:
            # Get browser state directly from the tool with no context parameter
            result = await browser_tool.get_current_state()

            if result.error:
                logger.debug(f"Browser state error: {result.error}")
                return None

            # Store screenshot if available
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image

            # Parse the state info
            return json.loads(result.output)

        except Exception as e:
            logger.debug(f"Failed to get browser state: {str(e)}")
            return None

    async def think(self) -> bool:
        # Add your custom pre-processing here
        browser_state = await self.get_browser_state()

        # Modify the next_step_prompt temporarily
        original_prompt = self.next_step_prompt
        current_next_step_prompt = list(
            self.next_step_prompt
        )  # Make it a list to append strings

        if browser_state and not browser_state.get("error"):
            current_next_step_prompt.append(
                f"\nCurrent browser state:\nURL: {browser_state.get('url', 'N/A')}\nTitle: {browser_state.get('title', 'N/A')}\n"
            )

        # Check if the last tool call was ANPTool or if ANPTool is intended to be called
        # This is a simplified check, more sophisticated logic might be needed
        anp_tool_was_called = False
        if self.memory.messages:
            last_message = self.memory.messages[-1]
            if (
                last_message.role == "tool" and last_message.name == ANPTool().name
            ):  # Accessing name directly
                anp_tool_was_called = True

        # Potential: Check if LLM intends to call ANPTool in the current thought process
        # This would require a preliminary call to LLM or parsing its initial thought if available
        # For now, we'll rely on the history or a more explicit trigger.

        if anp_tool_was_called:  # Or some other condition indicating ANPTool context
            anp_detailed_description = """

ANPTool Detailed Workflow:
1.  **Agent/Service Discovery**: If you don't know the specific service URL, you can first use ANPTool to query an agent directory (e.g., an initial URL provided by the user or planner, like 'https://agent-search.ai/ad.json').
2.  **Parse Agent Description (AD)**: The response from the discovery service or a direct agent URL will be an Agent Description (usually in JSON-LD or a similar format). Analyze it to understand the agent's capabilities, available API interfaces, and their descriptions.
3.  **Discover API Endpoint URL**: Extract specific API endpoint URLs from the AD (look for fields like `serviceEndpoint`, `url`, `@id`).
4.  **Fetch API Specification (OpenAPI/YAML)**: If its URL is provided in the AD, use ANPTool again to fetch the OpenAPI/YAML specification for the selected API endpoint. This specification details how to call the API (methods, parameters, request/response formats).
5.  **Execute Service Call**: Construct the correct request (method, headers, parameters, body) based on the API specification and use ANPTool to call the actual service API endpoint.
6.  **Process Response**: ANPTool will return the response from the external agent/service. This could be JSON, YAML (as text), or other text formats.
7.  **Iterative Process**: Complex tasks may involve multiple calls to ANPTool: discovery, fetching specifications, and then multiple interactions with the target service API.

**Key Rule for ANPTool**: Any URL obtained via ANPTool (e.g., agent description URL, API specification URL, or API endpoint URL) MUST subsequently be accessed using ANPTool itself. Do not attempt to use BrowserUseTool or other methods for these URLs.
"""
            current_next_step_prompt.append(anp_detailed_description)

        self.next_step_prompt = "".join(current_next_step_prompt)

        # Call parent implementation
        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result
