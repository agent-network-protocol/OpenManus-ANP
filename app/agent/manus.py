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
        if browser_state and not browser_state.get("error"):
            self.next_step_prompt += f"\nCurrent browser state:\nURL: {browser_state.get('url', 'N/A')}\nTitle: {browser_state.get('title', 'N/A')}\n"

        # Call parent implementation
        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result
