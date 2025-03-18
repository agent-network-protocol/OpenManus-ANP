import asyncio
import json
import yaml
import aiohttp
from pathlib import Path
from typing import Dict, Any, Optional

from app.logger import logger
from app.tool.base import BaseTool
from agent_connect.authentication import DIDWbaAuthHeader
from app.config import config, PROJECT_ROOT


class ANPTool(BaseTool):
    name: str = "anp_tool"
    description: str = """Use Agent Network Protocol (ANP) to interact with other agents.
1. For the first use, please enter the URL: https://agent-search.ai/ad.json, which is an agent search service. You can use the interfaces inside to query agents that can provide hotels, tickets, and attractions.
2. After receiving the agent's description document, you can crawl data based on the data link URL in the agent's description document.
3. During the process, you can call the API to complete the service until you think the task is completed.
4. Note, any URL obtained using ANPTool must be called using ANPTool, do not call it directly yourself.
"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "(required) URL of the agent description file or API endpoint",
            },
            "method": {
                "type": "string",
                "description": "(optional) HTTP method, such as GET, POST, PUT, etc., default is GET",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "default": "GET",
            },
            "headers": {
                "type": "object",
                "description": "(optional) HTTP request headers",
                "default": {},
            },
            "params": {
                "type": "object",
                "description": "(optional) URL query parameters",
                "default": {},
            },
            "body": {
                "type": "object",
                "description": "(optional) Request body for POST/PUT requests",
            },
        },
        "required": ["url"],
    }

    # Declare auth_client field
    auth_client: Optional[DIDWbaAuthHeader] = None

    def __init__(self, **data):
        super().__init__(**data)

        # Get default paths relative to project root
        default_did_path = str(PROJECT_ROOT / "config/did_test_public_doc/did.json")
        default_key_path = str(
            PROJECT_ROOT / "config/did_test_public_doc/key-1_private.pem"
        )

        # Use paths from configuration if available, otherwise use defaults
        did_path = default_did_path
        key_path = default_key_path

        if config.anp_config:
            if config.anp_config.did_document_path:
                did_path = config.anp_config.did_document_path
            if config.anp_config.private_key_path:
                key_path = config.anp_config.private_key_path

        logger.info(
            f"ANPTool initialized - DID path: {did_path}, private key path: {key_path}"
        )

        self.auth_client = DIDWbaAuthHeader(
            did_document_path=did_path, private_key_path=key_path
        )

    async def execute(
        self,
        url: str,
        method: str = "GET",
        headers: Dict[str, str] = None,
        params: Dict[str, Any] = None,
        body: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute HTTP request to interact with other agents

        Args:
            url (str): URL of the agent description file or API endpoint
            method (str, optional): HTTP method, default is "GET"
            headers (Dict[str, str], optional): HTTP request headers
            params (Dict[str, Any], optional): URL query parameters
            body (Dict[str, Any], optional): Request body for POST/PUT requests

        Returns:
            Dict[str, Any]: Response content
        """

        if headers is None:
            headers = {}
        if params is None:
            params = {}

        logger.info(f"ANP request: {method} {url}")

        # Add basic request headers
        if "Content-Type" not in headers and method in ["POST", "PUT", "PATCH"]:
            headers["Content-Type"] = "application/json"

        # Add DID authentication
        if self.auth_client:
            try:
                auth_headers = self.auth_client.get_auth_header(url)
                headers.update(auth_headers)
            except Exception as e:
                logger.error(f"Failed to get authentication header: {str(e)}")

        async with aiohttp.ClientSession() as session:
            # Prepare request parameters
            request_kwargs = {
                "url": url,
                "headers": headers,
                "params": params,
            }

            # If there is a request body and the method supports it, add the request body
            if body is not None and method in ["POST", "PUT", "PATCH"]:
                request_kwargs["json"] = body

            # Execute request
            http_method = getattr(session, method.lower())

            try:
                async with http_method(**request_kwargs) as response:
                    logger.info(f"ANP response: status code {response.status}")

                    # Check response status
                    if (
                        response.status == 401
                        and "Authorization" in headers
                        and self.auth_client
                    ):
                        logger.warning(
                            "Authentication failed (401), trying to get authentication again"
                        )
                        # If authentication fails and a token was used, clear the token and retry
                        self.auth_client.clear_token(url)
                        # Get authentication header again
                        headers.update(
                            self.auth_client.get_auth_header(url, force_new=True)
                        )
                        # Execute request again
                        request_kwargs["headers"] = headers
                        async with http_method(**request_kwargs) as retry_response:
                            logger.info(
                                f"ANP retry response: status code {retry_response.status}"
                            )
                            return await self._process_response(retry_response, url)

                    return await self._process_response(response, url)
            except aiohttp.ClientError as e:
                logger.error(f"HTTP request failed: {str(e)}")
                return {"error": f"HTTP request failed: {str(e)}", "status_code": 500}

    async def _process_response(self, response, url):
        """Process HTTP response"""
        # If authentication is successful, update the token
        if response.status == 200 and self.auth_client:
            try:
                self.auth_client.update_token(url, dict(response.headers))
            except Exception as e:
                logger.error(f"Failed to update token: {str(e)}")

        # Get response content type
        content_type = response.headers.get("Content-Type", "").lower()

        # Get response text
        text = await response.text()

        # Process response based on content type
        if "application/json" in content_type:
            # Process JSON response
            try:
                result = json.loads(text)
                logger.info("Successfully parsed JSON response")
            except json.JSONDecodeError:
                logger.warning(
                    "Content-Type declared as JSON but parsing failed, returning raw text"
                )
                result = {"text": text, "format": "text", "content_type": content_type}
        elif "application/yaml" in content_type or "application/x-yaml" in content_type:
            # Process YAML response
            try:
                result = yaml.safe_load(text)
                logger.info("Successfully parsed YAML response")
                result = {
                    "data": result,
                    "format": "yaml",
                    "content_type": content_type,
                }
            except yaml.YAMLError:
                logger.warning(
                    "Content-Type declared as YAML but parsing failed, returning raw text"
                )
                result = {"text": text, "format": "text", "content_type": content_type}
        else:
            # Default to text
            result = {"text": text, "format": "text", "content_type": content_type}

        # Add status code to result
        if isinstance(result, dict):
            result["status_code"] = response.status
        else:
            result = {
                "data": result,
                "status_code": response.status,
                "format": "unknown",
                "content_type": content_type,
            }

        # Add URL to result for tracking
        result["url"] = str(url)

        return result
