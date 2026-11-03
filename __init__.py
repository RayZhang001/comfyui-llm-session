"""ComfyUI LLM Session.

One session node that turns a prompt plus optional images into text using a
managed local inference server. The initial backend is llama.cpp
(``llama-server``); the package name and the node contract are
backend-neutral on purpose.

The package is standalone: it does not import or depend on any other custom
node pack.
"""

from .display_lab_node import DisplayLabNode
from .session_node import NODE_NAME, LLMSessionNode

NODE_CLASS_MAPPINGS = {
    NODE_NAME: LLMSessionNode,
    "ComfyLLMSessionDisplayLab": DisplayLabNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: "LLM Session (llama.cpp)",
    "ComfyLLMSessionDisplayLab": "LLM Session Display Lab",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
