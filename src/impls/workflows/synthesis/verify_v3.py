import re
from copy import deepcopy

from workflows import Workflow, step, Context
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage


class VerifyWorkflow(Workflow):
    def __init__(
            self,
            llm: LLM,
            threshold: int = 3,
            **kwargs
    ):
        self.llm = llm
        self.threshold = threshold
        super().__init__(**kwargs)

    @step

