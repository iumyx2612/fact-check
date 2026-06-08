from typing import Any, Optional
import asyncio
import json
import re

from llama_index.core.schema import TransformComponent, BaseNode, MetadataMode
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import EntityNode, Relation, KG_NODES_KEY, KG_RELATIONS_KEY

from src.modules.utils.json_utils import fix_broken_generated_json

NER_SYSTEM = """Your task is to extract named entities from the given paragraph.
Please do NOT alter the named entities, extract the entities verbatim.
Respond with a JSON list of entities.
"""

NER_EXAMPLE_INPUT = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

NER_EXAMPLE_OUTPUT = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

NER_USER = "{content}"

TRIPLET_SYSTEM = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
"""

TRIPLET_EXAMPLE_INPUT = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{content}
```

{named_entity_json}
""" # Later will be format with content=NER_EXAMPLE_INPUT, named_entity_json=NER_EXAMPLE_OUTPUT

TRIPLET_EXAMPLE_OUTPUT = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

TRIPLET_USER = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{content}
```

{named_entity_json}
"""


def _parse_ner_response(response: str) -> list[str]:
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, response, re.DOTALL)
    if match is None:
        return []
    try:
        return json.loads(match.group())["named_entities"]
    except (json.JSONDecodeError, KeyError):
        return []


def _parse_triplet_response(response: str) -> list[tuple[str, str, str]]:
    pattern = r'\{[^{}]*"triples"\s*:\s*\[.*?\][^{}]*\}'
    match = re.search(pattern, response, re.DOTALL)
    if match is None:
        return []
    try:
        raw_triples = json.loads(match.group())["triples"]
    except (json.JSONDecodeError, KeyError):
        return []
    return [
        (t[0], t[1], t[2])
        for t in raw_triples
        if isinstance(t, list) and len(t) == 3
    ]


class HippoRAGExtractor(TransformComponent):
    llm: Optional[LLM] = None
    num_workers: int = 4

    def __init__(self,
                 llm: LLM = None,
                 num_workers: int = 4):
        super().__init__(
            llm=llm,
            num_workers=num_workers
        )

    @classmethod
    def class_name(cls) -> str:
        return "HippoRAGExtractor"

    def __call__(
        self, nodes: list[BaseNode], show_progress: bool = True, **kwargs: Any
    ) -> list[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs)) # noqa

    async def acall(
        self, nodes: list[BaseNode], show_progress: bool = True, **kwargs: Any
    ) -> list[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting triples from documents",
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        # NER step
        content = node.get_content(metadata_mode=MetadataMode.NONE)
        if not content.strip():
            return node

        messages = [
            ChatMessage(role="system", content=NER_SYSTEM),
            ChatMessage(role="user", content=NER_EXAMPLE_INPUT),
            ChatMessage(role="assistant", content=NER_EXAMPLE_OUTPUT),
            ChatMessage(role="user", content=NER_USER.format(content=content)),
        ]

        response = await self.llm.achat(messages)
        raw = response.message.content or ""
        raw = fix_broken_generated_json(raw)
        entities = _parse_ner_response(raw)

        # Triplet extraction
        named_entity_json = json.dumps({"named_entities": entities})

        triplet_messages = [
            ChatMessage(role="system", content=TRIPLET_SYSTEM),
            ChatMessage(role="user", content=TRIPLET_EXAMPLE_INPUT.format(
                content=NER_EXAMPLE_INPUT,
                named_entity_json=NER_EXAMPLE_OUTPUT,
            )),
            ChatMessage(role="assistant", content=TRIPLET_EXAMPLE_OUTPUT),
            ChatMessage(role="user", content=TRIPLET_USER.format(
                content=content,
                named_entity_json=named_entity_json,
            )),
        ]

        response = await self.llm.achat(triplet_messages)
        raw = response.message.content or ""
        raw = fix_broken_generated_json(raw)
        triples = _parse_triplet_response(raw)

        # Convert triples to EntityNode + Relation and write back
        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        # TODO: Fix sometimes LLM generate wrong json template
        # Example:
        # {"triples": [
        #       [["Radio City", "ABC City"], "located in", "India"],  -> Entity is a list
        #   ]
        # }

        try:
            for subj, pred, obj in triples:
                subj_node = EntityNode(name=subj, properties=node.metadata.copy())
                obj_node = EntityNode(name=obj, properties=node.metadata.copy())
                rel = Relation(
                    label=pred,
                    source_id=subj_node.id,
                    target_id=obj_node.id,
                    properties=node.metadata.copy(),
                )
                existing_nodes.extend([subj_node, obj_node])
                existing_relations.append(rel)
        except Exception as e:
            print(e)
            print(node.metadata["doc_id"])
            print(raw)
        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node
