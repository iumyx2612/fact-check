import re

from workflows import Workflow, step, Context
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage
from llama_index.core.graph_stores.types import (
    Triplet,
    EntityNode,
    Relation
)

from ...events.synthesis.decompose_v2 import (
    DecomposeStartEvent,
    DecomposeExtractKnownEntities,
    DecomposeExtractRelation,
    DecomposeGleaningEntity,
    DecomposeMergeGraph,
    DecomposeVerifyGraph,
    DecomposeStopEvent
)
from .context import SynthesisContext
from src.modules.schema.synthesis.decompose_v2 import DecompositionVerdict, SubClaimVerdict
from src.modules.prompts.synthesis.decompose_v2 import (
    KNOWN_ENTITY_EXTRACTION_SYSTEM,
    KNOWN_ENTITY_EXTRACTION_USER,
    KNOWN_ENTITY_RANKING_SYSTEM,
    KNOWN_ENTITY_RANKING_USER,
    ENTITY_RELATION_EXTRACTION_SYSTEM,
    ENTITY_RELATION_EXTRACTION_SYSTEM_V2,
    ENTITY_RELATION_EXTRACTION_USER,
    ENTITY_GLEANING_SYSTEM,
    ENTITY_GLEANING_USER,
    MERGE_ENTITY_SYSTEM,
    MERGE_ENTITY_USER,
    VERIFY_ENTITY_SYSTEM,
    VERIFY_ENTITY_USER
)


class DecomposeWorkflow(Workflow):
    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.verify_llm = llm.as_structured_llm(DecompositionVerdict)

    @step
    async def initialize(
            self, ctx: Context[SynthesisContext], ev: DecomposeStartEvent
    ) -> DecomposeExtractKnownEntities:
        claim = ev.claim

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.claim = claim

        response = await self.llm.achat([
            ChatMessage(
                content=KNOWN_ENTITY_EXTRACTION_SYSTEM,
                role="system"
            ),
            ChatMessage(
                content=KNOWN_ENTITY_EXTRACTION_USER.format(
                    claim=claim,
                ),
                role="user"
            )
        ])

        content = response.message.content
        entities = []
        for line in content.strip().splitlines():
            line = line.strip()
            match = re.match(r"^\d+\.\s+(.+)$", line)
            if match:
                entities.append(match.group(1))

        return DecomposeExtractKnownEntities(entities=entities)

    @step
    async def extract_known_entities(
            self, ctx: Context[SynthesisContext], ev: DecomposeExtractKnownEntities
    ) -> DecomposeExtractRelation:
        claim = await ctx.store.get("claim")
        entities = ev.entities

        # Turn list of entities to bulleted list of entities
        entities = "\n".join(f"{i + 1}. {entity}" for i, entity in enumerate(entities))

        response = await self.llm.achat([
            ChatMessage(
                content=KNOWN_ENTITY_RANKING_SYSTEM,
                role="system"
            ),
            ChatMessage(
                content=KNOWN_ENTITY_RANKING_USER.format(
                    claim=claim,
                    entities=entities,
                ),
                role="user"
            )
        ])
        content = response.message.content
        # Extract entity after re-ranked
        ranked_entities = []
        in_final_ranking = False
        for line in content.strip().splitlines():
            line = line.strip()
            if line.startswith("# Final Ranking"):
                in_final_ranking = True
                continue
            if in_final_ranking:
                match = re.match(r"^\d+\.\s+(.+)$", line)
                if match:
                    ranked_entities.append(match.group(1))

        return DecomposeExtractRelation(entities=ranked_entities)

    @step
    async def extract_relation(
            self, ctx: Context[SynthesisContext], ev: DecomposeExtractRelation
    ) -> DecomposeGleaningEntity:
        claim = await ctx.store.get("claim")
        ranked_entities = ev.entities

        # Extract triplets from content, stored it in Triplet
        triplets: list[Triplet] = []
        for entity in ranked_entities:
            response = await self.llm.achat([
                ChatMessage(
                    content=ENTITY_RELATION_EXTRACTION_SYSTEM_V2,
                    role="system"
                ),
                ChatMessage(
                    content=ENTITY_RELATION_EXTRACTION_USER.format(
                        claim=claim,
                        entity=entity
                    ),
                    role="user"
                )
            ])
            content = response.message.content

            for line in content.strip().splitlines():
                parts = [p.strip() for p in line.split("->")]
                if len(parts) == 3:
                    source, rel, target = parts
                    source_entity = EntityNode(name=source)
                    target_entity = EntityNode(name=target)
                    triplets.append((
                        source_entity,
                        Relation(label=rel, source_id=source_entity.id, target_id=target_entity.id),
                        target_entity
                    ))

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.decompose_ctx.first_round_entities = ranked_entities
            ctx_state.decompose_ctx.first_round_triplets = triplets

        return DecomposeGleaningEntity(existing_triplets=triplets)

    @step
    async def gleaning(
            self, ctx: Context[SynthesisContext], ev: DecomposeGleaningEntity
    ) -> DecomposeMergeGraph:
        existing_triplets = ev.existing_triplets
        claim = await ctx.store.get("claim")
        existing_entities = await ctx.store.get("decompose_ctx.first_round_entities")

        # Extract entities from existing triplets that has different name from existing_entities
        existing_ids = set(existing_entities)
        remaining_entities = []
        existing_relations = []
        seen = set()
        for source, relation, target in existing_triplets:
            existing_relations.append(
                f"{source.id} -> {relation.id} -> {target.id}"
            )
            for node in (source, target):
                if node.id not in existing_ids and node.id not in seen:
                    remaining_entities.append(node.id)
                    seen.add(node.id)

        # Make prompt
        existing_relations = "\n".join(existing_relations)
        remaining_entities = "\n".join(f"{i + 1}. {entity}" for i, entity in enumerate(remaining_entities))
        response = await self.llm.achat([
            ChatMessage(
                content=ENTITY_GLEANING_SYSTEM,
                role="system"
            ),
            ChatMessage(
                content=ENTITY_GLEANING_USER.format(
                    claim=claim,
                    entities=remaining_entities,
                    triplets=existing_relations
                )
            )
        ])

        # Extract triplets from content
        content = response.message.content
        new_triplets: list[Triplet] = []
        if content:
            for line in content.strip().splitlines():
                parts = [p.strip() for p in line.split("->")]
                if len(parts) == 3:
                    source, rel, target = parts
                    source_entity = EntityNode(name=source)
                    target_entity = EntityNode(name=target)
                    new_triplets.append((
                        source_entity,
                        Relation(label=rel, source_id=source_entity.id, target_id=target_entity.id),
                        target_entity
                    ))
        all_triplets = existing_triplets + new_triplets

        return DecomposeMergeGraph(all_triplets=all_triplets)

    @step
    async def merge_graph(
            self, ctx: Context[SynthesisContext], ev: DecomposeMergeGraph
    ) -> DecomposeVerifyGraph:
        claim = await ctx.store.get("claim")
        triplets = ev.all_triplets
        triplet_string = []
        for triplet in triplets:
            triplet_string.append(
                f"{triplet[0].id} -> {triplet[1].id} -> {triplet[2].id}"
            )
        triplets = "\n".join(triplet_string)

        response = await self.llm.achat([
            ChatMessage(
                content=MERGE_ENTITY_SYSTEM,
                role="system"
            ),
            ChatMessage(
                content=MERGE_ENTITY_USER.format(
                    claim=claim,
                    triplets=triplets
                ),
                role="user"
            )
        ])

        # Extract the triplet from content
        content = response.message.content
        final_triplets: list[Triplet] = []
        for line in content.strip().splitlines():
            line = line.strip()
            if line.startswith("```") or not line:
                continue
            parts = [p.strip() for p in line.split("->")]
            if len(parts) == 3:
                source, rel, target = parts
                source_entity = EntityNode(name=source)
                target_entity = EntityNode(name=target)
                final_triplets.append((
                    source_entity,
                    Relation(label=rel, source_id=source_entity.id, target_id=target_entity.id),
                    target_entity
                ))

        return DecomposeVerifyGraph(all_triplets=final_triplets)

    @step
    async def verify_graph(
            self, ctx: Context[SynthesisContext], ev: DecomposeVerifyGraph
    ) -> DecomposeStopEvent:
        claim = await ctx.store.get("claim")
        triplets = ev.all_triplets
        triplet_string = []
        for i, triplet in enumerate(triplets):
            triplet_string.append(
                f"{i+1}. {triplet[0].id} -> {triplet[1].id} -> {triplet[2].id}"
            )
        triplet_string = "\n".join(triplet_string)

        response = await self.verify_llm.achat([
            ChatMessage(
                content=VERIFY_ENTITY_SYSTEM,
                role="system"
            ),
            ChatMessage(
                content=VERIFY_ENTITY_USER.format(
                    claim=claim,
                    triplets=triplet_string
                )
            )
        ])
        verdicts = response.raw.verdicts

        new_triplets: list[Triplet] = []
        for i, verdict in enumerate(verdicts):
            if verdict.correct:
                new_triplets.append(triplets[i])
            else:
                if not verdict.correct_sub_claim:
                    continue
                new_triplet = verdict.correct_sub_claim
                parts = [p.strip() for p in new_triplet.split("->")]
                if len(parts) == 3:
                    source, rel, target = parts
                    source_entity = EntityNode(name=source)
                    target_entity = EntityNode(name=target)
                    new_triplets.append((
                        source_entity,
                        Relation(label=rel, source_id=source_entity.id, target_id=target_entity.id),
                        target_entity
                    ))

        return DecomposeStopEvent(sub_claims=new_triplets)