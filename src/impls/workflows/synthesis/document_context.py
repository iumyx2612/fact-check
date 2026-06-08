import re

from workflows import Workflow, step, Context
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage
from llama_index.core.output_parsers.utils import parse_json_markdown

from .context import SynthesisContext
from ...events.synthesis.document_context import (
    DocumentContextStartEvent,
    ExtractKnownEntitiesFromClaim,
    KnownEntityGraphInit,
    ExtractSentencesFromKnownEntity,
    ReplaceEntityIntoSentences,
    ExtractTripletFromSentences,
    MergeKnownTriplets,
    RemoveDuplicatedKnownTriplets,
    DocumentContextStopEvent
)
from src.modules.prompts.synthesis.document_context import (
    KNOWN_ENTITY_EXTRACTION_SYSTEM,
    KNOWN_ENTITY_EXTRACTION_USER,
    EXTRACT_KNOWN_SENTENCES_SYSTEM,
    EXTRACT_KNOWN_SENTENCES_USER,
    REPLACE_COREFERENCE_SYSTEM,
    REPLACE_COREFERENCE_USER,
    EXTRACT_KNOWN_TRIPLET_SYSTEM,
    EXTRACT_KNOWN_TRIPLET_USER,
    MERGE_TRIPLETS_SYSTEM,
    MERGE_TRIPLETS_USER
)


class DocumentContextWorkflow(Workflow):
    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def initialize(
            self, ctx: Context[SynthesisContext], ev: DocumentContextStartEvent
    ) -> ExtractKnownEntitiesFromClaim:
        sub_claims = ev.sub_claims
        documents = ev.documents

        if not isinstance(sub_claims[0], str):
            sub_claims = [
                f"{sub_claim[0].id} -> {sub_claim[1].id} -> {sub_claim[2].id}"
                for sub_claim in sub_claims]

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.sub_claims = sub_claims
            ctx_state.documents = documents

        return ExtractKnownEntitiesFromClaim(sub_claims=sub_claims)

    @step
    async def extract_known_entities(
            self, ctx: Context[SynthesisContext], ev: ExtractKnownEntitiesFromClaim
    ) -> KnownEntityGraphInit:
        sub_claims = ev.sub_claims

        response = await self.llm.achat([
            ChatMessage(
                content=KNOWN_ENTITY_EXTRACTION_SYSTEM,
                role="system"
            ),
            ChatMessage(
                content=KNOWN_ENTITY_EXTRACTION_USER.format(
                    triplets="\n".join(sub_claims)
                ),
                role="user"
            )
        ])

        content = response.message.content
        entities_data = parse_json_markdown(content)

        # TODO: Need to add validation that entity is exactly the same string as in the subclaim
        known_entities = [
            item["entity"]
            for item in entities_data
            if item.get("type", "").lower() == "specific"
        ]

        unknown_entities = [
            item["entity"]
            for item in entities_data
            if item.get("type", "").lower() == "generic"
        ]

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.document_ctx.known_entities = known_entities
            ctx_state.document_ctx.unknown_entities = unknown_entities

        return KnownEntityGraphInit()

    @step
    async def known_entity_graph_init(
            self, ctx: Context[SynthesisContext], ev: KnownEntityGraphInit
    ) -> ExtractSentencesFromKnownEntity | RemoveDuplicatedKnownTriplets:
        known_entities = await ctx.store.get("document_ctx.known_entities")
        index = await ctx.store.get("document_ctx.known_entity_index")

        if index >= len(known_entities):
            return RemoveDuplicatedKnownTriplets()

        return ExtractSentencesFromKnownEntity(entity=known_entities[index])

    @step
    async def extract_known_sentences(
            self, ctx: Context[SynthesisContext], ev: ExtractSentencesFromKnownEntity
    ) -> ReplaceEntityIntoSentences | KnownEntityGraphInit:
        entity = ev.entity
        documents = await ctx.store.get("documents")
        documents_str = "\n".join(documents)

        response = await self.llm.achat([
            ChatMessage(
                content=EXTRACT_KNOWN_SENTENCES_SYSTEM,
                role="system"
            ),
            ChatMessage(
                content=EXTRACT_KNOWN_SENTENCES_USER.format(
                    documents=documents_str,
                    entity=entity
                ),
                role="user"
            )
        ])
        content = response.message.content

        if not content:
            async with ctx.store.edit_state() as ctx_state:
                ctx_state.document_ctx.known_entity_index += 1
                return KnownEntityGraphInit()

        sentences = [s for s in content.splitlines() if s.strip()]

        return ReplaceEntityIntoSentences(entity=entity, sentences=sentences)

    @step
    async def replace_entities(
            self, ctx: Context[SynthesisContext], ev: ReplaceEntityIntoSentences
    ) -> ExtractTripletFromSentences:
        entity = ev.entity
        sentences = ev.sentences

        documents = await ctx.store.get("documents")
        all_sentences = []
        for doc in documents:
            all_sentences.extend([s.strip() for s in re.split(r'(?<=[.!?])\s+', doc) if s.strip()])

        results = []
        for sentence in sentences:
            has_coreference = entity not in sentence

            if has_coreference:
                context = []
                try:
                    idx = all_sentences.index(sentence)
                    prev_sentence = all_sentences[idx - 1] if idx > 0 else ""
                    next_sentence = all_sentences[idx + 1] if idx < len(all_sentences) - 1 else ""
                    context.append(" ".join(filter(None, [prev_sentence, sentence, next_sentence])))
                except ValueError:
                    context.append(sentence)

                context_str = "\n".join(context)

                response = await self.llm.achat([
                    ChatMessage(content=REPLACE_COREFERENCE_SYSTEM, role="system"),
                    ChatMessage(
                        content=REPLACE_COREFERENCE_USER.format(
                            entity=entity,
                            sentence=sentence,
                            context=context_str
                        ),
                        role="user"
                    )
                ])
                results.append(response.message.content.strip())
            else:
                results.append(sentence)

        return ExtractTripletFromSentences(entity=entity, sentences=results)

    @step
    async def extract_known_triplet(
            self, ctx: Context[SynthesisContext], ev: ExtractTripletFromSentences
    ) -> MergeKnownTriplets:
        entity = ev.entity
        sentences = ev.sentences

        triplets = []
        for sentence in sentences:
            response = await self.llm.achat([
                ChatMessage(
                    content=EXTRACT_KNOWN_TRIPLET_SYSTEM,
                    role="system"
                ),
                ChatMessage(
                    content=EXTRACT_KNOWN_TRIPLET_USER.format(
                        documents=sentence,
                        entity=entity
                    ),
                    role="user"
                )
            ])
            content = response.message.content
            for line in content.splitlines():
                parts = [p.strip() for p in line.split("->")]
                if len(parts) == 3 and line not in triplets:
                    triplets.append(line)

        return MergeKnownTriplets(triplets=triplets)

    @step
    async def merge_known_triplets(
            self, ctx: Context[SynthesisContext], ev: MergeKnownTriplets,
    ) -> KnownEntityGraphInit:
        triplets = ev.triplets

        if len(triplets) == 1: # Nothing to merge
            async with ctx.store.edit_state() as ctx_state:
                ctx_state.document_ctx.triplets.extend(triplets)
                ctx_state.document_ctx.known_entity_index += 1
            return KnownEntityGraphInit()

        triplet_str = "\n".join(triplets)
        merged_triplets = []
        response = await self.llm.achat([
            ChatMessage(
                content=MERGE_TRIPLETS_SYSTEM,
                role="system"
            ),
            ChatMessage(
                content=MERGE_TRIPLETS_USER.format(
                    triplets=triplet_str
                ),
                role="user"
            )
        ])

        content = response.message.content
        for line in content.splitlines():
            parts = [p.strip() for p in line.split("->")]
            if len(parts) == 3:
                merged_triplets.append(line)

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.document_ctx.triplets.extend(merged_triplets)
            ctx_state.document_ctx.known_entity_index += 1

        return KnownEntityGraphInit()

    @step
    async def remove_known_duplicated(
            self, ctx: Context[SynthesisContext], ev: RemoveDuplicatedKnownTriplets
    ) -> DocumentContextStopEvent:
        triplets = await ctx.store.get("document_ctx.triplets")
        known_entities = await ctx.store.get("document_ctx.known_entities")
        unknown_entities = await ctx.store.get("document_ctx.unknown_entities")

        seen = set()
        unique_triplets = []
        for triplet in triplets:
            normalized = " -> ".join(p.strip().lower() for p in triplet.split("->"))
            if normalized not in seen:
                seen.add(normalized)
                unique_triplets.append(triplet.strip())

        return DocumentContextStopEvent(
            triplets=unique_triplets,
            known_entities=known_entities,
            unknown_entities=unknown_entities,
        )
