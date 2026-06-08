import re
from copy import deepcopy

from workflows import Workflow, step, Context
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage

from .context import SynthesisContext
from .document_context import DocumentContextWorkflow, DocumentContextStartEvent
from ...events.synthesis.verify_v2 import (
    VerifyStartEvent,
    BuildDocumentContext,
    VerifyLoopInit,
    TryVerifyEvent,
    UnknownEntityResolutionLoop,
    ExtractPossibleDocumentTriplets,
    ExtractRelevantDocumentSentences,
    ExtractTripletsForUnknownEntity,
    InfillDocumentTriplets,
    MergeTriplets,
    RemapSubClaim,
    TryInfillFullUnknowns,
    VerifyAggregateEvent,
    VerifyStopEvent
)
from src.modules.prompts.synthesis.verify_v2 import (
    TRY_VERIFY_SYSTEM,
    TRY_VERIFY_USER,
    EXTRACT_DOC_TRIPLETS_SYSTEM,
    EXTRACT_DOC_TRIPLETS_USER,
    EXTRACT_SOURCE_SENTENCES_SYSTEM,
    EXTRACT_SOURCE_SENTENCES_USER,
    EXTRACT_ENTITY_SENTENCES_0_SYSTEM,
    EXTRACT_ENTITY_SENTENCES_0_USER,
    EXTRACT_ENTITY_SENTENCES_1_SYSTEM,
    EXTRACT_ENTITY_SENTENCES_1_SYSTEM_V2,
    EXTRACT_ENTITY_SENTENCES_1_USER,
    EXTRACT_TRIPLETS_FOR_UNK_SYSTEM,
    EXTRACT_TRIPLETS_FOR_UNK_USER,
    INFILL_TRIPLETS_SYSTEM,
    INFILL_TRIPLETS_USER,
    INFILL_EXTRACTED_TRIPLETS_SYSTEM,
    INFILL_EXTRACTED_TRIPLETS_USER,
    MERGE_TRIPLETS_SYSTEM,
    MERGE_TRIPLETS_USER,
    REMAP_SUB_CLAIM_SYSTEM,
    REMAP_SUB_CLAIM_USER,
    TRY_INFILL_FULL_UNKS_SYSTEM,
    TRY_INFILL_FULL_UNKS_USER,
    INFILL_FULL_UNKS_FROM_DOC_SYSTEM,
    INFILL_FULL_UNKS_FROM_DOC_USER,
    EXTRACT_RELEVANT_SENTENCES_SYSTEM,
    EXTRACT_RELEVANT_SENTENCES_USER,
)
from src.modules.datasets.base import LABELS
from src.modules.evaluator import char_sim

_CHAR_SIM_THRESHOLD = 0.8


def _fuzzy_matching_entity(e: str, entity_list, threshold: float = _CHAR_SIM_THRESHOLD) -> bool:
    """Check if entity e fuzzy-matches any entry in unknown_entities."""
    e_l = e.lower()
    for ent in entity_list:
        ent_l = ent.lower()
        if e_l in ent_l or ent_l in e_l:
            return True
        if char_sim(e_l, ent_l) >= threshold:
            return True
    return False


class VerifyWorkflow(Workflow):
    def __init__(self,
                 llm: LLM,
                 threshold: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.threshold = threshold
        self.document_ctx = DocumentContextWorkflow(llm=llm, **kwargs)

    @step
    async def initialize(
            self, ctx: Context[SynthesisContext], ev: VerifyStartEvent
    ) -> BuildDocumentContext:
        sub_claims = ev.sub_claims
        documents = ev.documents

        # init counter
        verify_counter = {}
        for sub_claim in sub_claims:
            verify_counter[sub_claim] = 0
        async with ctx.store.edit_state() as ctx_state:
            ctx_state.claim = ev.claim
            ctx_state.sub_claims = ev.sub_claims
            ctx_state.documents = ev.documents
            ctx_state.verify_ctx.remained_subclaims = deepcopy(sub_claims)
            ctx_state.verify_ctx.verify_counter = verify_counter

        return BuildDocumentContext(documents=documents, sub_claims=sub_claims)

    @step
    async def build_doc_ctx(
            self, ctx: Context[SynthesisContext], ev: BuildDocumentContext
    ) -> VerifyLoopInit:
        # Run in its own fresh context to avoid the is_running guard skipping the start event
        result = await self.document_ctx.run(
            start_event=DocumentContextStartEvent(
                sub_claims=ev.sub_claims,
                documents=ev.documents
            )
        )

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.document_ctx.triplets = result.triplets
            ctx_state.document_ctx.known_entities = result.known_entities
            ctx_state.document_ctx.unknown_entities = result.unknown_entities
            ctx_state.verify_ctx.remapped_triplets = result.triplets

        return VerifyLoopInit()

    @step
    async def verify_loop_init(
            self, ctx: Context[SynthesisContext], ev: VerifyLoopInit
    ) -> TryVerifyEvent | UnknownEntityResolutionLoop | VerifyAggregateEvent:
        verify_index = await ctx.store.get("verify_ctx.verify_index")
        sub_claims = await ctx.store.get("sub_claims")
        remained_subclaims = await ctx.store.get("verify_ctx.remained_subclaims")

        if not len(remained_subclaims):
            return VerifyAggregateEvent()

        if len(remained_subclaims) and verify_index >= len(sub_claims):
            return UnknownEntityResolutionLoop()

        current_sub_claim = sub_claims[verify_index]
        async with ctx.store.edit_state() as ctx_state:
            ctx_state.verify_ctx.current_sub_claim = current_sub_claim

        return TryVerifyEvent(sub_claim=current_sub_claim)

    @step
    async def try_verify(
            self, ctx: Context[SynthesisContext], ev: TryVerifyEvent
    ) -> VerifyLoopInit:
        original_sub_claim = await ctx.store.get("verify_ctx.current_sub_claim")
        sub_claim = ev.sub_claim
        doc_triplets = await ctx.store.get("verify_ctx.remapped_triplets")
        # TODO: Could add a Context retrieval step, keep simple for now

        triplet_str = "\n".join(doc_triplets)
        response = await self.llm.achat([
            ChatMessage(
                content=TRY_VERIFY_SYSTEM,
                role="system"
            ),
            ChatMessage(
                content=TRY_VERIFY_USER.format(
                    sub_claim=sub_claim,
                    triplets=triplet_str
                ),
                role="user"
            )
        ])
        content = response.message.content

        # Regex to extract Verdict
        match = re.search(r"verdict:\s*(true|false|not enough information)", content, re.IGNORECASE)
        verdict_str = match.group(1).lower() if match else "not enough information"

        verdict_map = {
            "true": LABELS[0],
            "false": LABELS[1],
            "not enough information": LABELS[2],
        }
        verdict = verdict_map.get(verdict_str, LABELS[2])

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.verify_ctx.verify_counter[original_sub_claim] += 1
            ctx_state.verify_ctx.verify_index += 1

        verify_counter = await ctx.store.get("verify_ctx.verify_counter")
        if verdict != LABELS[2] or verify_counter[original_sub_claim] >= self.threshold:
            async with ctx.store.edit_state() as ctx_state:
                ctx_state.verify_ctx.verify_mapping[original_sub_claim] = verdict
                ctx_state.verify_ctx.remained_subclaims.remove(original_sub_claim)

        return VerifyLoopInit()

    @step
    async def unknown_entity_resolution_init(
            self, ctx: Context[SynthesisContext], ev: UnknownEntityResolutionLoop
    ) -> ExtractPossibleDocumentTriplets | TryInfillFullUnknowns:
        unk_sub_claims = await ctx.store.get("verify_ctx.remained_subclaims")
        unk_entities = await ctx.store.get("document_ctx.unknown_entities")
        known_entities = await ctx.store.get("document_ctx.known_entities")

        claim_to_resolve = ""
        full_unknowns = True
        for sub_claim in unk_sub_claims:
            s, r, t = [sc.strip() for sc in sub_claim.split('->')]
            # Can only resolve if at least one entity is KNOWN
            if (s in known_entities and t in unk_entities
                    or s in unk_entities and t in known_entities):
                claim_to_resolve = sub_claim
                full_unknowns = False
                break

        # Resolve Entity with full UNKNOWN
        if not claim_to_resolve:
            claim_to_resolve = unk_sub_claims[0] # Just pick the first one to resolve

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.verify_ctx.current_sub_claim = claim_to_resolve

        if full_unknowns:
            return TryInfillFullUnknowns(sub_claim=claim_to_resolve)

        return ExtractPossibleDocumentTriplets(sub_claim=claim_to_resolve)

    @step
    async def extract_doc_triplets(
            self, ctx: Context[SynthesisContext], ev: ExtractPossibleDocumentTriplets
    ) -> ExtractRelevantDocumentSentences:
        sub_claim = ev.sub_claim
        doc_triplets = await ctx.store.get("verify_ctx.remapped_triplets")
        triplet_str = "\n".join(doc_triplets)

        # Ask LLM to select relevant triplets for this claim
        response = await self.llm.achat([
            ChatMessage(content=EXTRACT_DOC_TRIPLETS_SYSTEM, role="system"),
            ChatMessage(
                content=EXTRACT_DOC_TRIPLETS_USER.format(
                    sub_claim=sub_claim,
                    triplets=triplet_str,
                ),
                role="user",
            ),
        ])
        content = response.message.content
        extracted = content.strip()

        # Parse out the selected triplets (filter out "None" or empty responses)
        if extracted.lower() == "none" or not extracted:
            # If this happens it means the document context extracted triplets got clears along the way
            # Rerun with document_ctx triplets
            doc_triplets = await ctx.store.get("document_ctx.triplets")
            triplet_str = "\n".join(doc_triplets)
            # Ask LLM to select relevant triplets for this claim
            response = await self.llm.achat([
                ChatMessage(content=EXTRACT_DOC_TRIPLETS_SYSTEM, role="system"),
                ChatMessage(
                    content=EXTRACT_DOC_TRIPLETS_USER.format(
                        sub_claim=sub_claim,
                        triplets=triplet_str,
                    ),
                    role="user",
                ),
            ])
            content = response.message.content
            extracted = content.strip()

        relevant_triplets = [line.strip() for line in extracted.splitlines() if line.strip()]

        return ExtractRelevantDocumentSentences(triplets=relevant_triplets)

    @step
    async def extract_relevant_sentences(
            self, ctx: Context[SynthesisContext], ev: ExtractRelevantDocumentSentences
    ) -> ExtractTripletsForUnknownEntity:
        triplets = ev.triplets
        documents = await ctx.store.get("documents")

        document_str = "\n".join(documents)
        unknown_entities = await ctx.store.get("document_ctx.unknown_entities")
        known_entities = await ctx.store.get("document_ctx.known_entities")

        unk_triplet_sentences_mapping = {}

        for triplet in triplets:
            seen = []
            # Step 1: retrieve the direct source sentences that form the triplet
            response = await self.llm.achat([
                ChatMessage(content=EXTRACT_SOURCE_SENTENCES_SYSTEM, role="system"),
                ChatMessage(
                    content=EXTRACT_SOURCE_SENTENCES_USER.format(
                        triplet=triplet,
                        document=document_str,
                    ),
                    role="user",
                ),
            ])
            content = response.message.content
            source_sentences = [
                line.strip() for line in content.strip().splitlines()
                if line.strip()
            ]

            # Step 2: use the source sentences as anchors to find sentences that identify the unknown entity
            # 3 possibilities:
            # 1 unknown
            # 0 unknown
            # 2 unknowns

            # Classify possibility
            parts = [sc.strip() for sc in triplet.split('->')]
            s, r, t = parts
            unknowns = [e for e in (s, t) if _fuzzy_matching_entity(e, unknown_entities)]
            knowns = [e for e in (s, t) if _fuzzy_matching_entity(e, known_entities)]
            if not len(unknowns):
                unknowns = [e for e in (s, t) if e not in knowns]

            async with ctx.store.edit_state() as ctx_state:
                ctx_state.verify_ctx.unknown_entities.extend(unknowns)
                ctx_state.verify_ctx.known_entities.extend(knowns)

            # Handling 1 unknown
            if len(unknowns) == 1:
                system_prompt = EXTRACT_ENTITY_SENTENCES_1_SYSTEM_V2
                unk_entity = unknowns[0]
                known_entity = s if s not in unknowns else t
                user_prompt = EXTRACT_ENTITY_SENTENCES_1_USER.format(
                    known_entity=known_entity,
                    relation=r,
                    unk_entity=unk_entity,
                    source_sentences="\n".join(source_sentences),
                    document=document_str
                )
            # Handling 0 unknown
            elif len(unknowns) == 0:
                system_prompt = EXTRACT_ENTITY_SENTENCES_0_SYSTEM
                user_prompt = EXTRACT_ENTITY_SENTENCES_0_USER.format(
                    entity_1=s,
                    entity_2=t,
                    source_sentences="\n".join(source_sentences),
                    document=document_str
                )
            # Handling 2 unknowns
            elif len(unknowns) == 2:
                # TODO: Implement
                pass

            entity_response = await self.llm.achat([
                ChatMessage(content=system_prompt, role="system"),
                ChatMessage(
                    content=user_prompt,
                    role="user",
                ),
            ])
            entity_content = entity_response.message.content
            entity_sentences = [
                line.strip() for line in entity_content.strip().splitlines()
                if line.strip() and line.strip().lower() != "none"
            ]

            for sentence in source_sentences + entity_sentences:
                if sentence not in seen:
                    seen.append(sentence)
            unk_triplet_sentences_mapping[triplet] = seen

        # Extract relevant sentences for the sub claim
        sub_claim = await ctx.store.get("verify_ctx.current_sub_claim")
        response = await self.llm.achat([
            ChatMessage(content=EXTRACT_RELEVANT_SENTENCES_SYSTEM, role="system"),
            ChatMessage(
                content=EXTRACT_RELEVANT_SENTENCES_USER.format(
                    sub_claim=sub_claim,
                    documents=document_str,
                ),
                role="user",
            ),
        ])
        content = response.message.content.strip()
        if content.lower() == "none" or not content:
            pass
        else:
            relevant_sentences = [line.strip() for line in content.splitlines() if line.strip()]
            unk_triplet_sentences_mapping[sub_claim] = relevant_sentences

        return ExtractTripletsForUnknownEntity(triplets_sentences_mapping=unk_triplet_sentences_mapping)

    @step
    async def extract_triplets_for_unk(
            self, ctx: Context[SynthesisContext], ev: ExtractTripletsForUnknownEntity
    ) -> InfillDocumentTriplets:
        unk_triplet_sentences_mapping = ev.triplets_sentences_mapping
        unknown_entities = await ctx.store.get("verify_ctx.unknown_entities")
        doc_unk_entities = await ctx.store.get("document_ctx.unknown_entities")

        triplets = []
        for triplet, sentences in unk_triplet_sentences_mapping.items():
            s, r, t = [part.strip() for part in triplet.split('->')]
            unk_entity = [
                e for e in (s, t) if e in unknown_entities or e in doc_unk_entities
            ]

            sentences_str = "\n".join(sentences)
            response = await self.llm.achat([
                ChatMessage(content=EXTRACT_TRIPLETS_FOR_UNK_SYSTEM, role="system"),
                ChatMessage(
                    content=EXTRACT_TRIPLETS_FOR_UNK_USER.format(
                        unk_entity=unk_entity,
                        sentences=sentences_str,
                    ),
                    role="user",
                ),
            ])
            content = response.message.content.strip()

            # TODO: Add error handling when fails to extract triplets
            extracted_triplets = [
                line.strip() for line in content.splitlines() if line.strip()
            ]
            triplets.extend(extracted_triplets)

        return InfillDocumentTriplets(
            triplets=triplets,
            triplets_sentences_mapping=unk_triplet_sentences_mapping
        )

    @step
    async def infill_triplets(
            self, ctx: Context[SynthesisContext], ev: InfillDocumentTriplets,
    ) -> MergeTriplets:
        extracted_triplets = ev.triplets
        triplets_sentences_mapping = ev.triplets_sentences_mapping
        remapped_triplets = await ctx.store.get("verify_ctx.remapped_triplets")

        infilled_original = []

        for original_triplet, sentences in triplets_sentences_mapping.items():
            sentences_str = "\n".join(sentences)
            extracted_triplets_str = "\n".join(extracted_triplets)

            # Step 1: Infill the current original triplet
            response = await self.llm.achat([
                ChatMessage(content=INFILL_TRIPLETS_SYSTEM, role="system"),
                ChatMessage(
                    content=INFILL_TRIPLETS_USER.format(
                        original_triplet=original_triplet,
                        extracted_triplets=extracted_triplets_str,
                        sentences=sentences_str,
                    ),
                    role="user",
                ),
            ])
            infilled_triplet = response.message.content.strip()
            if infilled_triplet.lower() != "none" and infilled_triplet:
                infilled_original.append(infilled_triplet)

            # Step 2: Infill the entire list of extracted triplets at once
            ext_response = await self.llm.achat([
                ChatMessage(content=INFILL_EXTRACTED_TRIPLETS_SYSTEM, role="system"),
                ChatMessage(
                    content=INFILL_EXTRACTED_TRIPLETS_USER.format(
                        infilled_triplet=infilled_triplet,
                        extracted_triplets=extracted_triplets_str,
                        sentences=sentences_str,
                    ),
                    role="user",
                ),
            ])
            ext_content = ext_response.message.content.strip()
            extracted_triplets = [line.strip() for line in ext_content.splitlines() if line.strip()]

        new_triplets = infilled_original + extracted_triplets + remapped_triplets
        return MergeTriplets(triplets=new_triplets)

    @step
    async def merge_triplets(
            self, ctx: Context[SynthesisContext], ev: MergeTriplets,
    ) -> RemapSubClaim:
        triplets = ev.triplets
        sub_claim = await ctx.store.get("verify_ctx.current_sub_claim")

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
            ctx_state.verify_ctx.remapped_triplets = merged_triplets

        return RemapSubClaim(sub_claim=sub_claim)

    @step
    async def remap_sub_claim(
            self, ctx: Context[SynthesisContext], ev: RemapSubClaim,
    ) -> TryVerifyEvent:
        sub_claim = ev.sub_claim
        triplets = await ctx.store.get("verify_ctx.remapped_triplets")
        unknown_entities = await ctx.store.get("document_ctx.unknown_entities")

        s, r, t = [part.strip() for part in sub_claim.split('->')]
        unk_entity = [e for e in (s, t) if e in unknown_entities]

        triplet_str = "\n".join(triplets)
        response = await self.llm.achat([
            ChatMessage(content=REMAP_SUB_CLAIM_SYSTEM, role="system"),
            ChatMessage(
                content=REMAP_SUB_CLAIM_USER.format(
                    sub_claim=sub_claim,
                    unk_entity=unk_entity,
                    triplets=triplet_str,
                ),
                role="user",
            ),
        ])
        remapped = response.message.content.strip()

        # Fall back to original if response is malformed
        parts = [p.strip() for p in remapped.split('->')]
        if len(parts) != 3:
            remapped = sub_claim

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.verify_ctx.remapped_sub_claims[sub_claim] = remapped

        return TryVerifyEvent(sub_claim=remapped)

    @step
    async def try_infill_full_unks(
            self, ctx: Context[SynthesisContext], ev: TryInfillFullUnknowns,
    ) -> TryVerifyEvent | ExtractPossibleDocumentTriplets:
        sub_claim = ev.sub_claim
        remapped_sub_claims: dict = await ctx.store.get("verify_ctx.remapped_sub_claims")

        # Format resolved claims as numbered "original => resolved" lines for the prompt
        remapped_lines = "\n".join(
            f"{i + 1}. {original}  =>  {resolved}"
            for i, (original, resolved) in enumerate(remapped_sub_claims.items())
        )

        response = await self.llm.achat([
            ChatMessage(content=TRY_INFILL_FULL_UNKS_SYSTEM, role="system"),
            ChatMessage(
                content=TRY_INFILL_FULL_UNKS_USER.format(
                    sub_claim=sub_claim,
                    remapped_sub_claims=remapped_lines,
                ),
                role="user",
            ),
        ])
        infilled = response.message.content.strip()

        # Fall back to original if response is malformed
        parts = [p.strip() for p in infilled.split('->')]
        if len(parts) != 3:
            infilled = sub_claim
            parts = [p.strip() for p in infilled.split('->')]

        # 3 possibilities
        # 1. Resolve 0 unknown
        # 2. Resolve 1 unknown
        # 3. Resolve 2 unknowns

        # Detect possibilities
        s, _, t = parts
        unknown_entities = await ctx.store.get("verify_ctx.unknown_entities")
        unknown_remains = [e for e in (s, t) if e in unknown_entities]

        # Resolve 1 unknown -> Back to normal 1 unknown loop
        if len(unknown_remains) == 1:
            return ExtractPossibleDocumentTriplets(sub_claim=infilled)

        # Resolve 2 unknowns -> Just verify at this point
        if not len(unknown_remains):
            return TryVerifyEvent(sub_claim=infilled)

        # Resolve 0 unknown -> Resolve using Document triplets
        triplets = await ctx.store.get("verify_ctx.remapped_triplets")
        triplet_str = "\n".join(triplets)

        doc_response = await self.llm.achat([
            ChatMessage(content=INFILL_FULL_UNKS_FROM_DOC_SYSTEM, role="system"),
            ChatMessage(
                content=INFILL_FULL_UNKS_FROM_DOC_USER.format(
                    sub_claim=sub_claim,
                    triplets=triplet_str,
                ),
                role="user",
            ),
        ])
        infilled = doc_response.message.content.strip()

        # Fall back to original if response is malformed
        if len([p for p in infilled.split('->')]) != 3:
            infilled = sub_claim

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.verify_ctx.remapped_sub_claims[sub_claim] = infilled

        return TryVerifyEvent(sub_claim=infilled)

    @step
    async def aggregate(
            self, ctx: Context[SynthesisContext], ev: VerifyAggregateEvent,
    ) -> VerifyStopEvent:
        sub_claim_mapping = await ctx.store.get("verify_ctx.verify_mapping")
        for sub_claim, result in sub_claim_mapping.items():
            if result == "REFUTE":
                return VerifyStopEvent(result="REFUTE")
            elif result == "NEI": # TODO: Need more investigation
                return VerifyStopEvent(result="NEI")

        return VerifyStopEvent(result="SUPPORT")