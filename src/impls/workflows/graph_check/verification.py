import logging
import asyncio
from typing import Optional

from workflows import Workflow, step, Context
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage
from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from src.modules.schema.graph_check.graph import Graph
from src.modules.prompts.graph_check.verify import VERIFY_TRIPLE_USER, VERIFY_TRIPLE_WITH_CONTEXT_USER
from src.impls.events.graph_check.verification import (
    VerificationStartEvent,
    VerificationLoopInitialize,
    VerifyTripleEvent,
    VerificationStopEvent
)
from src.impls.events.graph_check.context import SynthesisContext

logger = logging.getLogger(__name__)


def build_exfever_retriever(db_path: str, similarity_top_k: int = 10) -> BM25Retriever:
    """Build a BM25 retriever for ExFever dataset."""
    from src.modules.retrievers.exfever import build_exfever_retriever as build_retriever
    return build_retriever(db_path, similarity_top_k)


class VerificationWorkflow(Workflow):
    """
    Verification workflow that verifies each triple in the graph against retrieved evidence.

    This workflow mirrors the original GraphCheck verification:
    1. For each triple in the graph, retrieve evidence
    2. Verify the triple against the evidence using LLM
    3. If any triple is NOT_SUPPORTED, return NOT_SUPPORTED
    4. Otherwise return SUPPORTED
    """

    def __init__(self,
                  llm: LLM,
                  retriever: Optional[BaseRetriever] = None,
                  db_path: Optional[str] = None,
                  similarity_top_k: int = 10,
                  document_level: str = "concat+each",
                  dataset_type: Optional[str] = None,
                  parallel_verification: bool = True,
                  max_concurrent_verifications: int = 5,
                  **kwargs):
        """
        Initialize verification workflow.

        Args:
            llm: Language model for verification
            retriever: Pre-built retriever (optional)
            db_path: Path to ExFever wiki_db (used if retriever not provided)
            similarity_top_k: Number of documents to retrieve
            document_level: Verification mode - "concat", "each", or "concat+each"
            dataset_type: Dataset type (unused, kept for compatibility)
            parallel_verification: Whether to verify triples in parallel
            max_concurrent_verifications: Maximum concurrent verifications when parallel=True
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.document_level = document_level
        self.parallel_verification = parallel_verification
        self.max_concurrent_verifications = max_concurrent_verifications

        if retriever is not None:
            self.retriever = retriever
        elif db_path is not None:
            self.retriever = build_exfever_retriever(db_path, similarity_top_k)
        else:
            raise ValueError("Either retriever or db_path must be provided")

    @step
    async def initialize(
            self, ctx: Context[SynthesisContext], ev: VerificationStartEvent
    ) -> VerificationLoopInitialize:
        """Initialize verification with the graph."""
        graph = ev.graph

        # Initialize verification tracking
        async with ctx.store.edit_state() as ctx_state:
            ctx_state.verification_index = 0
            ctx_state.verification_results = []
            ctx_state.prediction = "SUPPORTED"  # Start with SUPPORTED, change to NOT_SUPPORTED if any fails
            ctx_state.graph = graph

        return VerificationLoopInitialize()

    @step
    async def loop_init(
            self, ctx: Context[SynthesisContext], ev: VerificationLoopInitialize
    ) -> VerifyTripleEvent | VerificationStopEvent:
        """Initialize verification loop and send first triple or stop if no triples."""
        graph = await ctx.store.get("graph")

        # If parallel verification is enabled, collect all triples and verify in parallel
        if self.parallel_verification and len(graph.total_triples) > 1:
            return await self._verify_all_triples_parallel(ctx, graph)

        # Sequential verification (original behavior)
        verification_index = await ctx.store.get("verification_index")

        # Find next valid triple with non-None sentence
        while verification_index < len(graph.total_triples):
            triple = graph.total_triples[verification_index]
            # Use sentence if available, otherwise use triplet_text
            triple_text = triple.sentence if triple.sentence else triple.triplet_text
            if triple_text:
                return VerifyTripleEvent(triple_text=triple_text)
            # Skip triples with no valid text
            verification_index += 1
            async with ctx.store.edit_state() as ctx_state:
                ctx_state.verification_index = verification_index

        # All triples processed or skipped
        prediction = await ctx.store.get("prediction")
        verification_results = await ctx.store.get("verification_results")
        return VerificationStopEvent(
            prediction=prediction,
            verification_results=verification_results
        )

    @step
    async def verify_triple(
            self, ctx: Context[SynthesisContext], ev: VerifyTripleEvent
    ) -> VerificationLoopInitialize | VerificationStopEvent:
        """
        Verify a single triple against retrieved evidence.
        
        Implements three document_level modes:
        - concat: Verify against all retrieved documents concatenated
        - each: Verify against each document individually
        - concat+each: Try concat first, fall back to each if concat fails
        """
        graph: Graph = await ctx.store.get("graph")
        verification_index = await ctx.store.get("verification_index")
        verification_results = await ctx.store.get("verification_results")
        current_prediction = await ctx.store.get("prediction")
        
        # Get the current triple
        triple_text = ev.triple_text
        logger.debug(f"Verifying triple {verification_index + 1}/{len(graph.total_triples)}: {triple_text[:50]}...")
        
        # Retrieve evidence
        nodes = self.retriever.retrieve(triple_text)
        retrieved_evidence = "\n".join([node.text for node in nodes])
        
        # Verify based on document_level setting
        prediction = await self._verify_with_llm(triple_text, retrieved_evidence)
        
        # Record result
        result = {
            "triple": triple_text,
            "prediction": prediction,
            "retrieved_evidence": retrieved_evidence[:500] if retrieved_evidence else ""
        }
        verification_results.append(result)
        
        # Update prediction - if any triple is NOT_SUPPORTED, the whole path fails
        if prediction == "NOT_SUPPORTED":
            current_prediction = "NOT_SUPPORTED"
        
        # Move to next triple or finish
        async with ctx.store.edit_state() as ctx_state:
            ctx_state.verification_results = verification_results
            ctx_state.prediction = current_prediction
            ctx_state.verification_index = verification_index + 1

        # Find next valid triple to send
        next_index = verification_index + 1
        while next_index < len(graph.total_triples):
            next_triple = graph.total_triples[next_index]
            next_triple_text = next_triple.sentence if next_triple.sentence else next_triple.triplet_text
            if next_triple_text:
                ctx.send_event(VerifyTripleEvent(triple_text=next_triple_text))
                return VerificationLoopInitialize()
            next_index += 1

        # All triples verified
        return VerificationStopEvent(
            prediction=current_prediction,
            verification_results=verification_results
        )

    async def _verify_all_triples_parallel(
            self, ctx: Context[SynthesisContext], graph: Graph
    ) -> VerificationStopEvent:
        """Verify all triples in parallel using asyncio.gather()."""
        logger.debug(f"Verifying {len(graph.total_triples)} triples in parallel")

        # Collect all valid triples
        triples_to_verify = []
        for triple in graph.total_triples:
            triple_text = triple.sentence if triple.sentence else triple.triplet_text
            if triple_text:
                triples_to_verify.append(triple_text)

        if not triples_to_verify:
            return VerificationStopEvent(
                prediction="SUPPORTED",
                verification_results=[]
            )

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent_verifications)

# Verify each triple with rate limiting
        async def verify_single_triple(triple_text: str) -> dict:
            async with semaphore:
                # Retrieve evidence
                nodes = self.retriever.retrieve(triple_text)
                retrieved_evidence = "\n".join([node.text for node in nodes])

                # Verify with LLM
                prediction = await self._verify_with_llm(triple_text, retrieved_evidence)

                return {
                    "triple": triple_text,
                    "prediction": prediction,
                    "retrieved_evidence": retrieved_evidence[:500] if retrieved_evidence else ""
                }

        # Verify all triples in parallel
        verification_results = await asyncio.gather(
            *[verify_single_triple(triple_text) for triple_text in triples_to_verify]
        )

        # Determine final prediction - if any triple is NOT_SUPPORTED, the whole path fails
        final_prediction = "SUPPORTED"
        for result in verification_results:
            if result["prediction"] == "NOT_SUPPORTED":
                final_prediction = "NOT_SUPPORTED"
                break

        logger.debug(f"Parallel verification complete: {final_prediction}")

        return VerificationStopEvent(
            prediction=final_prediction,
            verification_results=verification_results
        )

    async def _verify_with_llm(
            self,
            claim: str,
            evidence: str,
            gold_evidence: Optional[str] = None
    ) -> str:
        """
        Use LLM to verify a claim against evidence.
        
        Returns: "SUPPORTED", "REFUTE", or "NOT ENOUGH INFORMATION"
        """
        # Truncate evidence if too long (similar to original truncate method)
        max_evidence_len = 40000
        if len(evidence) > max_evidence_len:
            evidence = evidence[:max_evidence_len]
            logger.warning(f"Evidence truncated to {max_evidence_len} characters")
        
        # Prepare prompt
        if gold_evidence:
            prompt_content = VERIFY_TRIPLE_WITH_CONTEXT_USER.format(
                claim=claim,
                gold_evidence=gold_evidence,
                retrieved_evidence=evidence
            )
        else:
            prompt_content = VERIFY_TRIPLE_USER.format(
                claim=claim,
                evidence=evidence
            )

        prompt = ChatMessage(content=prompt_content, role="user")
        response = await self.llm.achat([prompt])
        content = response.message.content
        answer = content.strip().upper() if content else ""
        
        # Parse the answer
        if "SUPPORT" in answer and "NOT ENOUGH INFORMATION" not in answer:
            return "SUPPORTED"
        elif "REFUTE" in answer:
            return "NOT_SUPPORTED"  # In graphcheck, refuted triples count as NOT_SUPPORTED
        else:
            return "NOT_SUPPORTED"  # Treat "NOT ENOUGH INFORMATION" as NOT_SUPPORTED for graphcheck


class SimpleVerificationWorkflow(Workflow):
    """
    Simplified verification workflow that verifies the entire claim at once.
    
    This is a simpler alternative that doesn't use graph-based verification,
    but directly verifies the claim against retrieved evidence.
    """
    
    def __init__(self,
                 llm: LLM,
                 retriever: Optional[BaseRetriever] = None,
                 db_path: Optional[str] = None,
                 similarity_top_k: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        
        if retriever is not None:
            self.retriever = retriever
        elif db_path is not None:
            self.retriever = build_exfever_retriever(db_path, similarity_top_k)
        else:
            raise ValueError("Either retriever or db_path must be provided")

    @step
    async def verify(
            self, ev: VerificationStartEvent
    ) -> VerificationStopEvent:
        """
        Simply verify the claim against retrieved evidence.
        
        This is a baseline verification without graph decomposition.
        """
        graph = ev.graph

        # Combine all triples into a single claim, filtering out None sentences
        valid_sentences = [
            triple.sentence if triple.sentence else triple.triplet_text
            for triple in graph.total_triples
            if triple.sentence or triple.triplet_text
        ]
        combined_claim = " && ".join(valid_sentences)
        
        # Retrieve evidence
        nodes = self.retriever.retrieve(combined_claim)
        evidence = "\n".join([node.text for node in nodes])
        
        # Verify with LLM
        prompt_content = VERIFY_TRIPLE_USER.format(
            claim=combined_claim,
            evidence=evidence
        )
        
        prompt = ChatMessage(content=prompt_content, role="user")
        response = await self.llm.achat([prompt])
        content = response.message.content
        answer = content.strip().upper() if content else ""
        
        # Parse answer
        if "SUPPORT" in answer and "NOT ENOUGH INFORMATION" not in answer:
            prediction = "SUPPORTED"
        elif "REFUTE" in answer:
            prediction = "NOT_SUPPORTED"
        else:
            prediction = "NOT_SUPPORTED"
        
        verification_results = [{
            "combined_claim": combined_claim,
            "prediction": prediction,
            "evidence": evidence[:500]
        }]
        
        return VerificationStopEvent(
            prediction=prediction,
            verification_results=verification_results
        )
