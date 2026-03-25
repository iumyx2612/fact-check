import logging
from typing import Optional

from workflows import Workflow, step, Context
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage
from llama_index.core.retrievers import BaseRetriever

from src.modules.graph_check.graph import Graph
from src.modules.prompts.graph_check.verify import (
    VERIFY_TRIPLE_USER,
    VERIFY_TRIPLE_WITH_CONTEXT_USER,
    GRAPH_CHECK_VERIFY_NO_EVIDENCE,
    GRAPH_CHECK_VERIFY_WITH_EVIDENCE,
    GRAPH_CHECK_VERIFY_WITH_GOLD_AND_RETRIEVED,
)
from src.impls.events.graph_check.verification import (
    VerificationStartEvent,
    VerificationLoopInitialize,
    VerifyTripleEvent,
    VerificationStopEvent
)
from src.impls.events.graph_check.context import SynthesisContext
from .debug_utils import log_retrieved_nodes
from .trace_sink import (
    GraphCheckTraceSink,
    NullGraphCheckTrace,
    nodes_to_previews,
    trace_uses_logger_fallback,
)

logger = logging.getLogger(__name__)

# Match ``tests/original_graphcheck.TracedGraphCheck`` / ``OpenAIBaseModel.verify``:
# retrieved text capped at 4000 chars, then prompt uses up to 3000 chars.
GRAPH_CHECK_RETRIEVAL_EVIDENCE_CAP = 4000
GRAPH_CHECK_PROMPT_EVIDENCE_CAP = 3000


async def _llm_verify_chat_raw_static(llm: LLM, prompt: ChatMessage) -> str:
    """Run verification LLM call; on failure return empty string like ``OpenAIBaseModel.generate``."""
    try:
        response = await llm.achat([prompt])
        return (response.message.content or "").strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return ""


def parse_graphcheck_binary_answer(raw_out: str) -> str:
    """Match ``tests/original_graphcheck.OpenAIBaseModel.verify`` boolean parsing."""
    answer = raw_out.lower().strip(" .")
    if answer in ("true", "yes", "supported"):
        return "SUPPORTED"
    if answer in ("false", "no", "not supported"):
        return "NOT_SUPPORTED"
    if "not " in answer or "false" in answer:
        return "NOT_SUPPORTED"
    return "NOT_SUPPORTED"


def build_exfever_retriever(db_path: str, similarity_top_k: int = 10) -> BaseRetriever:
    """Build an ExFever retriever; db_path is kept for compatibility."""
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
                   classification_mode: str = "binary",
                   dataset_type: Optional[str] = None,
                   trace: GraphCheckTraceSink | None = None,
                   **kwargs):
        """
        Initialize verification workflow.

        Args:
            llm: Language model for verification
            retriever: Pre-built retriever (optional)
            db_path: Path to ExFever wiki_db (used if retriever not provided)
            similarity_top_k: Number of documents to retrieve
            document_level: Verification mode - "concat", "each", or "concat+each"
            classification_mode: Classification mode - "three_way" or "binary"
            dataset_type: Dataset type (unused, kept for compatibility)
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.document_level = document_level
        self.classification_mode = classification_mode

        if retriever is not None:
            self.retriever = retriever
        elif db_path is not None:
            self.retriever = build_exfever_retriever(db_path, similarity_top_k)
        else:
            raise ValueError("Either retriever or db_path must be provided")
        self.trace: GraphCheckTraceSink = trace or NullGraphCheckTrace()
        self._trace_uses_logger_fallback = trace_uses_logger_fallback(self.trace)

    async def _llm_verify_chat_raw(self, prompt: ChatMessage) -> str:
        return await _llm_verify_chat_raw_static(self.llm, prompt)

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
            ctx_state.path_index_for_trace = ev.path_index

        return VerificationLoopInitialize()

    @step
    async def loop_init(
            self, ctx: Context[SynthesisContext], ev: VerificationLoopInitialize
    ) -> VerifyTripleEvent | VerificationStopEvent:
        """Initialize verification loop and send first triple or stop if no triples.
        
        Matches original GraphCheck.verify_graph: simple for-loop iteration
        with natural break on NOT_SUPPORTED.
        """
        graph = await ctx.store.get("graph")
        verification_index = await ctx.store.get("verification_index")

        # Match original: simple sequential iteration through all triples
        if verification_index < len(graph.total_triples):
            triple = graph.total_triples[verification_index]
            # Defensive: handle None sentence (fallback to triplet_text if needed)
            triple_text = triple.sentence if triple.sentence else triple.triplet_text
            return VerifyTripleEvent(triple_text=triple_text)

        # All triples processed - return final result
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
        """Verify a single triple against retrieved evidence."""
        graph: Graph = await ctx.store.get("graph")
        verification_index = await ctx.store.get("verification_index")
        verification_results = await ctx.store.get("verification_results")
        current_prediction = await ctx.store.get("prediction")

        triple_text = ev.triple_text

        # Retrieve evidence
        nodes = self.retriever.retrieve(triple_text)

        # Verify based on document_level setting
        if self.document_level == "concat":
            prediction, verification_details, llm_raw = await self._verify_concat(triple_text, nodes)
        elif self.document_level == "each":
            prediction, verification_details, llm_raw = await self._verify_each(triple_text, nodes)
        elif self.document_level == "concat+each":
            prediction, verification_details, llm_raw = await self._verify_concat_each(triple_text, nodes)
        else:
            raise ValueError(f"Unknown document_level: {self.document_level}")

        # Record result
        result = {
            "subclaim": triple_text,
            "prediction": prediction,
        }
        verification_results.append(result)

        path_idx = int(await ctx.store.get("path_index_for_trace"))
        doc_previews = (
            [] if self._trace_uses_logger_fallback else nodes_to_previews(nodes)
        )
        self.trace.verify_block(
            path_idx,
            verification_index + 1,
            len(graph.total_triples),
            triple_text,
            doc_previews,
            llm_raw,
            prediction,
        )

        # Match original: if NOT_SUPPORTED, break immediately
        if prediction == "NOT_SUPPORTED":
            current_prediction = "NOT_SUPPORTED"
            async with ctx.store.edit_state() as ctx_state:
                ctx_state.verification_results = verification_results
                ctx_state.prediction = current_prediction
                ctx_state.verification_index = len(graph.total_triples)
            
            return VerificationStopEvent(
                prediction=current_prediction,
                verification_results=verification_results
            )

        # Advance to next triple
        async with ctx.store.edit_state() as ctx_state:
            ctx_state.verification_results = verification_results
            ctx_state.prediction = current_prediction
            ctx_state.verification_index = verification_index + 1

        return VerificationLoopInitialize()

    async def _verify_with_llm(
            self,
            claim: str,
            evidence: str,
            gold_evidence: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Use LLM to verify a claim against evidence.

        Returns:
            Tuple of (prediction "SUPPORTED"|"NOT_SUPPORTED", raw model text).
        """
        # Prepare prompt based on classification mode
        if self.classification_mode == "binary":
            if gold_evidence:
                ev_for_prompt = (
                    evidence[:GRAPH_CHECK_PROMPT_EVIDENCE_CAP]
                    if evidence
                    else ""
                )
                gold_trim = gold_evidence[:GRAPH_CHECK_PROMPT_EVIDENCE_CAP]
                prompt_content = GRAPH_CHECK_VERIFY_WITH_GOLD_AND_RETRIEVED.format(
                    claim=claim,
                    gold_evidence=gold_trim,
                    retrieved_evidence=ev_for_prompt,
                )
            elif evidence:
                # Match ``OpenAIBaseModel.verify``: use ``if not evidence`` (truthiness), not strip()
                ev_for_prompt = evidence[:GRAPH_CHECK_PROMPT_EVIDENCE_CAP]
                prompt_content = GRAPH_CHECK_VERIFY_WITH_EVIDENCE.format(
                    claim=claim,
                    evidence=ev_for_prompt,
                )
            else:
                prompt_content = GRAPH_CHECK_VERIFY_NO_EVIDENCE.format(claim=claim)
        else:  # three_way — align evidence length with binary graphcheck (max 3000 in prompt)
            ev_prompt = evidence[:GRAPH_CHECK_PROMPT_EVIDENCE_CAP] if evidence else ""
            if gold_evidence:
                gold_prompt = gold_evidence[:GRAPH_CHECK_PROMPT_EVIDENCE_CAP]
                prompt_content = VERIFY_TRIPLE_WITH_CONTEXT_USER.format(
                    claim=claim,
                    gold_evidence=gold_prompt,
                    retrieved_evidence=ev_prompt,
                )
            else:
                prompt_content = VERIFY_TRIPLE_USER.format(
                    claim=claim,
                    evidence=ev_prompt,
                )

        prompt = ChatMessage(content=prompt_content, role="user")
        raw_out = await self._llm_verify_chat_raw(prompt)

        # Parse the answer based on classification mode
        if self.classification_mode == "binary":
            return parse_graphcheck_binary_answer(raw_out), raw_out
        answer_upper = raw_out.upper()
        if "SUPPORT" in answer_upper and "NOT ENOUGH INFORMATION" not in answer_upper:
            return "SUPPORTED", raw_out
        if "REFUTE" in answer_upper:
            return "NOT_SUPPORTED", raw_out
        return "NOT_SUPPORTED", raw_out

    async def _verify_concat(
            self,
            claim: str,
            nodes: list
    ) -> tuple[str, dict]:
        """
        Verify claim with concatenated evidence.

        Args:
            claim: The claim to verify
            nodes: Retrieved document nodes

        Returns:
            Tuple of (prediction, verification_details)
        """
        # Same join/cap as ``TracedGraphCheck.retrieve_evidence`` (newline + 4000 chars)
        combined_evidence = "\n".join([node.text for node in nodes])
        if len(combined_evidence) > GRAPH_CHECK_RETRIEVAL_EVIDENCE_CAP:
            combined_evidence = combined_evidence[:GRAPH_CHECK_RETRIEVAL_EVIDENCE_CAP]

        # Call verification model
        prediction, llm_raw = await self._verify_with_llm(claim, combined_evidence)

        verification_details = {
            "method": "concatenated",
            "num_documents": len(nodes),
            "evidence_preview": combined_evidence[:200] if combined_evidence else ""
        }

        return prediction, verification_details, llm_raw

    async def _verify_each(
            self,
            claim: str,
            nodes: list
    ) -> tuple[str, dict]:
        """
        Verify claim with each evidence separately.

        Args:
            claim: The claim to verify
            nodes: Retrieved document nodes

        Returns:
            Tuple of (prediction, verification_details, concatenated raw LLM outputs)
        """
        individual_results = []
        raws: list[str] = []

        for node in nodes:
            evidence = node.text[:GRAPH_CHECK_RETRIEVAL_EVIDENCE_CAP]

            # Call verification model
            prediction, raw_one = await self._verify_with_llm(claim, evidence)
            raws.append(raw_one)

            individual_results.append({
                "evidence_preview": evidence[:100] + "...",
                "prediction": prediction
            })

        # Aggregate results (majority vote)
        aggregated = self._aggregate_results(individual_results)

        verification_details = {
            "method": "individual",
            "num_documents": len(nodes),
            "individual_results": individual_results,
            "aggregated": aggregated
        }

        return aggregated["prediction"], verification_details, "\n---\n".join(raws)

    async def _verify_concat_each(
            self,
            claim: str,
            nodes: list
    ) -> tuple[str, dict]:
        """
        Verify with concat first, fallback to each if needed.

        Args:
            claim: The claim to verify
            nodes: Retrieved document nodes

        Returns:
            Tuple of (prediction, verification_details, raw LLM text for trace)
        """
        # Try concat mode first
        concat_prediction, concat_details, concat_raw = await self._verify_concat(claim, nodes)

        # Fallback logic: if concat result is uncertain, try each mode
        if self._is_uncertain(concat_prediction):
            logger.debug(f"Concat result uncertain ({concat_prediction}), falling back to each mode")
            each_prediction, each_details, each_raw = await self._verify_each(claim, nodes)

            verification_details = {
                "method": "concat+each (fallback)",
                "concat_prediction": concat_prediction,
                "each_prediction": each_prediction,
                "final_method": "each"
            }

            return each_prediction, verification_details, f"concat:\n{concat_raw}\n---\neach:\n{each_raw}"

        verification_details = {
            "method": "concat+each (concat succeeded)",
            "final_method": "concat"
        }

        return concat_prediction, verification_details, concat_raw

    def _is_uncertain(self, _prediction: str) -> bool:
        """
        Whether concat mode should fall back to per-document verification.

        Treating NOT_SUPPORTED as "uncertain" was wrong: it ran ``each`` mode (one LLM
        call per retrieved document) on top of concat, exploding API usage and diverging
        from GraphCheck's single-shot verify per subclaim.
        """
        return False

    def _aggregate_results(self, individual_results: list[dict]) -> dict:
        """
        Aggregate individual verification results using majority vote.

        Args:
            individual_results: List of individual verification results

        Returns:
            Dictionary with aggregated prediction and details
        """
        # Count labels
        label_counts = {}
        for r in individual_results:
            label = r["prediction"]
            label_counts[label] = label_counts.get(label, 0) + 1

        # Return majority label
        if not label_counts:
            return {"prediction": "NOT_SUPPORTED", "confidence": 0.0, "label_counts": {}}

        majority_label = max(label_counts, key=label_counts.get)
        confidence = label_counts[majority_label] / len(individual_results)

        return {
            "prediction": majority_label,
            "confidence": confidence,
            "label_counts": label_counts
        }


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
        
        # Retrieve evidence (same cap as graph path)
        nodes = self.retriever.retrieve(combined_claim)
        evidence = "\n".join([node.text for node in nodes])
        if len(evidence) > GRAPH_CHECK_RETRIEVAL_EVIDENCE_CAP:
            evidence = evidence[:GRAPH_CHECK_RETRIEVAL_EVIDENCE_CAP]

        if evidence:
            prompt_content = GRAPH_CHECK_VERIFY_WITH_EVIDENCE.format(
                claim=combined_claim,
                evidence=evidence[:GRAPH_CHECK_PROMPT_EVIDENCE_CAP],
            )
        else:
            prompt_content = GRAPH_CHECK_VERIFY_NO_EVIDENCE.format(claim=combined_claim)

        prompt = ChatMessage(content=prompt_content, role="user")
        raw_out = await _llm_verify_chat_raw_static(self.llm, prompt)
        prediction = parse_graphcheck_binary_answer(raw_out)
        
        verification_results = [{
            "combined_claim": combined_claim,
            "prediction": prediction,
            "evidence": evidence[:500]
        }]
        
        return VerificationStopEvent(
            prediction=prediction,
            verification_results=verification_results
        )
