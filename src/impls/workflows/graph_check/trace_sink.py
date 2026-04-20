"""Optional stdout trace for GraphCheck (single-claim runs); replaces DEBUG logging for visibility."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from .debug_utils import preview_text


def _sep(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


@runtime_checkable
class GraphCheckTraceSink(Protocol):
    """Hooks for structured GraphCheck tracing (print or no-op)."""

    def construct_raw_llm(self, raw: str) -> None: ...

    def construct_parsed(
        self,
        definition_triples: list[str],
        triples: list[str],
    ) -> None: ...

    def graph_latent_and_paths(
        self,
        num_latent: int,
        latent_order: list[str],
        path_limit: int,
        paths: list[list[str]],
    ) -> None: ...

    def path_only(self, path_index: int, path: list[str]) -> None: ...

    def infill_retrieval_and_query(
        self,
        path_index: int,
        step_1based: int,
        num_steps: int,
        latent_entity: str,
        retrieval_query: str,
        doc_previews: list[str],
        infilling_query_full: str,
    ) -> None: ...

    def infill_llm_answer(
        self,
        path_index: int,
        ent_index: int,
        latent_entity: str,
        answer: str,
    ) -> None: ...

    def verify_block(
        self,
        path_index: int,
        triple_1based: int,
        num_triples: int,
        subclaim: str,
        doc_previews: list[str],
        llm_raw: str,
        prediction: str,
    ) -> None: ...

    def path_prediction(self, path_index: int, prediction: str) -> None: ...


class NullGraphCheckTrace:
    """No-op trace (benchmark / default)."""

    def construct_raw_llm(self, raw: str) -> None:
        pass

    def construct_parsed(
        self,
        definition_triples: list[str],
        triples: list[str],
    ) -> None:
        pass

    def graph_latent_and_paths(
        self,
        num_latent: int,
        latent_order: list[str],
        path_limit: int,
        paths: list[list[str]],
    ) -> None:
        pass

    def path_only(self, path_index: int, path: list[str]) -> None:
        pass

    def infill_retrieval_and_query(
        self,
        path_index: int,
        step_1based: int,
        num_steps: int,
        latent_entity: str,
        retrieval_query: str,
        doc_previews: list[str],
        infilling_query_full: str,
    ) -> None:
        pass

    def infill_llm_answer(
        self,
        path_index: int,
        ent_index: int,
        latent_entity: str,
        answer: str,
    ) -> None:
        pass

    def verify_block(
        self,
        path_index: int,
        triple_1based: int,
        num_triples: int,
        subclaim: str,
        doc_previews: list[str],
        llm_raw: str,
        prediction: str,
    ) -> None:
        pass

    def path_prediction(self, path_index: int, prediction: str) -> None:
        pass


def trace_uses_logger_fallback(trace: GraphCheckTraceSink) -> bool:
    """True when stdout structured trace is off; workflows emit logger.debug instead."""
    return isinstance(trace, NullGraphCheckTrace)


class StdoutGraphCheckTrace:
    """Human-readable sections for single-claim debug output (print, not logging)."""

    def construct_raw_llm(self, raw: str) -> None:
        _sep("CONSTRUCT — full LLM output (raw)")
        print(raw.rstrip())
        print()

    def construct_parsed(
        self,
        definition_triples: list[str],
        triples: list[str],
    ) -> None:
        _sep("CONSTRUCT — parse_graph result (input to Graph)")
        print("definition_triples:")
        for line in definition_triples:
            print(f"  {line}")
        print("triples:")
        for line in triples:
            print(f"  {line}")
        print()

    def graph_latent_and_paths(
        self,
        num_latent: int,
        latent_order: list[str],
        path_limit: int,
        paths: list[list[str]],
    ) -> None:
        _sep("GRAPH — latent entities and paths (LLM + parse_graph)")
        print(f"Latent entity count: {num_latent}")
        print(f"Latent order: {latent_order!r}")
        print()
        n_paths = len(paths)
        print(f"Generated {n_paths} path(s) (path_limit={path_limit}):")
        for i, p in enumerate(paths):
            arrow = " -> ".join(p)
            print(f"  [{i}] {arrow}")
        print()

    def path_only(self, path_index: int, path: list[str]) -> None:
        arrow = " -> ".join(path)
        _sep(f"PROCESS PATH ONLY [{path_index}]: {arrow}")
        print()

    def infill_retrieval_and_query(
        self,
        path_index: int,
        step_1based: int,
        num_steps: int,
        latent_entity: str,
        retrieval_query: str,
        doc_previews: list[str],
        infilling_query_full: str,
    ) -> None:
        _sep(
            f"INFILL — path {path_index} — step {step_1based}/{num_steps} — entity {latent_entity}"
        )
        print("Retrieval query:")
        print(f"  {retrieval_query}")
        print()
        print("[Retrieval] Top 10 documents (~100 chars each):")
        print()
        for i, prev in enumerate(doc_previews, start=1):
            print(f"   {i}. {prev}")
        print()
        print("Infilling query (full):")
        print(infilling_query_full)
        print()

    def infill_llm_answer(
        self,
        path_index: int,
        ent_index: int,
        latent_entity: str,
        answer: str,
    ) -> None:
        label = f"path{path_index}_ent{ent_index}_{latent_entity} (infill prompt -> answer)"
        _sep(f"LLM OUTPUT [{label}] — full")
        print(answer.rstrip() if answer else "")
        print()

    def verify_block(
        self,
        path_index: int,
        triple_1based: int,
        num_triples: int,
        subclaim: str,
        doc_previews: list[str],
        llm_raw: str,
        prediction: str,
    ) -> None:
        _sep(f"VERIFY — path {path_index} — triple {triple_1based}/{num_triples}")
        print("Subclaim:")
        print(f"  {subclaim}")
        print()
        print("[Retrieval (verify)] Top 10 documents (~100 chars each):")
        print()
        for i, prev in enumerate(doc_previews, start=1):
            print(f"   {i}. {prev}")
        print()
        label = f"path{path_index}_triple{triple_1based - 1} (verify prompt -> answer)"
        _sep(f"LLM OUTPUT [{label}] — full")
        print(llm_raw.rstrip() if llm_raw else "")
        print()
        mapped = (
            "SUPPORTED"
            if prediction == "SUPPORTED"
            else ("NOT_SUPPORTED" if prediction == "NOT_SUPPORTED" else prediction)
        )
        print(f"=> Subclaim prediction: {mapped}")
        print()

    def path_prediction(self, path_index: int, prediction: str) -> None:
        _sep("SELECTED PATH RESULT")
        print(f"path_prediction: {prediction}")
        print()


def nodes_to_previews(nodes: Iterable[object], preview_len: int = 100) -> list[str]:
    """Top-k document one-line previews for trace output."""
    out: list[str] = []
    for node in nodes:
        text = getattr(node, "text", "") or ""
        out.append(preview_text(text, preview_len))
    return out
