"""
Debug script to trace path processing in GraphCheck
"""
import asyncio
import time
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

from src.modules.datasets.exfever import ExFever
from src.modules.schema.graph_check.graph import Graph
from src.impls.workflows.graph_check.construct_graph import GraphConstructWorkflow
from src.impls.workflows.graph_check.infilling import InfillingWorkflow
from src.impls.workflows.graph_check.verification import VerificationWorkflow
from src.impls.events.graph_check.construct_graph import ConstructGraphStartEvent
from src.impls.events.graph_check.infilling import InfillingStartEvent
from src.impls.events.graph_check.verification import VerificationStartEvent

load_dotenv('.env')

EXFEVER_DATA_PATH = "datas/ex-fever/dev.csv"


async def debug_single_sample():
    """Debug a single sample to trace path processing."""
    # Load dataset
    dataset = ExFever.from_csv(EXFEVER_DATA_PATH)
    sample = dataset[0]
    claim = sample["claim"]
    label = sample["label"]

    print(f"=== Claim ===")
    print(f"{claim[:100]}...")
    print(f"Label: {label}")
    print()

    # Configure LLM
    llm = OpenAI(model="gpt-4.1-mini", max_retries=3, timeout=120.0)

    # Build retriever
    from src.modules.retrievers.exfever import build_exfever_milvus_retriever
    retriever = build_exfever_milvus_retriever(similarity_top_k=10)

    # Graph Construction
    print("=== Graph Construction ===")
    construct_wf = GraphConstructWorkflow(llm=llm)
    graph_result = await construct_wf.run(
        start_event=ConstructGraphStartEvent(claim=claim)
    )
    graph_obj = graph_result.graph
    print(f"Latent entities: {graph_obj.num_la_ent}")
    print(f"Total triples: {len(graph_obj.total_triples)}")
    print(f"Def triples: {len(graph_obj.def_triples)}")
    print(f"Regular triples: {len(graph_obj.triples)}")
    print()

    # Path Generation
    print("=== Path Generation ===")
    paths = graph_obj.get_valid_paths(5)
    print(f"Generated {len(paths)} paths:")
    for i, path in enumerate(paths):
        print(f"  Path {i+1}: {path}")
    print()

    if not paths:
        print("No paths generated, skipping infilling")
        return

    # Try each path
    path_results = []
    for path_idx, path in enumerate(paths):
        print(f"=== Processing Path {path_idx + 1}/{len(paths)}: {path} ===")

        try:
            # Infilling
            print("  Starting infilling...")
            infilling_start = time.time()
            infilling_wf = InfillingWorkflow(
                llm=llm,
                retriever=retriever,
                dataset_type="exfever"
            )

            infilling_result = await infilling_wf.run(
                start_event=InfillingStartEvent(claim=claim, path=path, graph=graph_obj)
            )
            infilling_time = time.time() - infilling_start

            print(f"  Infilling completed in {infilling_time:.2f}s")
            print(f"  Infilling result type: {type(infilling_result)}")
            print(f"  Has graph attribute: {hasattr(infilling_result, 'graph')}")

            if hasattr(infilling_result, 'graph'):
                infilled_graph = infilling_result.graph
                print(f"  Infilled graph triples: {len(infilled_graph.total_triples)}")
                print(f"  Infilled graph def triples: {len(infilled_graph.def_triples)}")
                print(f"  Infilled graph regular triples: {len(infilled_graph.triples)}")

                # Check first few triples
                print(f"  First 3 triples:")
                for i, triple in enumerate(infilled_graph.total_triples[:3]):
                    print(f"    {i+1}. sentence: {triple.sentence[:60] if triple.sentence else 'None'}...")
                    print(f"       triplet_text: {triple.triplet_text[:60] if triple.triplet_text else 'None'}...")
            else:
                print(f"  ERROR: Infilling result does not have graph attribute!")
                print(f"  Infilling result: {infilling_result}")
                continue

            # Verification
            print("  Starting verification...")
            verification_start = time.time()
            verification_wf = VerificationWorkflow(
                llm=llm,
                retriever=retriever,
                dataset_type="exfever"
            )

            result = await verification_wf.run(
                start_event=VerificationStartEvent(graph=infilled_graph)
            )
            verification_time = time.time() - verification_start

            print(f"  Verification completed in {verification_time:.2f}s")
            print(f"  Prediction: {result.prediction}")
            print(f"  Verification results count: {len(result.verification_results)}")

            # Store path result
            path_result = {
                "path_index": path_idx,
                "path": path,
                "prediction": result.prediction,
                "infilling_time": infilling_time,
                "verification_time": verification_time,
                "verification_results_count": len(result.verification_results)
            }
            path_results.append(path_result)

            print(f"  Path result appended. Total path results: {len(path_results)}")

            if result.prediction == "SUPPORTED":
                print(f"  Path succeeded! Breaking loop.")
                break

        except Exception as e:
            print(f"  ERROR processing path: {e}")
            import traceback
            traceback.print_exc()

    print()
    print(f"=== Summary ===")
    print(f"Total path results: {len(path_results)}")
    for pr in path_results:
        print(f"  Path {pr['path_index']}: {pr['prediction']}")


if __name__ == '__main__':
    asyncio.run(debug_single_sample())