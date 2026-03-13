"""
Test to verify the infilling graph reconstruction bug
"""
import re

def simulate_infilling_bug():
    """Simulate the infilling workflow to demonstrate the bug."""

    # Initial definition triplets (with latent entities)
    def_triple_sents = [
        "(ENT1) [SEP] is [SEP] an American comedian",
        "(ENT2) [SEP] is [SEP] the founder of Apatow Productions"
    ]

    # Regular triplets
    triple_sents = [
        "Pineapple Express [SEP] is [SEP] a 2008 American stoner comedy film",
        "Pineapple Express [SEP] was produced by [SEP] (ENT1)",
        "(ENT1) [SEP] is part of [SEP] (ENT2)"
    ]

    print("=== Initial State ===")
    print(f"Definition triplets: {def_triple_sents}")
    print(f"Regular triplets: {triple_sents}")
    print()

    # Simulate filling ENT1 with "Judd Apatow"
    print("=== Filling ENT1 with 'Judd Apatow' ===")
    current_latent_entity = "(ENT1)"
    answer = "Judd Apatow"

    # Replace in all triplets
    infilled_def_triplets_texts = [
        text.replace(current_latent_entity, answer) for text in def_triple_sents
    ]
    infilled_triplets_texts = [
        text.replace(current_latent_entity, answer) for text in triple_sents
    ]

    print(f"After replacement:")
    print(f"Definition triplets: {infilled_def_triplets_texts}")
    print(f"Regular triplets: {infilled_triplets_texts}")
    print()

    # Filter to keep only definition triplets with latent entities (BUGGY LOGIC)
    remained_def_triplet_texts = [
        text for text in infilled_def_triplets_texts if re.search(r"\(ENT\d+\)", text.split()[0])
    ]

    print(f"Remained definition triplets (with latent entities): {remained_def_triplet_texts}")
    print()

    # BUGGY: Only reconstruct graph if there are remaining definition triplets
    if remained_def_triplet_texts:
        print("BUGGY: Reconstructing graph with remained_def_triplet_texts")
        new_def_triples = remained_def_triplet_texts
        new_triples = infilled_triplets_texts
    else:
        print("BUGGY: NOT reconstructing graph (no remained_def_triplet_texts)")
        print("Graph stays in old state!")
        new_def_triples = def_triple_sents  # OLD STATE
        new_triples = triple_sents  # OLD STATE

    print(f"Graph definition triplets: {new_def_triples}")
    print(f"Graph regular triplets: {new_triples}")
    print()

    # Simulate filling ENT2 with "Apatow Productions"
    print("=== Filling ENT2 with 'Apatow Productions' ===")
    current_latent_entity = "(ENT2)"
    answer = "Apatow Productions"

    # Replace in all triplets (using the buggy graph state)
    infilled_def_triplets_texts = [
        text.replace(current_latent_entity, answer) for text in new_def_triples
    ]
    infilled_triplets_texts = [
        text.replace(current_latent_entity, answer) for text in new_triples
    ]

    print(f"After replacement:")
    print(f"Definition triplets: {infilled_def_triplets_texts}")
    print(f"Regular triplets: {infilled_triplets_texts}")
    print()

    # Filter to keep only definition triplets with latent entities
    remained_def_triplet_texts = [
        text for text in infilled_def_triplets_texts if re.search(r"\(ENT\d+\)", text.split()[0])
    ]

    print(f"Remained definition triplets (with latent entities): {remained_def_triplet_texts}")
    print()

    # BUGGY: Only reconstruct graph if there are remaining definition triplets
    if remained_def_triplet_texts:
        print("BUGGY: Reconstructing graph with remained_def_triplet_texts")
        final_def_triples = remained_def_triplet_texts
        final_triples = infilled_triplets_texts
    else:
        print("BUGGY: NOT reconstructing graph (no remained_def_triplet_texts)")
        print("Graph stays in old state!")
        final_def_triples = new_def_triples  # OLD STATE
        final_triples = new_triples  # OLD STATE

    print(f"Final graph definition triplets: {final_def_triples}")
    print(f"Final graph regular triplets: {final_triples}")
    print()

    print("=== EXPECTED vs ACTUAL ===")
    print("Expected (all entities filled):")
    expected_def = [
        "Judd Apatow [SEP] is [SEP] an American comedian",
        "Apatow Productions [SEP] is [SEP] the founder of Apatow Productions"
    ]
    expected_regular = [
        "Pineapple Express [SEP] is [SEP] a 2008 American stoner comedy film",
        "Pineapple Express [SEP] was produced by [SEP] Judd Apatow",
        "Judd Apatow [SEP] is part of [SEP] Apatow Productions"
    ]
    print(f"Definition triplets: {expected_def}")
    print(f"Regular triplets: {expected_regular}")
    print()

    print("Actual (buggy behavior):")
    print(f"Definition triplets: {final_def_triples}")
    print(f"Regular triplets: {final_triples}")
    print()

    # Check if ENT2 is still latent in the final graph
    has_ent2 = any("(ENT2)" in t for t in final_triples)
    print(f"BUG CONFIRMED: ENT2 still latent in final graph: {has_ent2}")


def simulate_fix():
    """Simulate the fix for the infilling workflow."""

    print("\n" + "="*80)
    print("=== SIMULATING FIX ===")
    print("="*80 + "\n")

    # Initial definition triplets (with latent entities)
    def_triple_sents = [
        "(ENT1) [SEP] is [SEP] an American comedian",
        "(ENT2) [SEP] is [SEP] the founder of Apatow Productions"
    ]

    # Regular triplets
    triple_sents = [
        "Pineapple Express [SEP] is [SEP] a 2008 American stoner comedy film",
        "Pineapple Express [SEP] was produced by [SEP] (ENT1)",
        "(ENT1) [SEP] is part of [SEP] (ENT2)"
    ]

    print("=== Initial State ===")
    print(f"Definition triplets: {def_triple_sents}")
    print(f"Regular triplets: {triple_sents}")
    print()

    # Simulate filling ENT1 with "Judd Apatow"
    print("=== Filling ENT1 with 'Judd Apatow' ===")
    current_latent_entity = "(ENT1)"
    answer = "Judd Apatow"

    # Replace in all triplets
    infilled_def_triplets_texts = [
        text.replace(current_latent_entity, answer) for text in def_triple_sents
    ]
    infilled_triplets_texts = [
        text.replace(current_latent_entity, answer) for text in triple_sents
    ]

    print(f"After replacement:")
    print(f"Definition triplets: {infilled_def_triplets_texts}")
    print(f"Regular triplets: {infilled_triplets_texts}")
    print()

    # FIX: Always reconstruct graph with all updated triplets
    print("FIX: Reconstructing graph with all updated triplets")
    new_def_triples = infilled_def_triplets_texts
    new_triples = infilled_triplets_texts

    print(f"Graph definition triplets: {new_def_triples}")
    print(f"Graph regular triplets: {new_triples}")
    print()

    # Simulate filling ENT2 with "Apatow Productions"
    print("=== Filling ENT2 with 'Apatow Productions' ===")
    current_latent_entity = "(ENT2)"
    answer = "Apatow Productions"

    # Replace in all triplets (using the fixed graph state)
    infilled_def_triplets_texts = [
        text.replace(current_latent_entity, answer) for text in new_def_triples
    ]
    infilled_triplets_texts = [
        text.replace(current_latent_entity, answer) for text in new_triples
    ]

    print(f"After replacement:")
    print(f"Definition triplets: {infilled_def_triplets_texts}")
    print(f"Regular triplets: {infilled_triplets_texts}")
    print()

    # FIX: Always reconstruct graph with all updated triplets
    print("FIX: Reconstructing graph with all updated triplets")
    final_def_triples = infilled_def_triplets_texts
    final_triples = infilled_triplets_texts

    print(f"Final graph definition triplets: {final_def_triples}")
    print(f"Final graph regular triplets: {final_triples}")
    print()

    print("=== EXPECTED vs ACTUAL ===")
    expected_def = [
        "Judd Apatow [SEP] is [SEP] an American comedian",
        "Apatow Productions [SEP] is [SEP] the founder of Apatow Productions"
    ]
    expected_regular = [
        "Pineapple Express [SEP] is [SEP] a 2008 American stoner comedy film",
        "Pineapple Express [SEP] was produced by [SEP] Judd Apatow",
        "Judd Apatow [SEP] is part of [SEP] Apatow Productions"
    ]
    print("Expected:")
    print(f"Definition triplets: {expected_def}")
    print(f"Regular triplets: {expected_regular}")
    print()

    print("Actual (fixed behavior):")
    print(f"Definition triplets: {final_def_triples}")
    print(f"Regular triplets: {final_triples}")
    print()

    # Check if all entities are filled
    has_ent1 = any("(ENT1)" in t for t in final_triples)
    has_ent2 = any("(ENT2)" in t for t in final_triples)
    print(f"FIX VERIFIED: ENT1 still latent: {has_ent1}, ENT2 still latent: {has_ent2}")
    print(f"All entities filled: {not (has_ent1 or has_ent2)}")


if __name__ == '__main__':
    simulate_infilling_bug()
    simulate_fix()