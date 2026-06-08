KNOWN_ENTITY_EXTRACTION_SYSTEM = """Read the claim carefully and extract all the KNOWN entities in the claim
KNOWN entities are entities that has explicit naming
Please only rely on the claim to extract entity, do NOT use internal knowledge or things that are not explicitly mention in the claim
# Response format
Return a numbered list of KNOWN entities.
# Example
Claim: Albert Einstein was born in 1879 is a German and won the 1921 Nobel Prize in Physics
Output:
1. Albert Einstein
2. 1879
3. German
4. 1921 Nobel Prize in Physics
"""

KNOWN_ENTITY_EXTRACTION_USER = "Claim: {claim}"

KNOWN_ENTITY_RANKING_SYSTEM = """You are provided with a claim and extracted KNOWN entities
Your job is to rank each entity, an entity should be higher rank if it's connected to many UNKNOWN entities
# Response Format
```
1. [ENT1] - its connection to UNKNOWN entities 
2. [ENT2] - its connection to UNKNOWN entities
# Final Ranking
A numbered list of KNOWN entities ranked
```
"""

KNOWN_ENTITY_RANKING_USER = "Claim: {claim}\nEntities:\n{entities}"

ENTITY_RELATION_EXTRACTION_SYSTEM = """You are provided with a claim and an entity
Your job is to extract *1-hop*, *atomic* relation for that entity
Please only rely on the claim to extract entity, do NOT use internal knowledge or things that are not explicitly mention in the claim
# Rule for Relation extraction
- Relation MUST be single, simple verbs or verb phrases or prepositions
- Compound relation is NOT allowed
- If a relation contains a noun, it MUST be split into:
    - A structural relation (e.g. "has")
    - A type relation (e.g "is")
- The source entity and target entity MUST be NOUN
# Response Format
[Ent] -> [Relation] -> [Ent]
# Example 1
Claim: There was a physicist born in Germany, Theory of Relativity and Quantum theory was developed by him, who received the 1921 Nobel Prize in Physics
Entity: Germany
Relation:
Physicist -> born in -> Germany
# Example 2
Claim: There was a physicist born in Germany, Theory of Relativity and Quantum Theory was developed by him, who received the 1921 Nobel Prize in Physics
Entity: Theory of Relativity
Relation:
Theory of Relativity -> developed by -> Physicist
# Example 3
Claim: In Germany had the owner of 1921 Nobel Prize in Physics developed Theory of Relativity
Entity: 1921 Nobel Prize in Physics
Relation:
A person -> received -> 1921 Nobel Prize in Physics
"""

ENTITY_RELATION_EXTRACTION_SYSTEM_V2 = """You are provided with a claim and an entity
Your task is to extract structured knowledge graph triplets for that entity
Please only rely on the claim to extract entity, do NOT use internal knowledge or things that are not explicitly mention in the claim
# Response Format
[Entity] -> [Relation] -> [Entity]
# Rule for Relation
- Relation MUST be single, simple verbs or verb phrases or prepositions
- Compound relation is NOT allowed
- If a relation contains a noun, decompose it into two triplets using intermediate node:
    - A structural relation (e.g. "has")
    - A type relation (e.g "is")
    - Intermediate node MUST have unique identifier
    - Do not reuse node names for different entities.
    - Example: A -> has -> uncle_1, uncle_1 -> is -> B
- The source entity and target entity MUST be NOUN
- ONLY extract *1-hop*, *atomic* relation
# Rule for Entity
- Be careful of Capitalization Entity, e.g. Einstein is different from einstein
"""

ENTITY_RELATION_EXTRACTION_USER = "Claim: {claim}\nEntity: {entity}"

ENTITY_GLEANING_SYSTEM = """You are provided with 3 inputs:
- A claim
- List of entities
- List of existing triplets
Your job is to extract ALL *1-hop* relation for list of provided entities that is DIFFERENT from existing triplets
# Rule for Relation extraction
- Relation MUST be single, simple verbs or verb phrases or prepositions
- Compound relation is NOT allowed
- If a relation contains a noun, decompose it into two triplets using intermediate node:
    - A structural relation (e.g. "has")
    - A type relation (e.g "is")
    - Intermediate node MUST have unique identifier
    - Do not reuse node names for different entities.
    - Example: A -> has -> uncle_1, uncle_1 -> is -> B
- The source entity and target entity MUST be NOUN
# Response Format
[Entity] -> [Relation] -> [Entity]
# Example 1
Claim: The physicist, who developed Theory of Relativity, he was awarded with 1921 Nobel Prize and was born in Germany
Entities: 
1. Person
2. Physicist
Existing Triplets:
Physicist -> developed -> Theory of Relativity
Person -> awarded -> 1921 Nobel Prize
Output:
Physicist -> is -> Person_A
Person_A -> born in -> Germany
# Example 2
Claim: An inventor, who developed light bulb, is also a businessman in America where Phonograph is developed
Entities:
1. Inventor
2. America
Existing Triplets:
Inventor -> developed -> light bulb
Phonograph -> developed in -> America
Output:
Inventor -> is -> businessman
America -> has -> businessman
"""

ENTITY_GLEANING_USER = "Claim: {claim}\nEntities:\n{entities}\nTriplets:\n{triplets}"

MERGE_ENTITY_SYSTEM = """You are given a claim and list of knowledge graph triplets
Your task is to merge, normalize and connect these triplets into a coherent knowledge graph that represent the claim
# Instruction
1. Understand the claim
- Identify the main entity (or entities) the claim is about
- Determine the relationships described in the claim
2. Normalize entities
- Merge entities that refer to the same concept
- Replace generic entities with specific entities from the claim when appropriate
3. Output the final Graph
- Provide a cleaned list of triplets in the format: `source -> relation -> target`
- Double check each triplet to ensure proper formatting
# Rule for entity normalizing
- Each triplet MUST be 1-hop relation
- Relation MUST be single, simple verbs or verb phrases or prepositions
- Compound relation is NOT allowed
- If a relation contains a noun, it MUST be split into:
    - A structural relation (e.g. "has")
    - A type relation (e.g "is")
- The source entity and target entity must be NOUN
# Output format
```
<cleaned triplet 1>
<cleaned triplet 2>
<cleaned triplet 3>
```
"""

MERGE_ENTITY_USER = "Claim: {claim}\nTriplets:\n{triplets}"

VERIFY_ENTITY_SYSTEM = """You are an expert fact decomposition verifier.
You will be given an original claim and a numbered list of sub-claims extracted from it.
Each sub-claim is a triplet: Subject -> Relation -> Object.

Verify whether each sub-claim faithfully and correctly represents the original claim.
Pay close attention to relation direction — a reversed triplet is WRONG.

For each sub-claim provide:
- correct: true/false
- reason: short explanation
- correct_sub_claim: null if correct, or the fixed triplet if incorrect

Example:
  Claim: "Alice is the mother of Bob."
  CORRECT: Alice -> mother of -> Bob  →  correct=true, correct_sub_claim=null
  WRONG:   Bob -> mother of -> Alice  →  correct=false, correct_sub_claim="Alice -> mother of -> Bob"\
"""

VERIFY_ENTITY_USER = """Original claim:
{claim}

Sub-claims to verify:
{triplets}
"""