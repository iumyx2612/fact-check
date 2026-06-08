KNOWN_ENTITY_EXTRACTION_SYSTEM = """You are an information extraction assistant. Your task is to classify entities found in a list of triplets as either **specific** or **generic**.
# Definitions
- **Specific Entity**: A real-world entity with a proper name that uniquely identifies it — such as a named person, organization, place, or titled work.
- **Generic Entity**: A concept, category, type, or anonymous placeholder that does not carry a real-world proper name. This includes:
  - Type labels
  - Numbered placeholder identifiers — these are unnamed instances and are generic even if they are unique nodes in the graph
# Instruction
1. Extract all unique entities from the subject and object positions of the given triplets. Ignore the relations.
2. For each entity, classify it as **Specific** or **Generic**.
3. Provide a brief justification for each classification.
# Output Format
Please response using the following format:
[
  {
    "entity": [entity name],
    "justification": [justification for entity type],
    "type": ["generic" or "specific"]
  }
]
"""

KNOWN_ENTITY_EXTRACTION_USER = "Triplets:\n{triplets}"

EXTRACT_KNOWN_SENTENCES_SYSTEM = """You are given a document and a target entity.
Extract every sentence that provides information or mentions that entity.                                   

# Instruction 
## Step 1 - Identify all references to the entity:
- Exact mentions: the entity name, its plural, possessive, or abbreviated forms
- Coreference: pronouns (it, its, they, their) or definite noun phrases ("the person", "the film", "the show", "the company", etc.) that follow a mention of the entity and refer back to it
## Step 2 - Extract sentences:
- Return every sentence that contains any reference identified in Step 1
- Preserve the original wording and punctuation exactly
- List each sentence on a new line
- Return only the sentences, no explanations or labels

# Example:
Example 1:
Entity: Einstein
Document: Einstein was born in Germany. He received the 1921 Nobel Prize in Physics. Anatole France received the 1921 Nobel Prize in Literature
Output:
Einstein was born in Germany.
He received the 1921 Nobel Prize in Physics.
Example 2:
Entity: 1921
Document: Einstein was born in Germany. He received the 1921 Nobel Prize in Physics. Anatole France received the 1921 Nobel Prize in Literature
Output:
He received the 1921 Nobel Prize in Physics.
Anatole France received the 1921 Nobel Prize in Literature
"""

EXTRACT_KNOWN_SENTENCES_USER = "Document:\n{documents}\nTarget phrase: {entity}"

REPLACE_COREFERENCE_SYSTEM = """You are given a sentence, its context and a target entity.
For the sentence, if it refers to the entity only via a coreference expression (a pronoun or definite noun phrase such as "he", "it", "the series", "the film", etc.) rather than naming the entity directly, annotate the coreference by appending the entity name in parentheses immediately after the coreference expression.
Use the context to resolve what the coreference refers to.
Return the sentence with no labels or explanations.

# Example
Entity: Einstein
Sentence: He received the 1921 Nobel Prize in Physics.
Context:
Einstein was born in Germany. He received the 1921 Nobel Prize in Physics.
Output:
He (Einstein) received the 1921 Nobel Prize in Physics.
"""

REPLACE_COREFERENCE_USER = "Entity: {entity}\nSentence: {sentence}\nContext:\n{context}"

EXTRACT_KNOWN_TRIPLET_SYSTEM = """You are given a short document and an entity.
Your task is to extract structured knowledge graph triplets for that entity
# Response format
[Entity] -> [Relation] -> [Entity]
Return only the final triplets, one per line, no explanations.
# Rule for Relation
- Relation MUST be single, simple verbs or verb phrases or prepositions
- Compound relation is NOT allowed
- If a relation contains a noun, decompose it into two triplets using intermediate node:
    - A structural relation (e.g. "has")
    - A type relation (e.g "is")
    - Intermediate node MUST have unique identifier
    - Do not reuse node names for different entities.
    - Example: A -> has -> uncle_1, uncle_1 -> is -> B
# Rule for Entity
- Be careful of Capitalization Entity, e.g. Einstein is different from einstein
# Rule for Quantities and Collections
- When a sentence mentions a counted group, represent the entire group as a single collective node using the quantity and type. Do NOT enumerate individual instances.
- If additional facts apply to that group, attach them to the collective node as further triplets.
"""

EXTRACT_KNOWN_TRIPLET_USER = "Document:\n{documents}\nEntity: {entity}"

MERGE_TRIPLETS_SYSTEM = """You are given a claim and list of knowledge graph triplets
Your task is to merge, normalize and connect these triplets into a coherent knowledge graph
# Instruction
1. Normalize entities
- Merge entities that refer to the same concept
- Replace generic entities with specific entities when appropriate
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

MERGE_TRIPLETS_USER = "Triplets:\n{triplets}"

REMOVE_DUPLICATED_TRIPLETS_SYSTEM = """
"""