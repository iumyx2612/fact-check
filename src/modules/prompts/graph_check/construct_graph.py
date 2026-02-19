GRAPH_CONSTRUCT_USER = """
We are conducting fact-checking on multi-hop claims. To facilitate this process, we need to decompose each claim into triples for more granular and accurate fact-checking. Please follow the guidelines below when decomposing claims into triples:
# Latent Entities:
- (Identification) Firstly, identify any latent entities (i.e., implicit references not directly mentioned in the claim) that need to be clarified for accurate fact-checking.
- (Definition) Define these identified latent entities in triple format, using placeholders like (ENT1), (ENT2), etc.
# Triples:
- (Basic Information Unit) Decompose the claim into triples, ensuring you reach the most fundamental verifiable information while preserving the original meaning. Be careful not to lose important information during decomposition.
- (Triple Structure) Each triple should follow this format: ‘subject [SEP] relation [SEP] object’. Both the subject and object should be noun phrases, while the relation should be a verb or verb phrase, forming a complete sentence.
- (Prepositional Phrases) In exceptional cases where a prepositional phrase modifies the entire triple (rather than just the subject or object) and splitting it into another triple would alter the meaning of the claim, do not divide it. Instead, append it to the end of the triple: ‘subject [SEP] relation [SEP] object [PREP] preposition phrase’.
- (Pronoun Resolution) Replace any pronouns with the corresponding entities to ensure that each triple is self-contained and independent of external context.
- (Entity Consistency) Use the exact same string to represent entities (i.e., the ‘subject’ or ‘object’) whenever they refer to the same entity across different triples.

# Claim: 
The fairy Queen Mab orginated with William Shakespeare.
# Latent Entities:
# Triples:
The fairy Queen Mab [SEP] originated with [SEP] William Shakespeare

# Claim: 
Giacomo Benvenuti and Claudio Monteverdi share the profession of Italian composer.
# Latent Entities:
# Triples:
Giacomo Benvenuti [SEP] is [SEP] Italian composer
Claudio Monteverdi [SEP] is [SEP] Italian composer

# Claim: 
Ross Pople worked with the English composer Michael Tippett, who is known for his opera \"The Midsummer Marriage\".
# Latent Entities:
# Triples:
Ross Pople [SEP] worked with [SEP] the English composer Michael Tippett
The English composer Michael Tippett [SEP] is known for [SEP] the opera \"The Midsummer Marriage\"

# Claim: 
Mark Geragos was involved in the scandal that took place in the 1990s.
# Latent Entities:
(ENT1) [SEP] is [SEP] a scandal
# Triples:
Mark Geragos [SEP] was involved in [SEP] (ENT1)
(ENT1) [SEP] took place in [SEP] the 1990s

# Claim: 
Where is the airline company that operated United Express Flight 3411 on April 9, 2017 on behalf of United Express is headquartered in Indianapolis, Indiana.
# Latent Entities:
(ENT1) [SEP] is [SEP] an airline company
# Triples:
(ENT1) [SEP] operated [SEP] United Express Flight 3411 [PREP] on April 9, 2017 on behalf of United Express
(ENT1) [SEP] is headquartered in [SEP] Indianapolis, Indiana

# Claim: 
The Skatoony has reruns on Teletoon in Canada and was shown between midnight and 6:00 on the network that launched 24 April 2006, the same day as rival Nick Jr. Too.
# Latent Entities:
(ENT1) [SEP] is [SEP] a network
# Triples: 
Skatoony [SEP] has reruns on [SEP] Teletoon
Teletoon [SEP] is located in [SEP] Canada
Skatoony [SEP] was shown on [SEP] (ENT1) [PREP] between midnight and 6:00
(ENT1) [SEP] launched on [SEP] 24 April 2006
Nick Jr. Too [SEP] launched on [SEP] 24 April 2006

# Claim: 
Danny Shirley is older than Kevin Parker.
# Latent Entities:
(ENT1) [SEP] is [SEP] a date
(ENT2) [SEP] is [SEP] a date
# Triples:
Danny Shirley [SEP] was born on [SEP] (ENT1)
Kevin Parker [SEP] was born on [SEP] (ENT2)
(ENT1) [SEP] is before [SEP] (ENT2)

# Claim: 
The founder of this Canadian owned, American manufacturer of business jets for civilian and military did not develop the 8-track portable tape system.
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] an American manufacturer
# Triples:
(ENT1) [SEP] founded [SEP] (ENT2)
(ENT2) [SEP] is owned by [SEP] Canadian
(ENT2) [SEP] made [SEP] business jets for civilian and military
(ENT1) [SEP] did not develop [SEP] 8-track portable tape system

# Claim: 
The Dutch man who along with Dennis Bergkamp was acquired in the 1993\u201394 Inter Milan season, manages Cruyff Football together with the footballer who is also currently manager of Tel Aviv team.
# Latent Entities:
(ENT1) [SEP] is [SEP] a Dutch man
(ENT2) [SEP] is [SEP] a footballer
# Triples:
(ENT1) [SEP] was acquired in [SEP] the 1993\u201394 Inter Milan season [PREP] along with Dennis Bergkamp
(ENT1) [SEP] manages [SEP] Cruyff Football [PREP] together with (ENT2)
(ENT2) [SEP] currently manages [SEP] Tel Aviv team

# Claim: 
An actor starred in the 2007 film based on a former FBI agent. That agent was Robert Philip Hanssen. The actor starred in the 2005 Capitol film Chaos.
# Latent Entities:
(ENT1) [SEP] is [SEP] an actor
(ENT2) [SEP] is [SEP] a 2007 film
# Triples:
(ENT1) [SEP] starred in [SEP] (ENT2)
(ENT2) [SEP] is based on [SEP] Robert Philip Hanssen
Robert Philip Hanssen [SEP] is [SEP] a former FBI agent
(ENT1) [SEP] starred in [SEP] the 2005 Capitol film Chaos

# Claim: 
<<target_claim>>
"""