from typing import Optional, DefaultDict
from collections import defaultdict
import re
import logging
import itertools
import random
import numpy as np

logger = logging.getLogger(__name__)


class Triplet:
    def __init__(
            self,
            triplet_text: str
    ):
        self.triplet_text = triplet_text
        self.elements = self.split_all(triplet_text)
        self.sentence = " ".join(self.elements) if self.elements else None

    def split_all(self, triplet_text: str) -> Optional[list[str]]:
        if "[SEP]" not in triplet_text:
            logger.error(f"Invalid triple format: '{triplet_text}'")
            return None

        elements = re.split(r"\[SEP\]|\[PREP\]", triplet_text)
        elements = [ele.strip() for ele in elements if ele.strip()]

        return elements


class DefinitionTriplet(Triplet):
    def __init__(
            self,
            def_triple_text: str
    ):
        super().__init__(def_triple_text)

        if not self.elements:
            self.latent_entity = None
            self.definition = None
        else:
            self.latent_entity = self.elements[0]
            self.definition = " ".join(self.elements[2:]) if len(self.elements) > 2 else ""


class Graph:
    def __init__(
            self,
            def_triple_sents: list[str],
            triple_sents: list[str]
    ):
        self.def_triple_sents = def_triple_sents
        self.triple_sents = triple_sents

        self.def_triples = [
            def_triple for sent in def_triple_sents if (def_triple := DefinitionTriplet(sent)).elements is not None]
        self.triples: list[Triplet] = [
            triple for sent in triple_sents if (triple := Triplet(sent)).elements is not None]
        self.total_triples = self.def_triples + self.triples

        self.la_ent_2_def = self.get_la_ent_2_def()
        self.la_ent_2_def_triple = self.get_la_ent_2_def_triple()

        self.la_ent_list = list(self.la_ent_2_def.keys())
        self.la_ent_index = {la_ent: idx for idx, la_ent in enumerate(self.la_ent_list)}
        self.num_la_ent = len(self.la_ent_list)

        self.has_la_ent_w_no_def = 0
        self.la_ent_2_sub_triples = self.get_la_ent_2_sub_triples()
        self.adjacent_la_ent_pairs = None

    def get_la_ent_2_def(
            self
    ) -> dict[str, str]:

        la_ent_2_def = {}
        for def_triple in self.def_triples:
            la_ent_2_def[def_triple.latent_entity] = def_triple.definition

        return la_ent_2_def

    def get_la_ent_2_def_triple(
            self
    ) -> dict[str, DefinitionTriplet]:

        la_ent_2_def_triple = {}
        for def_triple in self.def_triples:
            la_ent_2_def_triple[def_triple.latent_entity] = def_triple

        return la_ent_2_def_triple

    def get_la_ent_2_sub_triples(
            self
    ) -> DefaultDict[str, list[Triplet]]:

        la_ent_2_sub_triples = defaultdict(list)

        for triple in self.triples:
            la_ents = set(re.findall(r"\(ENT\d+\)", triple.sentence))

            for la_ent in la_ents:
                if la_ent in self.la_ent_list:
                    la_ent_2_sub_triples[la_ent].append(triple)
                else:
                    self.has_la_ent_w_no_def = 1

        return la_ent_2_sub_triples

    def get_adjacent_la_ent_pairs(
            self
    ) -> list[(str, str)]:

        adjacency_matrix = np.zeros((self.num_la_ent, self.num_la_ent), dtype=int)

        for triple in self.total_triples:
            la_ents = re.findall(r"\(ENT\d+\)", triple.sentence)

            for i in range(len(la_ents)):
                for j in range(i + 1, len(la_ents)):
                    if la_ents[i] in self.la_ent_list and la_ents[j] in self.la_ent_list:
                        idx1, idx2 = self.la_ent_index[la_ents[i]], self.la_ent_index[la_ents[j]]
                        adjacency_matrix[idx1][idx2] = 1
                        adjacency_matrix[idx2][idx1] = 1

        pair_list = []
        for idx1 in range(self.num_la_ent):
            for idx2 in range(idx1 + 1, self.num_la_ent):
                if adjacency_matrix[idx1][idx2] == 1:
                    pair_list.append((self.la_ent_list[idx1], self.la_ent_list[idx2]))

        return pair_list

    def backtrack(
            self,
            rule: list[(str, str)],
            path: list[str],
            used_ent: list[str]
    ) -> Optional[list[str]]:

        if len(path) == self.num_la_ent:
            return path

        for ent in self.la_ent_list:
            if ent not in used_ent:
                follow_rule = True
                updated_path = path + [ent]

                # Check if the new entity maintains the rule constraints
                for (a, b) in rule:
                    if a in updated_path and b in updated_path:
                        if updated_path.index(a) > updated_path.index(b):
                            follow_rule = False
                            break

                if follow_rule:
                    used_ent.add(ent)
                    result = self.backtrack(rule, updated_path, used_ent)
                    used_ent.remove(ent)

                    if result:
                        return result  # Return the first valid sequence found

        return None

    def get_valid_paths(
            self,
            path_limit: int = 5
    ) -> list[list[str]]:
        """
        Generate latent entity sequences where order variations may lead to different results in latent entity identification.

        - The order of adjacent nodes (i.e., latent entities with direct connections) affects the outcome.
        - The order of non-adjacent nodes (i.e., latent entities without direct connections) does not affect the outcome.

        Based on this, the function generates sequences with different orderings of adjacent nodes.
        """

        if not self.adjacent_la_ent_pairs:
            self.adjacent_la_ent_pairs = self.get_adjacent_la_ent_pairs()

        # Generate all possible adjacency pair permutations
        rule_list = []
        for do_flip in itertools.product([False, True], repeat=len(self.adjacent_la_ent_pairs)):
            rule = []
            for (left_ent, right_ent), flip in zip(self.adjacent_la_ent_pairs, do_flip):
                if flip:
                    rule.append((right_ent, left_ent))
                else:
                    rule.append((left_ent, right_ent))
            rule_list.append(rule)

        # Shuffle rules if there are more than path_limit to introduce randomness
        if len(rule_list) > path_limit:
            random.shuffle(rule_list)

        valid_paths = []

        for rule in rule_list:
            path = self.backtrack(rule, [], set())
            if path and path not in valid_paths:
                valid_paths.append(path)

            if len(valid_paths) >= path_limit:
                break

        valid_paths = [list(seq) for seq in valid_paths]
        return valid_paths