import json
import csv
import ast
from typing import Optional, Any

from .base import Dataset, LABELS


class ExFever(Dataset):
    """
    EX-FEVER dataset loader for multi-hop explainable fact verification.
    
    Supports both CSV and JSON formats.
    
    CSV Format (dev.csv, test.csv, mini_test.csv):
        - claim: The claim to verify
        - explanation: Multi-hop reasoning path
        - label: Verdict (SUPPORT/REFUTE/NOT ENOUGH INFO)
        - golden entity: List of gold entities
        - mention: Entity mentions in claim
        - result entity: Result entities from reasoning
        - discard number: Discarded evidence count
    
    JSON Format (train_nei_x.json):
        - Same fields as CSV, stored as JSON objects
    """
    
    def __init__(
        self,
        claims: list[str],
        explanations: list[str],
        labels: list[str],
        golden_entities: Optional[list[list[str]]] = None,
        mentions: Optional[list[list[str]]] = None,
        result_entities: Optional[list[list[str]]] = None,
        discard_numbers: Optional[list[float]] = None,
        **kwargs
    ):
        """
        Initialize ExFever dataset.
        
        Args:
            claims: List of claims to verify
            explanations: List of multi-hop explanations
            labels: List of labels (SUPPORT/REFUTE/NEI)
            golden_entities: List of gold entity lists
            mentions: List of mention entity lists
            result_entities: List of result entity lists
            discard_numbers: List of discard count values
        """
        # Store explanations as both contexts (for base class compatibility) and explanations
        super().__init__(
            claims=claims,
            contexts=explanations,
            evidences=None,
            labels=labels,
            **kwargs
        )
        self.explanations = explanations
        self.golden_entities = golden_entities
        self.mentions = mentions
        self.result_entities = result_entities
        self.discard_numbers = discard_numbers

    @classmethod
    def from_csv(cls, path: str) -> "ExFever":
        """
        Load EX-FEVER dataset from CSV file.
        
        Args:
            path: Path to CSV file (e.g., dev.csv, mini_test.csv)
            
        Returns:
            ExFever dataset instance
        """
        claims = []
        explanations = []
        labels = []
        golden_entities = []
        mentions = []
        result_entities = []
        discard_numbers = []
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                claims.append(row['claim'])
                explanations.append(row['explanation'])
                
                # Normalize label to SUPPORT/REFUTE/NEI
                raw_label = row['label'].strip()
                if raw_label == 'NOT ENOUGH INFO':
                    label = 'NEI'
                else:
                    label = raw_label
                labels.append(label)
                
                # Parse list fields (may be empty or None)
                ge_str = row.get('golden entity', '') or ''
                golden_entities.append(cls._parse_list_field(ge_str))
                
                m_str = row.get('mention', '') or ''
                mentions.append(cls._parse_list_field(m_str))
                
                re_str = row.get('result entity', '') or ''
                result_entities.append(cls._parse_list_field(re_str))
                
                dn_str = row.get('discard number', '') or ''
                discard_numbers.append(cls._parse_float_field(dn_str))
        
        return cls(
            claims=claims,
            explanations=explanations,
            labels=labels,
            golden_entities=golden_entities,
            mentions=mentions,
            result_entities=result_entities,
            discard_numbers=discard_numbers
        )

    @classmethod
    def from_json(cls, path: str) -> "ExFever":
        """
        Load EX-FEVER dataset from JSON file.
        
        Args:
            path: Path to JSON file (e.g., train_nei_x.json)
            
        Returns:
            ExFever dataset instance
        """
        claims = []
        explanations = []
        labels = []
        golden_entities = []
        mentions = []
        result_entities = []
        discard_numbers = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                
                claims.append(record['claim'])
                explanations.append(record['explanation'])
                
                # Normalize label
                raw_label = record['label'].strip()
                if raw_label == 'NOT ENOUGH INFO':
                    label = 'NEI'
                else:
                    label = raw_label
                labels.append(label)
                
                golden_entities.append(record.get('golden entity', []))
                mentions.append(record.get('mention', []))
                result_entities.append(record.get('result entity', []))
                discard_numbers.append(record.get('discard number'))
        
        return cls(
            claims=claims,
            explanations=explanations,
            labels=labels,
            golden_entities=golden_entities,
            mentions=mentions,
            result_entities=result_entities,
            discard_numbers=discard_numbers
        )

    @staticmethod
    def _parse_list_field(value: str) -> list[str]:
        """Parse a string representation of a list (e.g., \"['entity1', 'entity2']\")."""
        if not value or value.strip() == '':
            return []
        try:
            # Handle string representation of list
            parsed = ast.literal_eval(value.strip())
            if isinstance(parsed, list):
                return parsed
            return []
        except (ValueError, SyntaxError):
            return []

    @staticmethod
    def _parse_float_field(value: str) -> Optional[float]:
        """Parse a float value, returning None if empty or invalid."""
        if not value or value.strip() == '':
            return None
        try:
            return float(value.strip())
        except ValueError:
            return None

    def __iter__(self):
        """
        Iterate over dataset samples.
        
        Yields:
            dict with keys: claim, explanation, label, golden_entities, mentions, 
                          result_entities, discard_numbers, and base keys (context, evidence)
        """
        for i in range(len(self.claims)):
            sample = {
                "claim": self.claims[i],
                "explanation": self.explanations[i],
                "label": self.labels[i] if self.labels else None,
                "golden_entities": self.golden_entities[i] if self.golden_entities else None,
                "mentions": self.mentions[i] if self.mentions else None,
                "result_entities": self.result_entities[i] if self.result_entities else None,
                "discard_numbers": self.discard_numbers[i] if self.discard_numbers else None,
                # Base Dataset keys for compatibility
                "context": self.contexts[i] if self.contexts else None,
                "evidence": self.evidences[i] if self.evidences else None
            }
            yield sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get a single sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            dict with all EX-FEVER fields
        """
        return {
            "claim": self.claims[index] if index < len(self.claims) else None,
            "explanation": self.explanations[index] if self.explanations and index < len(self.explanations) else None,
            "label": self.labels[index] if self.labels and index < len(self.labels) else None,
            "golden_entities": self.golden_entities[index] if self.golden_entities and index < len(self.golden_entities) else None,
            "mentions": self.mentions[index] if self.mentions and index < len(self.mentions) else None,
            "result_entities": self.result_entities[index] if self.result_entities and index < len(self.result_entities) else None,
            "discard_numbers": self.discard_numbers[index] if self.discard_numbers and index < len(self.discard_numbers) else None,
            # Base Dataset keys for compatibility
            "context": self.contexts[index] if self.contexts and index < len(self.contexts) else None,
            "evidence": self.evidences[index] if self.evidences and index < len(self.evidences) else None
        }


class ExFeverWithRetriever(ExFever):
    """
    EX-FEVER dataset loader with optional BM25 retrieval support.
    
    This class extends basic ExFever loading with retrieval capability
    using BM25 over the EX-FEVER Wikipedia corpus.
    """
    
    def __init__(
        self,
        claims: list[str],
        explanations: list[str],
        labels: list[str],
        golden_entities: Optional[list[list[str]]] = None,
        mentions: Optional[list[list[str]]] = None,
        result_entities: Optional[list[list[str]]] = None,
        discard_numbers: Optional[list[float]] = None,
        retriever: Optional[Any] = None,  # BM25Retriever
        **kwargs
    ):
        """
        Initialize ExFever dataset with optional retriever.
        
        Args:
            claims: List of claims to verify
            explanations: List of multi-hop explanations
            labels: List of labels (SUPPORT/REFUTE/NEI)
            golden_entities: List of gold entity lists
            mentions: List of mention entity lists
            result_entities: List of result entity lists
            discard_numbers: List of discard count values
            retriever: Optional BM25Retriever for document retrieval
        """
        super().__init__(
            claims=claims,
            explanations=explanations,
            labels=labels,
            golden_entities=golden_entities,
            mentions=mentions,
            result_entities=result_entities,
            discard_numbers=discard_numbers,
            **kwargs
        )
        self.retriever = retriever
