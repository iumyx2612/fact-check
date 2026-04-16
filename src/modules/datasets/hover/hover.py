import json
from typing import Optional, Any

from ..base import Dataset, LABELS


class HoverDataset(Dataset):
    """
    HoVer dataset loader for multi-hop fact verification.

    JSON Format (hover_dev_release_v1.1.json, hover_train_release_v1.1.json):
        - uid: Unique identifier
        - claim: The claim to verify
        - supporting_facts: List of [title, sentence_idx] pairs
        - label: Verdict (SUPPORTED / NOT_SUPPORTED)
        - num_hops: Number of hops required
        - hpqa_id: HotpotQA origin id (optional)
    """

    def __init__(
            self,
            claims: list[str],
            labels: list[str],
            golden_docs: list[list[str]],
            num_hops: Optional[list[int]] = None,
            uids: Optional[list[str]] = None,
            hpqa_ids: Optional[list[str]] = None,
            **kwargs
    ):
        super().__init__(
            claims=claims,
            golden_docs=golden_docs,
            contexts=None,
            evidences=None,
            labels=labels,
            **kwargs
        )
        self.num_hops = num_hops
        self.uids = uids
        self.hpqa_ids = hpqa_ids

    @classmethod
    def from_json(cls, path: str) -> "HoverDataset":
        """
        Load HoVer dataset from JSON file.

        Args:
            path: Path to JSON file (e.g., hover_dev_release_v1.1.json)

        Returns:
            HoverDataset instance
        """
        claims = []
        labels = []
        golden_docs = []
        num_hops = []
        uids = []
        hpqa_ids = []

        with open(path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        for record in records:
            claims.append(record['claim'])

            raw_label = record['label'].strip()
            if raw_label == 'SUPPORTED':
                label = 'SUPPORT'
            elif raw_label == 'NOT_SUPPORTED':
                label = 'REFUTE'
            else:
                label = raw_label
            labels.append(label)

            golden_docs.append([fact[0] for fact in record.get('supporting_facts', [])])
            num_hops.append(record.get('num_hops'))
            uids.append(record.get('uid'))
            hpqa_ids.append(record.get('hpqa_id'))

        return cls(
            claims=claims,
            labels=labels,
            golden_docs=golden_docs,
            num_hops=num_hops,
            uids=uids,
            hpqa_ids=hpqa_ids
        )

    def __iter__(self):
        for i in range(len(self.claims)):
            yield self[i]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "uid": self.uids[index] if self.uids else None,
            "claim": self.claims[index],
            "label": self.labels[index] if self.labels else None,
            "golden_docs": self.golden_docs[index] if self.golden_docs else None,
            "num_hops": self.num_hops[index] if self.num_hops else None,
            "hpqa_id": self.hpqa_ids[index] if self.hpqa_ids else None,
            # Base Dataset keys for compatibility
            "context": self.contexts[index] if self.contexts else None,
            "evidence": self.evidences[index] if self.evidences else None,
        }
