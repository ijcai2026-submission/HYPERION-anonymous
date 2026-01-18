from typing import Dict, List, Tuple

def process_triplets_generic(triplets: List[Dict], id_to_idx: Dict) -> Tuple[List[Dict], List[str]]:
    processed = []
    rel_types = set()
    for entry in triplets:
        if not all(k in entry for k in ('source', 'target', 'relation')):
            continue
        s = entry['source']
        t = entry['target']
        r = entry['relation']
        if s not in id_to_idx or t not in id_to_idx:
            continue
        processed.append({
            'source_id': s,
            'target_id': t,
            'source_idx': id_to_idx[s],
            'target_idx': id_to_idx[t],
            'relation': r
        })
        rel_types.add(r)
    return processed, list(rel_types)

def create_mappings_from_triplets(triplets: List[Dict]):
    ids = set()
    for e in triplets:
        ids.add(e['source'])
        ids.add(e['target'])
    unique_ids = sorted(ids)
    id_to_idx = {eid: idx for idx, eid in enumerate(unique_ids)}
    id_to_name = {eid: eid for eid in unique_ids}
    id_to_info = {eid: {'name': eid, 'label': '', 'type': ''} for eid in unique_ids}
    id_to_idx_strkeys = {str(k): v for k, v in id_to_idx.items()}
    return id_to_name, id_to_info, id_to_idx, {v: k for k, v in id_to_idx.items()}, id_to_idx_strkeys
