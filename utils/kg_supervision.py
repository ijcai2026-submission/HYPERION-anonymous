def extract_supervised_from_kg(processed_triplets, id_to_info, id_to_idx, relation_whitelist=None, require_types=False):
    supervised = set()
    def norm(s): return str(s).strip().lower() if s is not None else ''
    rel_set = None
    if relation_whitelist is not None:
        rel_set = set([norm(r) for r in relation_whitelist])
    for t in processed_triplets:
        rel = norm(t.get('relation',''))
        s_id = t.get('source_id')
        t_id = t.get('target_id')
        if rel_set is not None:
            if rel not in rel_set:
                continue
        else:
            if not any(sub in rel for sub in ('drug', 'target', 'bind', 'interact', 'inhibit', 'agonist', 'antagonist', 'ddi')):
                continue
        if require_types:
            si = id_to_info.get(s_id, {})
            ti = id_to_info.get(t_id, {})
            s_type = str(si.get('type','')).lower()
            t_type = str(ti.get('type','')).lower()
            if not (looks_like_drug(s_type) or looks_like_drug(t_type) or looks_like_protein(s_type) or looks_like_protein(t_type)):
                continue
        if str(s_id) in id_to_idx and str(t_id) in id_to_idx:
            supervised.add((int(id_to_idx[str(s_id)]), int(id_to_idx[str(t_id)])))
        else:
            si = t.get('source_idx')
            ti = t.get('target_idx')
            if si is not None and ti is not None:
                supervised.add((int(si), int(ti)))
    return supervised

def looks_like_drug(x: str) -> bool:
    if x is None: return False
    s = str(x).lower()
    return 'drug' in s or 'chemical' in s or 'compound' in s

def looks_like_protein(x: str) -> bool:
    if x is None: return False
    s = str(x).lower()
    return 'protein' in s or 'target' in s or 'gene' in s or 'enzyme' in s