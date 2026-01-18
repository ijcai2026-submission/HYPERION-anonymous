import numpy as np

def summarize_metrics(metrics_list, name="SET"):
    if not metrics_list:
        print(f"[INFO] No metrics to summarize for {name}.")
        return

    # gather all keys present
    all_keys = sorted({k for m in metrics_list for k in m.keys()})

    scalar_keys = []
    skipped_keys = []

    # helper to test if value is scalar numeric
    def _is_scalar_numeric(x):
        if x is None:
            return False
        if isinstance(x, (int, float, np.floating, np.integer, np.int64, np.float64)):
            return True
        # numpy scalar
        if isinstance(x, np.ndarray) and x.shape == ():
            return True
        return False

    # determine which keys are scalar across at least one entry
    for k in all_keys:
        # check at least one value is scalar numeric; if any non-scalar appears, we'll still include but will ignore non-scalar entries
        any_scalar = any(_is_scalar_numeric(m.get(k)) for m in metrics_list)
        if any_scalar:
            scalar_keys.append(k)
        else:
            skipped_keys.append(k)

    if skipped_keys:
        print(f"[INFO] Skipping non-scalar keys for summary: {', '.join(skipped_keys)}")

    summary = {}
    for k in scalar_keys:
        vals = []
        for m in metrics_list:
            v = m.get(k, np.nan)
            if _is_scalar_numeric(v):
                vals.append(float(v))
            else:
                # insert nan for non-scalar or missing entries
                vals.append(np.nan)
        vals = np.array(vals, dtype=float)
        summary[k] = (np.nanmean(vals), np.nanstd(vals))

    print(f"[SUMMARY] {name} mean ± std over {len(metrics_list)} folds:")
    for k, (mu, sd) in summary.items():
        print(f"  {k}: {mu:.4f} ± {sd:.4f}")