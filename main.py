import argparse
import json
import os
import numpy as np
import torch
import csv
import networkx as nx
from sklearn.model_selection import KFold


from utils.kg_io import parse_hetionet_triplets, safe_load_numpy_array
from utils.kg_preprocessing import process_triplets_generic, create_mappings_from_triplets
from utils.kg_supervision import extract_supervised_from_kg
from utils.statistics import summarize_metrics
from models.hybrid_denoiser import HybridDenoiser


def main(args):

    if not os.path.isfile(args.kge):
        raise FileNotFoundError(f"KGE file not found: {args.kge}")
    features = np.load(args.kge)

    semantic_features = None
    if args.semantic is not None:
        if not os.path.isfile(args.semantic):
            raise FileNotFoundError(f"Semantic features file not found: {args.semantic}")
        semantic_features = safe_load_numpy_array(args.semantic)
        if args.semantic_in_dim is None:
            args.semantic_in_dim = semantic_features.shape[1]

    if not os.path.isfile(args.kgpath):
        raise FileNotFoundError(args.kgpath)
    het_triplets = parse_hetionet_triplets(args.kgpath)
    id_to_name, id_to_info, id_to_idx, idx_to_id, id_to_idx_strkeys = create_mappings_from_triplets(het_triplets)
    processed_triplets, rel_types = process_triplets_generic(het_triplets, id_to_idx)
    # Save to tab-delimited file
    output_file = "structured_training_data/id_to_idx.tsv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["entity", "id"])  # header
        for key, value in id_to_idx.items():
            writer.writerow([key, value])
    with open("structured_training_data/processed_triplets_drkg.json", "w", encoding="utf-8") as f:
        json.dump(processed_triplets, f, indent=4, ensure_ascii=False)

    G = nx.DiGraph()
    for t in processed_triplets:
        G.add_edge(t['source_idx'], t['target_idx'], relation=t['relation'])
    print(f"[INFO] Nodes={len(G.nodes)}, Edges={len(G.edges)}, Triplets={len(processed_triplets)}")

    supervised_pairs = extract_supervised_from_kg(processed_triplets, id_to_info, id_to_idx, relation_whitelist=None, require_types=False)
    print(f"[INFO] Extracted supervised pairs from KG: {len(supervised_pairs)}")

    denoiser = HybridDenoiser(
        in_dim=None,
        hid_dim=256, out_dim=128, lr=1e-3, epochs=args.epochs, alpha=args.alpha,
        eval_interval=args.eval_interval, save_model_path=args.save_model or 'hybrid_denoiser.pt',
        metrics_out='training_metrics.json', random_seed=args.kfold_seed,
        prune_start_pct=args.prune_start_pct, prune_end_pct=args.prune_end_pct,
        prune_start_epoch=args.prune_start_epoch, prune_interval=args.prune_interval,
        smooth_alpha=args.smooth_alpha, batch_size=args.batch_size, use_amp=args.use_amp,
        semantic_in_dim=args.semantic_in_dim, proj_dim=args.proj_dim, lambda_mi=args.lambda_mi
    )

    # Use the fixed threshold (0.7) if provided via CLI or default to 0.7 when not provided
    fixed_thr = float(args.fixed_threshold) if args.fixed_threshold is not None else 0.5
    denoiser.fixed_prune_threshold = fixed_thr

    # ---------- K-Fold with train / val / test split per fold ----------
    if args.kfold > 1:
        if len(supervised_pairs) == 0:
            raise ValueError("K-Fold requested but no supervised pairs available.")
        # fraction of the training fold to reserve for validation
        val_frac = getattr(args, "kfold_val_frac", 0.1)  # default 10% for validation
        print(f"[INFO] Running {args.kfold}-fold CV on supervised pairs with seed {args.kfold_seed}...")
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.kfold_seed)
        sup_arr = np.array(list(supervised_pairs), dtype=object)

        # containers to accumulate metrics across folds
        val_metrics_list = []
        test_metrics_list = []

        fold_id = 0
        for train_idx, test_idx in kf.split(sup_arr):
            fold_id += 1
            # reproducible shuffle for splitting train -> train/val
            rng = np.random.RandomState(args.kfold_seed + fold_id)
            train_idx_shuffled = train_idx.copy()
            rng.shuffle(train_idx_shuffled)

            # determine val size (at least 1 if possible)
            n_train_fold = len(train_idx_shuffled)
            val_size = max(1, int(round(n_train_fold * val_frac))) if n_train_fold > 1 else 0

            if val_size >= n_train_fold:
                # fallback: keep at least one training example if possible
                val_size = max(0, n_train_fold - 1)

            val_idx = train_idx_shuffled[:val_size] if val_size > 0 else np.array([], dtype=int)
            new_train_idx = train_idx_shuffled[val_size:] if val_size < n_train_fold else np.array([], dtype=int)

            # build pair collections
            train_pairs = set(tuple(x) for x in sup_arr[new_train_idx].tolist()) if len(new_train_idx) > 0 else set()
            val_pairs = [tuple(x) for x in sup_arr[val_idx].tolist()] if len(val_idx) > 0 else []
            test_pairs = [tuple(x) for x in sup_arr[test_idx].tolist()] if len(test_idx) > 0 else []

            print(f"[INFO] Fold {fold_id}: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

            if len(train_pairs) == 0:
                print(f"[WARN] Fold {fold_id}: no training supervised pairs after splitting — skipping this fold.")
                continue

            # instantiate a fresh denoiser per fold (same as before)
            den = HybridDenoiser(in_dim=None, hid_dim=256, out_dim=128, lr=1e-3, epochs=args.epochs, alpha=args.alpha,
                                 eval_interval=args.eval_interval, save_model_path=f'hybrid_fold{fold_id}.pt',
                                 metrics_out=f'training_metrics_fold{fold_id}.json',
                                 random_seed=args.kfold_seed + fold_id,
                                 prune_start_pct=args.prune_start_pct, prune_end_pct=args.prune_end_pct,
                                 prune_start_epoch=args.prune_start_epoch, prune_interval=args.prune_interval,
                                 smooth_alpha=args.smooth_alpha, batch_size=args.batch_size, use_amp=args.use_amp,
                                 semantic_in_dim=args.semantic_in_dim, proj_dim=args.proj_dim, lambda_mi=args.lambda_mi)
            den.fixed_prune_threshold = fixed_thr

            # train on the train_pairs
            den.train(G, features, processed_triplets, supervised_pairs=train_pairs,
                      num_supervised_neg_per_pos=1, verbose=True, mode=args.mode, semantic_features=semantic_features)

            # prepare features (handle complex-valued RotatE embeddings split to real/imag if present)
            feat_for_eval = np.concatenate([features.real, features.imag], axis=1) if np.iscomplexobj(
                features) else features
            train_src_list = [int(x['source_idx']) for x in processed_triplets]
            train_dst_list = [int(x['target_idx']) for x in processed_triplets]
            train_edge_index = torch.tensor([train_src_list, train_dst_list], dtype=torch.long, device=denoiser.device)

            # evaluate on validation (if available)
            if len(val_pairs) > 0:
                eval_val = den.evaluate_pairs(feat_for_eval, val_pairs, edge_index=train_edge_index)
                if eval_val and 'metrics' in eval_val:
                    val_metrics_list.append(eval_val['metrics'])
                    print(f"[Fold {fold_id} VAL] {eval_val['metrics']}")
                else:
                    print(f"[WARN] Fold {fold_id}: validation evaluation returned no metrics.")
            else:
                print(f"[INFO] Fold {fold_id}: no validation pairs (skipped).")

            # evaluate on test
            if len(test_pairs) > 0:
                eval_test = den.evaluate_pairs(feat_for_eval, test_pairs, edge_index=train_edge_index)
                if eval_test and 'metrics' in eval_test:
                    test_metrics_list.append(eval_test['metrics'])
                    print(f"[Fold {fold_id} TEST] {eval_test['metrics']}")
                else:
                    print(f"[WARN] Fold {fold_id}: test evaluation returned no metrics.")
            else:
                print(f"[INFO] Fold {fold_id}: no test pairs (skipped).")

        summarize_metrics(val_metrics_list, name="Validation")
        summarize_metrics(test_metrics_list, name="Test")

        return


    if args.save_model is not None:
        try:
            den.save_checkpoint(args.save_model)
            print(f"[INFO] Model saved -> {args.save_model}")
        except Exception as e:
            print(f"[WARN] Could not save model: {e}")

    print("[INFO] Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid denoiser (Hetionet self-supervised + DrugBank supervised) - semantic + InfoNCE")
    parser.add_argument("--mode", choices=["hybrid", "selfsupervised", "supervised"], default="hybrid")
    parser.add_argument("--kge", type=str, required=True, help="KGE .npy file with node features")
    parser.add_argument("--semantic", type=str, default=None, help="Optional semantic features .npy (per-node LLM/text embeddings)")
    parser.add_argument("--kgpath", type=str, default=None, help="Path to Hetionet-style TSV of triplets (head\\trelation\\ttail)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.5, help="supervised loss weight")
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--kfold", type=int, default=5, help="K-Fold on supervised pairs. Must be >1.")
    parser.add_argument("--kfold_seed", type=int, default=42)
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--calibrate", action='store_true', help="Run calibration on supervised pairs and apply a threshold when saving denoised KG")
    parser.add_argument("--desired_precision", type=float, default=None, help="(optional) desired precision for calibration (0-1)")
    parser.add_argument("--prune_start_pct", type=float, default=40.0)
    parser.add_argument("--prune_end_pct", type=float, default=10.0)
    parser.add_argument("--prune_start_epoch", type=int, default=20)
    parser.add_argument("--prune_interval", type=int, default=2)
    parser.add_argument("--smooth_alpha", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=500000, help="Batch size for batched discriminator evaluation")
    parser.add_argument("--use_amp", action='store_true', help="Enable automatic mixed precision on CUDA")
    parser.add_argument("--semantic_in_dim", type=int, default=None, help="Dimension of semantic features (if provided)")
    parser.add_argument("--proj_dim", type=int, default=256, help="Projection dimension for InfoNCE")
    parser.add_argument("--lambda_mi", type=float, default=0.1, help="Weight for MI/InfoNCE loss")
    parser.add_argument("--fixed_threshold", type=float, default=None, help="If set, use this fixed threshold for pruning & denoising (overrides percentile pruning)")

    args = parser.parse_args()

    main(args)