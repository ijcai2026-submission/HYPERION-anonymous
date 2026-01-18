from typing import Optional, List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
import networkx as nx

from models.gcn_encoder import GCNEncoder
from models.semantic_encoder import SemanticEncoder
from models.edge_discriminator import EdgeDiscriminator
from models.projection_head import ProjectionHead
from torch_geometric.data import Data

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_recall_curve, 
    precision_score, recall_score, accuracy_score,
    cohen_kappa_score)


class HybridDenoiser:
    def __init__(
            self,
            in_dim: Optional[int] = None,
            hid_dim: int = 256,
            out_dim: int = 128,
            lr: float = 1e-3,
            epochs: int = 50,
            alpha: float = 0.5,
            eval_interval: int = 10,
            device: Optional[str] = None,
            save_model_path: str = "hybrid_denoiser.pt",
            metrics_out: str = "training_metrics.json",
            random_seed: int = 42,
            prune_start_pct: float = 40.0,
            prune_end_pct: float = 10.0,
            prune_start_epoch: int = 20,
            prune_interval: int = 2,
            smooth_alpha: float = 0.6,
            batch_size: int = 500000,
            use_amp: bool = False,
            semantic_in_dim: Optional[int] = None,
            proj_dim: int = 256,
            lambda_mi: float = 0.1,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.eval_interval = eval_interval
        self.in_dim = in_dim
        self.save_model_path = save_model_path
        self.metrics_out = metrics_out
        self.random_seed = random_seed

        self.prune_start_pct = float(prune_start_pct)
        self.prune_end_pct = float(prune_end_pct)
        self.prune_start_epoch = int(prune_start_epoch)
        self.prune_interval = int(prune_interval)
        self.smooth_alpha = float(smooth_alpha)

        self.batch_size = int(batch_size)
        self.use_amp = bool(use_amp)

        self.semantic_in_dim = semantic_in_dim or in_dim
        self.proj_dim = proj_dim
        self.lambda_mi = float(lambda_mi)

        # if set externally to a float, training will use this fixed threshold for pruning
        self.fixed_prune_threshold: Optional[float] = None

        self.encoder = None
        self.semantic_encoder = None
        self.discriminator = None
        self.proj_sub = None
        self.proj_sem = None
        self.opt = None
        self.scaler = torch.cuda.amp.GradScaler() if (self.use_amp and self.device.startswith("cuda")) else None

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        self.metrics_history = []

    def _init_models(self):
        assert self.in_dim is not None, "in_dim must be set before initializing models"
        self.encoder = GCNEncoder(self.in_dim, self.hid_dim, self.out_dim).to(self.device)
        self.semantic_encoder = SemanticEncoder(self.semantic_in_dim, hid_dim=self.hid_dim, out_dim=self.out_dim).to(self.device)
        joint_emb_dim = 2 * self.out_dim
        self.discriminator = EdgeDiscriminator(joint_emb_dim).to(self.device)
        self.proj_sub = ProjectionHead(self.out_dim, proj_dim=self.proj_dim).to(self.device)
        self.proj_sem = ProjectionHead(self.out_dim, proj_dim=self.proj_dim).to(self.device)

        params = list(self.encoder.parameters()) + list(self.discriminator.parameters()) + \
                 list(self.semantic_encoder.parameters()) + list(self.proj_sub.parameters()) + list(self.proj_sem.parameters())
        self.opt = torch.optim.Adam(params, lr=self.lr)

    def save_checkpoint(self, path: str):
        obj = {
            'encoder': self.encoder.state_dict() if self.encoder is not None else None,
            'semantic_encoder': self.semantic_encoder.state_dict() if self.semantic_encoder is not None else None,
            'proj_sub': self.proj_sub.state_dict() if self.proj_sub is not None else None,
            'proj_sem': self.proj_sem.state_dict() if self.proj_sem is not None else None,
            'discriminator': self.discriminator.state_dict() if self.discriminator is not None else None,
            'optimizer': self.opt.state_dict() if self.opt is not None else None,
        }
        torch.save(obj, path)

    def load_checkpoint(self, path: str, map_location: Optional[str] = None):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        ck = torch.load(path, map_location=map_location or self.device)
        if self.encoder is None or self.discriminator is None or self.semantic_encoder is None:
            raise RuntimeError("Call _init_models() with correct in_dim before loading checkpoint.")
        if ck.get('encoder') is not None:
            self.encoder.load_state_dict(ck['encoder'])
        if ck.get('semantic_encoder') is not None:
            self.semantic_encoder.load_state_dict(ck['semantic_encoder'])
        if ck.get('proj_sub') is not None:
            self.proj_sub.load_state_dict(ck['proj_sub'])
        if ck.get('proj_sem') is not None:
            self.proj_sem.load_state_dict(ck['proj_sem'])
        if ck.get('discriminator') is not None:
            self.discriminator.load_state_dict(ck['discriminator'])
        if ck.get('optimizer') is not None and self.opt is not None:
            try:
                self.opt.load_state_dict(ck['optimizer'])
            except Exception:
                pass

    @staticmethod
    def _info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        logits = torch.matmul(z_a, z_b.t()) / (temperature + 1e-12)
        labels = torch.arange(z_a.size(0), device=z_a.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)

    def structural_smoothing(self, h: torch.Tensor, edge_index: torch.Tensor, alpha: float = None) -> torch.Tensor:
        if alpha is None:
            alpha = self.smooth_alpha
        device = h.device
        num_nodes = h.shape[0]
        src = edge_index[0]
        dst = edge_index[1]
        neighbor_sum = torch.zeros_like(h, device=device)
        neighbor_sum.index_add_(0, dst, h[src])
        deg = torch.bincount(dst, minlength=num_nodes).to(h.dtype).unsqueeze(1).clamp(min=1.0).to(device)
        neighbor_mean = neighbor_sum / deg
        h_smooth = alpha * h + (1.0 - alpha) * neighbor_mean
        return h_smooth

    @staticmethod
    def adaptive_pruning(edge_index: torch.Tensor, scores: np.ndarray, percentile: float):
        if percentile <= 0.0:
            return edge_index
        if percentile >= 100.0:
            return edge_index
        thr = float(np.percentile(scores, percentile))
        mask = scores >= thr
        ei_np = edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index)
        if mask.sum() == 0:
            top_idx = int(np.argmax(scores))
            pruned = ei_np[:, [top_idx]]
        else:
            pruned = ei_np[:, mask]
        pruned_t = torch.tensor(pruned, dtype=torch.long, device=edge_index.device)
        return pruned_t

    def _batched_discriminator(self, h_joint: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        total = src_idx.size(0)
        if total == 0:
            return torch.empty((0,), device=h_joint.device)
        logits_parts = []
        B = max(1, self.batch_size)
        for i in range(0, total, B):
            s = src_idx[i:i + B]
            d = dst_idx[i:i + B]
            logits_parts.append(self.discriminator(h_joint[s], h_joint[d]))
        return torch.cat(logits_parts, dim=0)

    def train(
            self,
            graph_nx: Optional[nx.DiGraph],
            features: np.ndarray,
            processed_triplets: List[Dict],
            supervised_pairs: Optional[set] = None,
            num_supervised_neg_per_pos: int = 1,
            verbose: bool = True,
            mode: str = "hybrid",
            semantic_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert mode in {"hybrid", "selfsupervised", "supervised"}, "mode must be hybrid|selfsupervised|supervised"

        if np.iscomplexobj(features):
            features = np.concatenate([features.real, features.imag], axis=1)

        num_nodes = max(
            [t['source_idx'] for t in processed_triplets] + [t['target_idx'] for t in processed_triplets]) + 1
        feat_dim = features.shape[1]
        if self.in_dim is None:
            self.in_dim = feat_dim

        if features.shape[0] < num_nodes:
            if verbose:
                print(f"[WARN] Feature rows ({features.shape[0]}) < num_nodes ({num_nodes}) — padding features.")
            pad = np.zeros((num_nodes - features.shape[0], feat_dim), dtype=features.dtype)
            features = np.vstack([features, pad])
        elif features.shape[0] > num_nodes:
            if verbose:
                print(f"[WARN] Feature rows ({features.shape[0]}) > num_nodes ({num_nodes}) — trimming features.")
            features = features[:num_nodes]

        if semantic_features is None:
            sem_feat_dim = self.semantic_in_dim or self.in_dim
            semantic_features = np.zeros((features.shape[0], sem_feat_dim), dtype=np.float32)
        else:
            if np.iscomplexobj(semantic_features):
                semantic_features = np.concatenate([semantic_features.real, semantic_features.imag], axis=1)
            if semantic_features.shape[0] < num_nodes:
                pad = np.zeros((num_nodes - semantic_features.shape[0], semantic_features.shape[1]), dtype=semantic_features.dtype)
                semantic_features = np.vstack([semantic_features, pad])
            elif semantic_features.shape[0] > num_nodes:
                semantic_features = semantic_features[:num_nodes]

        src_list = [int(t['source_idx']) for t in processed_triplets]
        dst_list = [int(t['target_idx']) for t in processed_triplets]
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=self.device)
        current_edge_index = edge_index.clone()

        data = Data(x=torch.tensor(features, dtype=torch.float32, device=self.device), edge_index=current_edge_index)
        if semantic_features is not None:
            self.semantic_in_dim = semantic_features.shape[1]

        self._init_models()

        supervised_pairs = supervised_pairs or set()
        supervised_pairs_list = list(supervised_pairs)
        if mode == "supervised" and len(supervised_pairs_list) == 0:
            raise ValueError("Mode 'supervised' selected but no supervised pairs provided.")

        if verbose:
            print(
                f"[INFO] Mode={mode} | Nodes={data.x.shape[0]}, Edges={current_edge_index.size(1)}, FeatDim={feat_dim}")
            print(f"[INFO] Supervised positives: {len(supervised_pairs)} (alpha={self.alpha})")
            print(
                f"[INFO] Pruning schedule: start_pct={self.prune_start_pct} -> end_pct={self.prune_end_pct}, start_epoch={self.prune_start_epoch}, interval={self.prune_interval}")

        bce_loss = nn.BCEWithLogitsLoss()
        best_loss = float('inf')

        sem_tensor = torch.tensor(semantic_features, dtype=torch.float32, device=self.device)

        for epoch in range(self.epochs):
            self.encoder.train()
            self.discriminator.train()
            self.semantic_encoder.train()
            self.proj_sub.train()
            self.proj_sem.train()

            if self.use_amp and self.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    h_sub = self.encoder(data.x, current_edge_index)
            else:
                h_sub = self.encoder(data.x, current_edge_index)

            h_sem = self.semantic_encoder(sem_tensor)
            h_sub_smooth = self.structural_smoothing(h_sub, current_edge_index, alpha=self.smooth_alpha)
            h_joint = torch.cat([h_sub_smooth, h_sem], dim=-1)

            src = current_edge_index[0]
            dst = current_edge_index[1]
            num_edges = current_edge_index.size(1)

            pos_logits = self._batched_discriminator(h_joint, src, dst)

            if mode != "supervised":
                neg_dst = torch.randint(0, h_sub_smooth.shape[0], (num_edges,), device=self.device)
                neg_logits = self._batched_discriminator(h_joint, src, neg_dst)
                loss_pos = -F.logsigmoid(pos_logits).mean()
                loss_neg = -F.logsigmoid(-neg_logits).mean()
                loss_contrastive_graph = loss_pos + loss_neg
            else:
                loss_contrastive_graph = torch.tensor(0.0, device=self.device)

            loss_reg = (1 - torch.sigmoid(pos_logits)).pow(2).mean()

            loss_supervised = torch.tensor(0.0, device=self.device)
            if mode != "selfsupervised" and len(supervised_pairs_list) > 0:
                sp_src = []
                sp_dst = []
                for (a, b) in supervised_pairs_list:
                    if a < data.x.shape[0] and b < data.x.shape[0]:
                        sp_src.append(a)
                        sp_dst.append(b)
                if len(sp_src) > 0:
                    sp_src_t = torch.tensor(sp_src, dtype=torch.long, device=self.device)
                    sp_dst_t = torch.tensor(sp_dst, dtype=torch.long, device=self.device)
                    pos_sp_logits = self._batched_discriminator(h_joint, sp_src_t, sp_dst_t)

                    pos_labels = torch.ones_like(pos_sp_logits)
                    neg_sp_t = torch.randint(0, data.x.shape[0], (len(sp_src_t) * num_supervised_neg_per_pos,),
                                             device=self.device)
                    neg_sp_src_t = sp_src_t.repeat_interleave(num_supervised_neg_per_pos)
                    neg_sp_logits = self._batched_discriminator(h_joint, neg_sp_src_t, neg_sp_t)
                    neg_labels = torch.zeros_like(neg_sp_logits)

                    sup_logits = torch.cat([pos_sp_logits, neg_sp_logits], dim=0)
                    sup_labels = torch.cat([pos_labels, neg_labels], dim=0)
                    loss_supervised = bce_loss(sup_logits, sup_labels)

            z_sub = self.proj_sub(h_sub)
            z_sem = self.proj_sem(h_sem)
            try:
                loss_mi_1 = self._info_nce_loss(z_sub, z_sem)
                loss_mi_2 = self._info_nce_loss(z_sem, z_sub)
                loss_mi = 0.5 * (loss_mi_1 + loss_mi_2)
            except Exception:
                loss_mi = torch.tensor(0.0, device=self.device)

            if mode == "hybrid":
                loss = loss_contrastive_graph + self.alpha * loss_supervised + self.lambda_mi * loss_mi + 0.1 * loss_reg
            elif mode == "selfsupervised":
                loss = loss_contrastive_graph + self.lambda_mi * loss_mi + 0.1 * loss_reg
            else:
                loss = self.alpha * loss_supervised + self.lambda_mi * loss_mi + 0.1 * loss_reg

            self.opt.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                self.opt.step()

            # pruning: support fixed threshold pruning if set, otherwise percentile-based adaptive pruning
            if (epoch >= self.prune_start_epoch) and ((epoch - self.prune_start_epoch) % self.prune_interval == 0):
                if getattr(self, 'fixed_prune_threshold', None) is not None:
                    with torch.no_grad():
                        pos_scores_np = torch.sigmoid(pos_logits).cpu().numpy()
                    mask = pos_scores_np >= float(self.fixed_prune_threshold)
                    ei_np = current_edge_index.cpu().numpy()
                    if mask.sum() == 0:
                        top_idx = int(np.argmax(pos_scores_np))
                        pruned = ei_np[:, [top_idx]]
                    else:
                        pruned = ei_np[:, mask]
                    pruned_t = torch.tensor(pruned, dtype=torch.long, device=current_edge_index.device)
                    pruned_ei = pruned_t
                else:
                    prog = (epoch - self.prune_start_epoch) / max(1, (self.epochs - 1 - self.prune_start_epoch))
                    prog = float(np.clip(prog, 0.0, 1.0))
                    current_prune_pct = float(self.prune_start_pct * (1 - prog) + self.prune_end_pct * prog)
                    with torch.no_grad():
                        pos_scores_np = torch.sigmoid(pos_logits).cpu().numpy()
                    pruned_ei = self.adaptive_pruning(current_edge_index, pos_scores_np, percentile=current_prune_pct)

                if pruned_ei.size(1) < current_edge_index.size(1):
                    current_edge_index = pruned_ei.clone().to(self.device)
                    data.edge_index = current_edge_index
                    if verbose:
                        print(
                            f"[Prune] Epoch {epoch}: applied pruning -> kept {current_edge_index.size(1)} edges")

            if (epoch % self.eval_interval == 0) or (epoch == self.epochs - 1):
                self.encoder.eval()
                self.discriminator.eval()
                self.semantic_encoder.eval()
                self.proj_sub.eval()
                self.proj_sem.eval()
                with torch.no_grad():
                    if self.use_amp and self.device.startswith("cuda"):
                        with torch.cuda.amp.autocast():
                            h_sub_eval = self.encoder(data.x, current_edge_index)
                    else:
                        h_sub_eval = self.encoder(data.x, current_edge_index)
                    h_sem_eval = self.semantic_encoder(sem_tensor)
                    h_sub_eval_smooth = self.structural_smoothing(h_sub_eval, current_edge_index, alpha=self.smooth_alpha)
                    h_joint_eval = torch.cat([h_sub_eval_smooth, h_sem_eval], dim=-1)

                    pos_eval_logits = self._batched_discriminator(h_joint_eval, current_edge_index[0],
                                                                  current_edge_index[1])
                    neg_eval_dst = torch.randint(0, h_sub_eval_smooth.shape[0], (current_edge_index.size(1),),
                                                 device=self.device)
                    neg_eval_logits = self._batched_discriminator(h_joint_eval, current_edge_index[0], neg_eval_dst)

                    pos_scores = torch.sigmoid(pos_eval_logits).cpu().numpy()
                    neg_scores = torch.sigmoid(neg_eval_logits).cpu().numpy()

                    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
                    y_pred = np.concatenate([pos_scores, neg_scores])

                    metrics = self._safe_metrics(y_true, y_pred)

                    try:
                        K = min(10000, len(y_pred))
                        topk_idx = np.argsort(-y_pred)[:K]
                        prec_at_k = float(y_true[topk_idx].sum()) / K if K > 0 else float('nan')
                        recall_at_k = float(y_true[topk_idx].sum()) / (y_true.sum() + 1e-12)
                    except Exception:
                        prec_at_k = float('nan')
                        recall_at_k = float('nan')

                    metrics.update({
                        'epoch': epoch,
                        'loss': float(loss.item()),
                        'loss_contrastive_graph': float(loss_contrastive_graph.item()) if isinstance(loss_contrastive_graph, torch.Tensor) else 0.0,
                        'loss_supervised': float(loss_supervised.item()) if isinstance(loss_supervised, torch.Tensor) else 0.0,
                        'loss_mi': float(loss_mi.item()) if isinstance(loss_mi, torch.Tensor) else 0.0,
                        'prec_at_k': prec_at_k,
                        'recall_at_k': recall_at_k,
                        'time': time.time(),
                        'n_edges': int(current_edge_index.size(1))
                    })

                    if len(supervised_pairs_list) > 0:
                        try:
                            sp_src_arr = []
                            sp_dst_arr = []
                            for (a, b) in supervised_pairs_list:
                                if a < data.x.shape[0] and b < data.x.shape[0]:
                                    sp_src_arr.append(a)
                                    sp_dst_arr.append(b)
                            if len(sp_src_arr) > 0:
                                sp_src_t = torch.tensor(sp_src_arr, dtype=torch.long, device=self.device)
                                sp_dst_t = torch.tensor(sp_dst_arr, dtype=torch.long, device=self.device)
                                sp_logits = self._batched_discriminator(h_joint_eval, sp_src_t, sp_dst_t)
                                sp_scores = torch.sigmoid(sp_logits).cpu().numpy()
                                n_neg = len(sp_src_arr)
                                neg_t = torch.randint(0, data.x.shape[0], n_neg)
                                neg_src = np.array(sp_src_arr)
                                neg_logits_eval = self._batched_discriminator(
                                    h_joint_eval[torch.tensor(neg_src, dtype=torch.long, device=self.device)],
                                    h_joint_eval[torch.tensor(neg_t, dtype=torch.long, device=self.device)])
                                neg_scores_eval = torch.sigmoid(neg_logits_eval).cpu().numpy()
                                y_true_sp = np.concatenate([np.ones_like(sp_scores), np.zeros_like(neg_scores_eval)])
                                y_pred_sp = np.concatenate([sp_scores, neg_scores_eval])
                                sp_metrics = self._safe_metrics(y_true_sp, y_pred_sp)
                            else:
                                sp_metrics = {}
                        except Exception:
                            sp_metrics = {}
                        metrics['supervised_eval'] = sp_metrics

                    self.metrics_history.append(metrics)
                    if verbose:
                        print(
                            f"[Epoch {epoch}] loss={loss.item():.4f} edges={int(current_edge_index.size(1))} | AUC={metrics.get('auc'):.4f} AP={metrics.get('ap'):.4f} F1={metrics.get('f1'):.4f} P@{min(1000, len(y_pred))}={prec_at_k:.4f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                try:
                    self.save_checkpoint(self.save_model_path)
                except Exception:
                    pass

        self.encoder.eval()
        self.discriminator.eval()
        self.semantic_encoder.eval()
        with torch.no_grad():
            if self.use_amp and self.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    h_sub = self.encoder(data.x, current_edge_index)
            else:
                h_sub = self.encoder(data.x, current_edge_index)
            h_sem = self.semantic_encoder(sem_tensor)
            h_sub_smooth = self.structural_smoothing(h_sub, current_edge_index, alpha=self.smooth_alpha)
            h_joint = torch.cat([h_sub_smooth, h_sem], dim=-1)
            final_logits = self._batched_discriminator(h_joint, current_edge_index[0], current_edge_index[1])
            final_scores = torch.sigmoid(final_logits).cpu().numpy()

        try:
            with open(self.metrics_out, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Could not save metrics file: {e}")

        return final_scores

    def score_pairs(self, features: np.ndarray, pairs: List[Tuple[int, int]],
                    edge_index: Optional[torch.Tensor] = None) -> np.ndarray:
        """Score a list of (src_idx, dst_idx) pairs.

        Safe: pads features if they do not cover node indices referenced by edge_index.
        """
        if self.encoder is None or self.discriminator is None:
            raise RuntimeError("Models are not initialized/trained.")
        if np.iscomplexobj(features):
            features = np.concatenate([features.real, features.imag], axis=1)

        # determine required number of nodes from provided edge_index (if any) and from pairs
        req_nodes = 0
        if edge_index is not None:
            if isinstance(edge_index, torch.Tensor):
                req_nodes = int(edge_index.max().item()) + 1
            else:
                ei_np = np.asarray(edge_index)
                req_nodes = int(ei_np.max()) + 1
        if len(pairs) > 0:
            max_pair_idx = max(max(p) for p in pairs)
            req_nodes = max(req_nodes, int(max_pair_idx) + 1)

        # pad feature matrix if needed
        if features.shape[0] < req_nodes:
            pad_rows = req_nodes - features.shape[0]
            pad = np.zeros((pad_rows, features.shape[1]), dtype=features.dtype)
            features = np.vstack([features, pad])

        num_nodes = features.shape[0]
        x = torch.tensor(features, dtype=torch.float32, device=self.device)

        self.encoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            if edge_index is None:
                idxs = torch.arange(num_nodes, device=self.device, dtype=torch.long)
                edge_index_t = torch.stack([idxs, idxs], dim=0)
            else:
                edge_index_t = edge_index.to(self.device) if isinstance(edge_index, torch.Tensor) else torch.tensor(
                    edge_index, dtype=torch.long, device=self.device)
            h = self.encoder(x, edge_index_t)
            h_smooth = self.structural_smoothing(h, edge_index_t, alpha=self.smooth_alpha)
            src_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=self.device)
            dst_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=self.device)
            # final safety check
            max_idx_needed = int(max(src_idx.max().item() if src_idx.numel() else 0,
                                     dst_idx.max().item() if dst_idx.numel() else 0))
            if max_idx_needed >= h_smooth.shape[0]:
                raise RuntimeError(
                    f"score_pairs: required index {max_idx_needed} >= available nodes {h_smooth.shape[0]}")
            logits = self._batched_discriminator(h_smooth, src_idx, dst_idx)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs



    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, multi_class: bool):
        """
        Unified evaluation for binary and multi-class predictions.
        - If multi_class=False -> binary evaluation (ROC-AUC, PR-AUC, F1, etc.)
        - If multi_class=True  -> multi-class evaluation (macro/micro F1, kappa, accuracy)
        """

        if not multi_class:
            # =======================================
            # BINARY EVALUATION (original version)
            # =======================================
            metrics = {}

            # ROC-AUC
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
            except:
                metrics["roc_auc"] = float("nan")

            # AUPR
            try:
                metrics["pr_auc"] = float(average_precision_score(y_true, y_pred))
            except:
                metrics["pr_auc"] = float("nan")

            # Threshold-based predictions
            y_bin = (y_pred > 0.5).astype(int)

            metrics["f1"] = float(f1_score(y_true, y_bin))
            metrics["precision"] = float(precision_score(y_true, y_bin))
            metrics["recall"] = float(recall_score(y_true, y_bin))
            metrics["accuracy"] = float(accuracy_score(y_true, y_bin))

            # PR curve
            precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_pred)
            f1_curve = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-12)
            best_f1 = float(np.max(f1_curve))

            metrics["best_f1_curve"] = best_f1

            return metrics, precision_arr, recall_arr, thresholds

        else:
            # =======================================
            # MULTI-CLASS EVALUATION (BioKDN style)
            # =======================================
            metrics = {}

            # Convert predictions
            y_pred_labels = np.argmax(y_pred, axis=1)

            metrics["macro_f1"] = float(f1_score(y_true, y_pred_labels, average="macro"))
            metrics["micro_f1"] = float(f1_score(y_true, y_pred_labels, average="micro"))
            metrics["kappa"] = float(cohen_kappa_score(y_true, y_pred_labels))
            metrics["recall_macro"] = float(recall_score(y_true, y_pred_labels, average="macro"))
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred_labels))

            # No PR-curve in multi-class
            return metrics, None, None, None

    def evaluate_labeled_pairs(
            self,
            features: np.ndarray,
            pairs: List[Tuple[int, int]],
            y_true: np.ndarray,
            edge_index: Optional[torch.Tensor] = None,
            k: int = 1000
    ):
        """
        Evaluate using externally provided labels (test.tsv).
        """

        if len(pairs) == 0:
            return {}

        # Expand complex embeddings if needed
        if np.iscomplexobj(features):
            features = np.concatenate([features.real, features.imag], axis=1)

        # Score all pairs
        y_pred = self.score_pairs(features, pairs, edge_index=edge_index)

        # Metrics
        metrics = self._binary_metrics(y_true, y_pred, k)

        print(
            f"[EVAL-TSV] ROC-AUC={metrics['auc']:.4f} | "
            f"AUPR={metrics['aupr']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"P@{k}={metrics['p_at_k']:.4f}"
        )

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "metrics": metrics
        }


    def evaluate_pairs(
            self,
            features: np.ndarray,
            pos_pairs: List[Tuple[int, int]],
            num_neg: Optional[int] = None,
            edge_index: Optional[torch.Tensor] = None,
            k: int = 1000,
            use_multiclass: bool = False
    ) -> Dict[str, object]:
        """
        Evaluate positive pairs by sampling negatives and computing evaluation metrics.
        Supports both binary evaluation (DTI/link prediction)
        and multi-class evaluation (BioKDN-style).

        Args:
            use_multiclass (bool):
                False → binary (ROC-AUC, AUPR, PR-curve...)
                True  → multi-class (F1-macro, micro, kappa...)
        """

        if len(pos_pairs) == 0:
            return {}

        # Expand complex features if needed
        if np.iscomplexobj(features):
            features = np.concatenate([features.real, features.imag], axis=1)

        num_nodes = features.shape[0]

        if num_neg is None:
            num_neg = len(pos_pairs)

        # -----------------------------
        # Build negative pairs
        # -----------------------------
        pos_set = set(pos_pairs)
        neg_pairs = []
        tries = 0
        max_tries = num_neg * 50

        while len(neg_pairs) < num_neg and tries < max_tries:
            a = np.random.randint(0, num_nodes)
            b = np.random.randint(0, num_nodes)
            if (a, b) not in pos_set:
                neg_pairs.append((int(a), int(b)))
            tries += 1

        if len(neg_pairs) < num_neg:
            i = 0
            while len(neg_pairs) < num_neg:
                a = i % num_nodes
                b = (i + 1) % num_nodes
                if (a, b) not in pos_set:
                    neg_pairs.append((int(a), int(b)))
                i += 1

        # -----------------------------
        # Score positive / negative pairs
        # -----------------------------
        pos_scores = self.score_pairs(features, pos_pairs, edge_index=edge_index)
        neg_scores = self.score_pairs(features, neg_pairs, edge_index=edge_index)

        # Make y_true / y_pred
        y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        y_pred = np.concatenate([pos_scores, neg_scores])

        # -----------------------------
        # Evaluation mode selection
        # -----------------------------
        if not use_multiclass:
            metrics = self._binary_metrics(y_true, y_pred, k)
            print(
                f"[EVAL-BINARY] ROC-AUC={metrics['auc']:.4f} | AUPR={metrics['aupr']:.4f} "
                f"| F1={metrics['f1']:.4f} | Best-F1(PR)={metrics['best_f1_curve']:.4f} "
                f"| P@{k}={metrics['p_at_k']:.4f}"
            )

        else:
            metrics = self._multiclass_metrics(y_true, y_pred)
            print(
                f"[EVAL-MULTICLASS] F1-macro={metrics['f1_macro']:.4f} | "
                f"F1-micro={metrics['f1_micro']:.4f} | "
                f"Kappa={metrics['kappa']:.4f} | "
                f"Recall-macro={metrics['recall_macro']:.4f} | "
                f"Acc={metrics['accuracy']:.4f}"
            )

        # -----------------------------
        # Return results
        # -----------------------------
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "metrics": metrics
        }

    # ============================================================================
    # BINARY METRICS — (DTI, Link Prediction)
    # ============================================================================
    def _binary_metrics(self, y_true, y_pred, k):
        out = {}

        # AUC / AUPR
        try:
            out["auc"] = float(roc_auc_score(y_true, y_pred))
        except:
            out["auc"] = float("nan")

        try:
            out["aupr"] = float(average_precision_score(y_true, y_pred))
        except:
            out["aupr"] = float("nan")

        # F1 with threshold 0.5
        try:
            out["f1"] = float(f1_score(y_true, (y_pred > 0.5).astype(int)))
        except:
            out["f1"] = float("nan")

        # Precision–Recall curve
        try:
            prec, rec, thr = precision_recall_curve(y_true, y_pred)
            f1_curve = 2 * (prec * rec) / (prec + rec + 1e-12)
            out["best_f1_curve"] = float(np.max(f1_curve))
            out["precision_arr"] = prec
            out["recall_arr"] = rec
            out["pr_thresholds"] = thr
        except:
            out["best_f1_curve"] = float("nan")
            out["precision_arr"] = None
            out["recall_arr"] = None
            out["pr_thresholds"] = None

        # Precision@K
        try:
            K = min(k, len(y_pred))
            topk = np.argsort(-y_pred)[:K]
            out["p_at_k"] = float(y_true[topk].sum()) / float(K)
        except:
            out["p_at_k"] = float("nan")

        return out

    # ============================================================================
    # MULTICLASS METRICS — (BioKDN-style)
    # ============================================================================
    def _multiclass_metrics(self, y_true, y_pred):
        """
        Convert binary-like scores into multi-class decisions
        using an argmax thresholding.
        This is adapted to BioKDN-style evaluation.
        """

        out = {}

        # Create fake multi-class logits:
        # class 1 = score, class 0 = 1-score
        y_logits = np.vstack([1 - y_pred, y_pred]).T

        y_pred_class = np.argmax(y_logits, axis=1)

        # Macro / Micro F1
        out["f1_macro"] = float(f1_score(y_true, y_pred_class, average="macro"))
        out["f1_micro"] = float(f1_score(y_true, y_pred_class, average="micro"))

        # Cohen Kappa
        try:
            out["kappa"] = float(cohen_kappa_score(y_true, y_pred_class))
        except:
            out["kappa"] = float("nan")

        # Macro recall
        try:
            out["recall_macro"] = float(recall_score(y_true, y_pred_class, average="macro"))
        except:
            out["recall_macro"] = float("nan")

        # Accuracy
        try:
            out["accuracy"] = float(accuracy_score(y_true, y_pred_class))
        except:
            out["accuracy"] = float("nan")

        return out
    def _safe_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        out = {}
        if roc_auc_score is None:
            out['auc'] = float('nan')
            out['ap'] = float('nan')
            out['f1'] = float('nan')
            return out
        try:
            out['auc'] = float(roc_auc_score(y_true, y_pred))
        except Exception:
            out['auc'] = float('nan')
        try:
            out['ap'] = float(average_precision_score(y_true, y_pred))
        except Exception:
            out['ap'] = float('nan')
        try:
            out['f1'] = float(f1_score(y_true, (y_pred > 0.5).astype(int)))
        except Exception:
            out['f1'] = float('nan')
        return out

    def calibrate_threshold(self, features: np.ndarray, supervised_pairs: List[Tuple[int, int]],
                            edge_index: Optional[torch.Tensor] = None, desired_precision: Optional[float] = None) -> \
    Dict[str, float]:
        if precision_recall_curve is None:
            raise RuntimeError("scikit-learn precision_recall_curve is not available in this environment.")
        pos_pairs = supervised_pairs
        num_pos = len(pos_pairs)
        num_nodes = features.shape[0]
        neg_pairs = []
        pos_set = set(pos_pairs)
        tries = 0
        max_tries = num_pos * 50
        while len(neg_pairs) < num_pos and tries < max_tries:
            a = np.random.randint(0, num_nodes)
            b = np.random.randint(0, num_nodes)
            if (a, b) in pos_set:
                tries += 1
                continue
            neg_pairs.append((a, b))
            tries += 1
        y_pos = self.score_pairs(features, pos_pairs, edge_index=edge_index)
        y_neg = self.score_pairs(features, neg_pairs, edge_index=edge_index)
        y_true = np.concatenate([np.ones_like(y_pos), np.zeros_like(y_neg)])
        y_score = np.concatenate([y_pos, y_neg])
        prec, recall, thr = precision_recall_curve(y_true, y_score)
        ap = float(np.trapz(prec[::-1], recall[::-1])) if len(prec) and len(recall) else float('nan')
        if desired_precision is not None:
            idx = np.where(prec >= desired_precision)[0]
            if len(idx) == 0:
                chosen_thr = float(thr[-1]) if len(thr) else 0.9
                chosen_prec = float(prec[-1]) if len(prec) else float('nan')
                chosen_rec = float(recall[-1]) if len(recall) else float('nan')
            else:
                chosen_i = idx[-1]
                chosen_thr = float(thr[chosen_i - 1]) if chosen_i - 1 >= 0 and chosen_i - 1 < len(thr) else (
                    float(thr[-1]) if len(thr) else 0.9)
                chosen_prec = float(prec[chosen_i])
                chosen_rec = float(recall[chosen_i])
        else:
            f1 = (2 * prec * recall) / (prec + recall + 1e-12)
            best = np.nanargmax(f1)
            chosen_prec = float(prec[best])
            chosen_rec = float(recall[best])
            chosen_thr = float(thr[best - 1]) if best - 1 >= 0 and best - 1 < len(thr) else (
                float(thr[-1]) if len(thr) else 0.5)
        return {'threshold': chosen_thr, 'precision': chosen_prec, 'recall': chosen_rec, 'ap': ap}
