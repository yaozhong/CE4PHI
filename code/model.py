# preparing for the constrastive learning

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def pairwise_distance_eval(e_ph: torch.Tensor,
                           embed_bts: torch.Tensor,
                           metric: str = "chord") -> torch.Tensor:
    assert e_ph.dim() == 1
    assert embed_bts.dim() == 2
    D = e_ph.shape[0]
    M, Dh = embed_bts.shape
    assert D == Dh

    if metric == "chord":
        diff = embed_bts - e_ph              # [M,D]
        dist = diff.pow(2).sum(dim=-1).sqrt() / 2.0
        return dist

    elif metric == "euclidean":
        diff = embed_bts - e_ph
        dist = diff.pow(2).sum(dim=-1).sqrt()
        return dist

    elif metric == "cosine":
        e_ph_2d = e_ph.unsqueeze(0)              # [1,D]
        cos_sim = F.cosine_similarity(e_ph_2d, embed_bts)  # [M]
        dist = (1.0 - cos_sim) / 2.0
        return dist

    elif metric == "hyperbolic":
        eps = 1e-5
        x = e_ph.unsqueeze(0)     # [1,D]
        y = embed_bts             # [M,D]

        x_norm2 = (x ** 2).sum(dim=-1, keepdim=True)   # [1,1]
        y_norm2 = (y ** 2).sum(dim=-1, keepdim=True)   # [M,1]
        x_norm2 = torch.clamp(x_norm2, max=(1.0 - eps) ** 2)
        y_norm2 = torch.clamp(y_norm2, max=(1.0 - eps) ** 2)

        diff2 = torch.cdist(x, y) ** 2                 # [1,M]
        denom = (1.0 - x_norm2) @ (1.0 - y_norm2).t()  # [1,M]
        denom = torch.clamp(denom, min=eps)

        arg = 1.0 + 2.0 * diff2 / denom                # [1,M]
        arg = torch.clamp(arg, min=1.0 + eps)

        dist = torch.acosh(arg).squeeze(0)             # [M]
        return dist

    else:
        raise ValueError(f"Unknown metric: {metric}")


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]

        return loss


# matrix version for high efficient trainging, still using margin-based constrastive learning
class MatrixMarginContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0, normalize=True, metric="euclidean", reduction="mean"):
        super().__init__()
        self.margin = float(margin)
        self.normalize = normalize
        self.metric = metric
        assert reduction in {"mean", "sum", "none"}
        self.reduction = reduction

    def _pairwise_distance(self, z_anchor: torch.Tensor, z_cand: torch.Tensor) -> torch.Tensor:
        B, D = z_anchor.shape
        M, Dh = z_cand.shape
        assert D == Dh

        if self.metric == "euclidean":
            diff = z_anchor.unsqueeze(1) - z_cand.unsqueeze(0)   # [B,M,D]
            d = diff.pow(2).sum(dim=-1).sqrt()                  # [B,M]
            return d
        elif self.metric == "chord":
            diff = z_anchor.unsqueeze(1) - z_cand.unsqueeze(0)
            d = diff.pow(2).sum(dim=-1).sqrt() / 2.0
            return d
        elif self.metric == "cosine":
            a = z_anchor.unsqueeze(1)      # [B,1,D]
            b = z_cand.unsqueeze(0)        # [1,M,D]
            cos_sim = (a * b).sum(dim=-1)  # [B,M]
            d = (1.0 - cos_sim) / 2.0
            return d
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def forward(self, z_anchor: torch.Tensor, z_cand: torch.Tensor, pos_mask: torch.Tensor):
        """
        z_anchor: [B, D]
        z_cand:   [M, D]
        pos_mask: [B, M]  (1 for positive, 0 for negative)
        """
        B, D = z_anchor.shape
        M, Dh = z_cand.shape
        assert Dh == D
        assert pos_mask.shape == (B, M)

        pm = pos_mask.bool()
        pf = pm.float()
        nf = (~pm).float()

        # 1) optional normalization
        if self.normalize:
            z_anchor = F.normalize(z_anchor, dim=-1)
            z_cand   = F.normalize(z_cand, dim=-1)

        # 2) pairwise distance matrix [B,M]
        d = self._pairwise_distance(z_anchor, z_cand)  # [B,M]
        d2 = d.pow(2)

        pos_cnt = pf.sum(dim=1)                         # [B]
        has_pos = (pos_cnt > 0).float()
        pos_cnt_safe = pos_cnt.clamp_min(1.0)          

        pos_loss_i = ((d2 * pf).sum(dim=1) / pos_cnt_safe) * has_pos  # [B]

        neg_cnt = nf.sum(dim=1)                         # [B]
        has_neg = (neg_cnt > 0).float()
        neg_cnt_safe = neg_cnt.clamp_min(1.0)

        mdist = (self.margin - d).clamp_min(0.0)        
        neg_loss_mat = (mdist ** 2) * nf                


        neg_loss_i = (neg_loss_mat.sum(dim=1) / neg_cnt_safe) * has_neg  # [B]

        loss_vec = 0.5 * (pos_loss_i + neg_loss_i)      # [B]

        if self.reduction == "mean":
            return loss_vec.mean()
        elif self.reduction == "sum":
            return loss_vec.sum()
        else:
            return loss_vec
    

def distance(x1, x2, dist_type="euc"):
    if dist_type == "euc":
        dist = torch.cdist(x1, x2) ** 2
    if dist_type == "cos":
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        dist = cos(x1, x2)
    return dist


# encoder as the 
class cnn_module(nn.Module):
    def __init__(self, kernel_size=7, dr=0):
        super(cnn_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(4608, 512) # previous work 512

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.maxpool(x)

        x = self.fc1(torch.flatten(x, 1))
        x = F.normalize(x, p=2, dim=1)
        
        return x



class cnn_module_bac(nn.Module):
    def __init__(self, kernel_size=9, dr=0):
        super(cnn_module_bac, self).__init__()

        # basic part is the same as default CNN module
        self.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dr)

        self.fc1 = nn.Linear(3200, 2048)     # kernel_size=9
        self.fc_phi = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.maxpool(x)

        x = self.fc1(torch.flatten(x, 1))
        x = self.fc_phi(x)

        x = F.normalize(x, p=2, dim=1)
        return x


class TreePUInfoNCE(nn.Module):

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        reduction: str = "mean",

        # embedding distance metric
        metric: str = "chord",

        # tree kernel hyper-parameters
        tree_sigma: float = -1.0,   # <= 0: auto scale
        alpha: float = 2.0,         # (1 - Kbar)^alpha

        # phage–host–tree geometric regularizer
        lambda_ph_tree: float = 0.0,
        l2_lambda: float = 0.0,

        # weight
        margin = 0,
        tree_ce_eps = 0.02,  
           
    ):
        super().__init__()
        assert reduction in {"mean", "sum", "none"}
        assert metric in {"chord", "euclidean", "cosine", "hyperbolic"}

        self.t = float(temperature)
        self.normalize = normalize
        self.reduction = reduction
        self.metric = metric

        # tree-related
        self.tree_sigma = float(tree_sigma)
        self.alpha = float(alpha)
        self.lambda_ph_tree = float(lambda_ph_tree)
        self.l2_lambda = float(l2_lambda)

        self._cached_K = None      # [M,M] tree kernel
        self._cached_Dnorm = None  # [M,M] normalised tree distances

        # k-mer residual branch
        self.margin = margin

        self.tree_ce_eps = tree_ce_eps
        
    # ====== generic pairwise distance: [B,D] x [M,D] -> [B,M] ======
    def _pairwise_distance(self, z_anchor: torch.Tensor, z_cand: torch.Tensor) -> torch.Tensor:
        B, D = z_anchor.shape
        M, Dh = z_cand.shape
        assert D == Dh

        if self.metric == "chord":
            diff = z_anchor.unsqueeze(1) - z_cand.unsqueeze(0)   # [B,M,D]
            d = diff.pow(2).sum(dim=-1).sqrt() / 2.0            # [B,M]
            return d

        elif self.metric == "euclidean":
            diff = z_anchor.unsqueeze(1) - z_cand.unsqueeze(0)
            d = diff.pow(2).sum(dim=-1).sqrt()
            return d

        elif self.metric == "cosine":
            a = z_anchor.unsqueeze(1)        # [B,1,D]
            b = z_cand.unsqueeze(0)          # [1,M,D]
            cos_sim = (a * b).sum(dim=-1)    # [B,M]
            d = (1.0 - cos_sim) / 2.0
            return d

        elif self.metric == "hyperbolic":
            eps = 1e-5
            x = z_anchor
            y = z_cand

            x_norm2 = (x ** 2).sum(dim=-1, keepdim=True)
            y_norm2 = (y ** 2).sum(dim=-1, keepdim=True)
            x_norm2 = torch.clamp(x_norm2, max=(1.0 - eps) ** 2)
            y_norm2 = torch.clamp(y_norm2, max=(1.0 - eps) ** 2)

            diff2 = torch.cdist(x, y) ** 2

            denom = (1.0 - x_norm2) @ (1.0 - y_norm2).t()
            denom = torch.clamp(denom, min=eps)

            arg = 1.0 + 2.0 * diff2 / denom
            arg = torch.clamp(arg, min=1.0 + eps)

            d = torch.acosh(arg)
            return d

        else:
            raise ValueError(f"Unknown metric type: {self.metric}")

    def _spearman_1d(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

        ra = torch.argsort(torch.argsort(a))
        rb = torch.argsort(torch.argsort(b))

        ra = ra.float()
        rb = rb.float()

        ra = ra - ra.mean()
        rb = rb - rb.mean()

        denom = (ra.std(unbiased=False) * rb.std(unbiased=False)).clamp_min(1e-8)
        return (ra * rb).mean() / denom

    def _per_phage_corr_all(
        self,
        Dnorm: torch.Tensor,         
        pos_mask: torch.Tensor,      
        kmer_scores: torch.Tensor,   
        d_emb: torch.Tensor,         
        exclude_pos: bool = True,    
    ):

        device = Dnorm.device
        pos_mask = pos_mask.to(device)
        kmer_scores = kmer_scores.to(device).float()
        d_emb = d_emb.to(device).float()

        pf = pos_mask.float()                     
        cnt = pf.sum(dim=1, keepdim=True).clamp_min(1.0)

        d_tree_avg = (pf @ Dnorm) / cnt           # [B,M]

        d_kmer = kmer_scores                      # [B,M]
        d_e = d_emb                               # [B,M]

        B, M = d_kmer.shape
        corr_emb_tree_list = []
        corr_emb_kmer_list = []
        corr_tree_kmer_list = []   

        for i in range(B):
            mask = torch.ones(M, dtype=torch.bool, device=device)
            if exclude_pos:
                mask = mask & (~pos_mask[i])      

            x_tree = d_tree_avg[i, mask]          
            x_kmer = d_kmer[i, mask]
            x_emb  = d_e[i, mask]

            if x_tree.numel() < 2:
                nan_val = torch.tensor(float("nan"), device=device)
                corr_emb_tree_list.append(nan_val)
                corr_emb_kmer_list.append(nan_val)
                corr_tree_kmer_list.append(nan_val)
                continue


            corr_emb_tree_list.append(_spearman_1d(x_emb,  x_tree))
            corr_emb_kmer_list.append(_spearman_1d(x_emb,  x_kmer))
            corr_tree_kmer_list.append(_spearman_1d(x_tree, x_kmer))

        corr_emb_tree = torch.stack(corr_emb_tree_list, dim=0)   # [B]
        corr_emb_kmer = torch.stack(corr_emb_kmer_list, dim=0)   # [B]
        corr_tree_kmer = torch.stack(corr_tree_kmer_list, dim=0) # [B]

        def _stats_from_vec(vec: torch.Tensor):
            valid = vec[~torch.isnan(vec)]
            if valid.numel() > 0:
                    return dict(
                    mean=float(valid.mean().item()),
                    median=float(valid.median().item()),
                    std=float(valid.std(unbiased=False).item()),
                    min=float(valid.min().item()),
                    max=float(valid.max().item()),
                    valid_n=int(valid.numel()),
                    total_n=int(vec.numel()),
                )
            else:
                return dict(
                    mean=float("nan"), median=float("nan"), std=float("nan"),
                    min=float("nan"), max=float("nan"),
                    valid_n=0, total_n=int(vec.numel()),
                )

        result = dict(
            corr_emb_tree=corr_emb_tree,
            stats_emb_tree=_stats_from_vec(corr_emb_tree),

            corr_emb_kmer=corr_emb_kmer,
            stats_emb_kmer=_stats_from_vec(corr_emb_kmer),

            corr_tree_kmer=corr_tree_kmer,
            stats_tree_kmer=_stats_from_vec(corr_tree_kmer),
        )
        return result


    # ====== build tree kernel & cached normalised distances ======
    def _build_tree_kernel(self, tree_dists: torch.Tensor) -> torch.Tensor:
        td = tree_dists.float()
        M = td.shape[0]
        device = td.device
        eye = torch.eye(M, dtype=torch.bool, device=device)

        # (1) scale based on median of finite off-diagonal distances
        finite_raw_mask = torch.isfinite(td) & (~eye)
        if finite_raw_mask.any():
            raw_vals = td[finite_raw_mask]
            q50_raw = torch.quantile(raw_vals, q=0.50).item()
            scale = q50_raw if q50_raw > 0 else (raw_vals.mean().item() + 1e-6)
        else:
            scale = 1.0

        # (2) fill inf using block diameters (same as previous implementation)
        adj = torch.isfinite(td) & (~eye)
        visited = torch.zeros(M, dtype=torch.bool, device=device)
        comps = []
        for i in range(M):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = [i]
            while stack:
                u = stack.pop()
                nbrs = torch.where(adj[u])[0]
                for v in nbrs.tolist():
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
                        comp.append(v)
            comps.append(comp)

        def _block_diameter(idxs):
            if len(idxs) < 2:
                return 0.0
            sub = td[idxs][:, idxs]
            mask = torch.isfinite(sub) & (~torch.eye(len(idxs), dtype=torch.bool, device=device))
            if mask.any():
                return float(sub[mask].max().item())
            return 0.0

        diameters = [_block_diameter(comp) for comp in comps]
        global_max = max(diameters) if diameters else 0.0

        ratio = 0.20
        node2block = torch.empty(M, dtype=torch.long, device=device)
        for b_id, comp in enumerate(comps):
            node2block[torch.tensor(comp, dtype=torch.long, device=device)] = b_id

        td_fill = td.clone()
        bad = (~torch.isfinite(td_fill)) & (~eye)
        rows, cols = torch.where(bad)
        for i, j in zip(rows.tolist(), cols.tolist()):
            bi = int(node2block[i].item())
            bj = int(node2block[j].item())
            if bi == bj:
                base = max(diameters[bi], global_max)
            else:
                base = max(diameters[bi], diameters[bj], global_max) * (1.0 + ratio)
            td_fill[i, j] = base

        td_fill = (td_fill + td_fill.t()) * 0.5
        td_fill.fill_diagonal_(0.0)

        # (3) normalised distances used by geometric regulariser
        d_norm = td_fill / (scale + 1e-12)
        self._cached_Dnorm = d_norm

        # (4) Laplacian kernel K = exp(- d_norm / sigma_eff)
        K = torch.zeros_like(td_fill)

        if self.tree_sigma <= 0:
            sigma_eff = 0.7
        else:
            sigma_eff = float(self.tree_sigma) / (scale + 1e-12)

        denom = sigma_eff + 1e-12
        finite_mask = torch.isfinite(td_fill) & (~eye)
        if finite_mask.any():
            K[finite_mask] = torch.exp(-d_norm[finite_mask] / denom)
            K[finite_mask] = torch.clamp(K[finite_mask], min=1e-4)

        K.fill_diagonal_(1.0)
        K = 0.5 * (K + K.t())

        sigma_real = sigma_eff * scale
        print(f"[TreeKernel] scale_raw(q50)={scale:.4f}, "
              f"tree_sigma={self.tree_sigma}, "
              f"sigma_eff(d_norm)={sigma_eff:.4f}, "
              f"sigma_real(raw)={sigma_real:.4f}")

        return K


    def _loss_one_direction(
        self,
        z_anchor: torch.Tensor,            # [B, D]
        z_cand: torch.Tensor,              # [M, D]
        pos_mask: torch.Tensor,            # [B, M]
        tree_dists: torch.Tensor = None,   # [M,M]
        tree_mask: torch.Tensor = None,    # [B,M] or None (optional; can still be used to shape logits)
        kmer_scores: torch.Tensor = None,  # optional, unused here
        ):
    
        B, D = z_anchor.shape
        M, Dh = z_cand.shape
        assert Dh == D
        assert pos_mask.shape == (B, M)

        # 1) optional normalization
        if self.normalize:
            z_anchor = F.normalize(z_anchor, dim=-1)
            z_cand = F.normalize(z_cand, dim=-1)

        pm = pos_mask.bool()  # [B,M]
        pf = pm.float()

        # 2) embedding distance -> score -> logits
        d_emb = self._pairwise_distance(z_anchor, z_cand)  # [B,M]
        score_emb = -(d_emb ** 2)                          # [B,M]

        tau = float(self.t)
        gamma = float(getattr(self, "margin", 0.0))

        logits_total = score_emb / tau                     # [B,M]
        if gamma > 0.0:
            logits_total = logits_total - (gamma / tau) * pf   # subtract margin on positives

        # 3) model log-prob
        log_p = logits_total - torch.logsumexp(logits_total, dim=1, keepdim=True)  # [B,M]

        # 4) build target distribution q
        eps = float(getattr(self, "tree_ce_eps", 0.02))  # 0.02~0.10 recommended
        #eps = max(0.0, min(0.49, eps))                   # keep sane: positives remain dominant

        q = torch.zeros_like(logits_total)               # [B,M]

        # 4.1 positives: uniform mass (1-eps)
        pos_cnt = pm.sum(dim=1, keepdim=True).clamp_min(1)        # [B,1]
        q_pos = (1.0 - eps) / pos_cnt.float()                     # [B,1]
        q = q + pf * q_pos                                        # [B,M]

        # 4.2 negatives: eps mass by phylogenetic distance to positives
        if tree_dists is not None and eps > 0.0:
            # cache normalized distance matrix
            # reuse your existing cache builder to get Dnorm in [0,1] if available
            Dnorm = None
            if (
                getattr(self, "_cached_Dnorm", None) is None
                or getattr(self, "_cached_Dnorm", None).shape != tree_dists.shape
                or getattr(self, "_cached_Dnorm", None).device != tree_dists.device
            ):
                # build kernel also sets _cached_Dnorm in your codebase
                _ = self._build_tree_kernel(tree_dists)
            Dnorm = self._cached_Dnorm  # [M,M] normalized distances

            if Dnorm is not None:
                Dnorm = Dnorm.to(logits_total.device)

                # sigma for exp(-d/sigma); if you already have self.tree_sigma, reuse it
                sigma = float(getattr(self, "tree_ce_sigma", -1.0))
                if sigma <= 0:
                    ts = float(getattr(self, "tree_sigma", -1.0))
                    sigma = ts if ts > 0 else 0.5
                sigma = max(1e-6, sigma)

                # compute d_tree[b, j] = min_{p in pos(b)} Dnorm[j, p]
                # B is usually small (<=64), loop over B is fine and memory-safe
                d_tree = logits_total.new_empty((B, M))  # [B,M]
                for b in range(B):
                    pos_idx = pm[b].nonzero(as_tuple=False).view(-1)
                    if pos_idx.numel() == 0:
                        # no positives: give uniform neg distribution (fallback)
                        d_tree[b].fill_(0.0)
                    else:
                        d_tree[b] = Dnorm[:, pos_idx].min(dim=1).values


                # affinity on negatives
                a = torch.exp(-(d_tree / sigma))                           # [B,M]
                a = a * (~pm).float()                                    # keep negatives only

                # optional: apply tree_mask to restrict which negatives get mass, none -> all 
                if tree_mask is not None:
                    a = torch.where(tree_mask.bool(), a, a.new_zeros(()))

                a_sum = a.sum(dim=1, keepdim=True).clamp_min(1e-12)
                q_neg = eps * (a / a_sum)                                # [B,M]
                q = q + q_neg


        # 5) unified CE loss
        loss_vec = -(q * log_p).sum(dim=1)                               # [B]
        loss_vec = torch.nan_to_num(loss_vec, nan=0.0, posinf=0.0, neginf=0.0)

        if self.reduction == "mean":
            return loss_vec.mean()
        elif self.reduction == "sum":
            return loss_vec.sum()
        else:
            return loss_vec    


    # ====== forward: p->h InfoNCE + optional L2 regularisation ======
    def forward(
        self,
        z_p: torch.Tensor,
        z_h: torch.Tensor,
        pos_mask: torch.Tensor,
        tree_dists: torch.Tensor = None,
        tree_mask: torch.Tensor = None,
        kmer_scores: torch.Tensor = None,
    ):
        loss_p2h = self._loss_one_direction(
            z_anchor=z_p,
            z_cand=z_h,
            pos_mask=pos_mask,
            tree_dists=tree_dists,
            tree_mask=tree_mask,
            kmer_scores=kmer_scores,
        )

        if self.l2_lambda > 0.0:
            l2_loss = (z_p.pow(2).mean() + z_h.pow(2).mean()) / 2.0
            return loss_p2h + self.l2_lambda * l2_loss
        else:
            return loss_p2h

#----------------------------------------------------------------------------

def _upper_tri_vec(D: torch.Tensor) -> np.ndarray:
    """Return upper-triangle (k=1) as 1D numpy array."""
    assert D.ndim == 2 and D.shape[0] == D.shape[1]
    M = D.shape[0]
    iu = torch.triu_indices(M, M, offset=1, device=D.device)
    v = D[iu[0], iu[1]]
    return v.detach().float().cpu().numpy()

def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation implemented via rank + Pearson (no scipy)."""
    x = np.asarray(x)
    y = np.asarray(y)
    # ranks (average ties not handled; usually ok for continuous distances)
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float(np.mean(x * y))

def mantel_r(D1: torch.Tensor, D2: torch.Tensor, perms: int = 0, seed: int = 0):
    """
    Mantel r: Pearson correlation between upper-triangular vectors of two distance matrices.
    If perms > 0, also returns permutation p-value (two-sided).
    """
    v1 = _upper_tri_vec(D1)
    v2 = _upper_tri_vec(D2)
    r_obs = _pearson_r(v1, v2)

    if perms <= 0:
        return r_obs, None

    rng = np.random.default_rng(seed)
    M = D1.shape[0]
    ge = 0
    for _ in range(perms):
        perm = rng.permutation(M)
        D2p = D2[perm][:, perm]
        r_p, _ = mantel_r(D1, D2p, perms=0)  # just compute r
        if abs(r_p) >= abs(r_obs):
            ge += 1
    p = (ge + 1) / (perms + 1)
    return r_obs, p

@torch.no_grad()
def chord_dist_matrix_from_embeddings(Z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    chord distance: 0.5 * || ẑ_i - ẑ_j ||_2
    Z: [M, D]
    """
    Z = F.normalize(Z, dim=-1)
    # d^2 = ||a-b||^2 = 2 - 2cos
    cos = Z @ Z.t()
    d2 = (2.0 - 2.0 * cos).clamp_min(0.0)
    d = 0.5 * torch.sqrt(d2 + eps)
    return d

@torch.no_grad()
def chord_dist_matrix_from_fcgr(bts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    FCGR/k-mer chord-like distance: (cdist(normalized_flat)/2)
    bts: [M, ...] FCGR tensors
    """
    M = bts.shape[0]
    bt_flat = bts.view(M, -1)
    bt_flat = F.normalize(bt_flat, dim=-1)
    d = torch.cdist(bt_flat, bt_flat, p=2) / 2.0
    return d

@torch.no_grad()
def tree_alignment_metrics(D_tree: torch.Tensor, D_other: torch.Tensor, mantel_perms: int = 0, seed: int = 0):
    """
    returns spearman_rho, mantel_r, mantel_p
    """
    v_tree = _upper_tri_vec(D_tree)
    v_other = _upper_tri_vec(D_other)
    rho = _spearman_r(v_other, v_tree)
    mr, mp = mantel_r(D_tree, D_other, perms=mantel_perms, seed=seed)
    return rho, mr, mp

