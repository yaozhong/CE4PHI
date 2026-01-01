# data loading for the paired samples

import torch, io
from torch.utils.data import Dataset, DataLoader
from functools import partial

from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from fasta2CGR import *
import numpy as np
from PIL import Image
from typing import Dict, List
import pandas as pd

def get_data_host_sets(file_name_list):

	labels = []

	for fn in file_name_list:
		s_in = open(fn)
		
		for line in s_in:
			line_info = line.strip("\n")
			#labels.append(line_info)
			labels.extend(line_info.split(","))

		s_in.close()

	return list(set(labels))


def get_label_map(species_file):
    s_in = open(species_file)
    spiece_dic, label2species = {}, []

    for idx, line in enumerate(s_in):
        line_info = line.strip("\n")
        specie_label = line_info.split("\t")[1]
        spiece_dic[specie_label] = idx
        label2species.append(specie_label)

    s_in.close()

    return spiece_dic, label2species

def load_host_label(file_name, s2l_dic):

    if file_name == '':
        return []

    labels = []
    with open(file_name) as s_in:
        for line in s_in:
            line_info = line.strip("\n").strip()
            if line_info == "":
                labels.append([])
                continue

            parts = [p.strip() for p in line_info.split(",") if p.strip()]

            host_ids = []
            for p in parts:

                if p in s2l_dic:
                    host_ids.append(s2l_dic[p])
                    continue

                p_norm = " ".join(p.split())
                if p_norm in s2l_dic:
                    host_ids.append(s2l_dic[p_norm])
                    continue

                continue

            labels.append(host_ids)

    return labels

# get all host representation of the host_fa_file according to the keepList.
def get_host_fa(s2l_dic, host_fa_file, kmer, keep_list=[]):

	l2fa = {}
	# loading host fa information into 
	wgs = Fasta(host_fa_file)
	for bn in wgs.keys():
		# filtering labels not in the keep_list
		check = bn.replace("_", " ")
		if len(keep_list) > 0 and check not in keep_list:
			continue

		seq = wgs[bn][:].seq
		fc = count_kmers(seq, kmer)
		f_prob = probabilities(seq, fc, kmer)
		chaos_k = chaos_game_representation(f_prob, kmer)

		label = s2l_dic[bn.replace("_", " ")]
		l2fa[label] = chaos_k

	return l2fa


def my_collate_fn(batch, kmer, l2fa):

    images, hosts, labels = [], [], []

    for name, seq, label_ids in batch:
        if isinstance(label_ids, (int, np.integer)):
            pos_set = {int(label_ids)}
        elif isinstance(label_ids, (list, tuple, np.ndarray)):
            pos_set = set(int(x) for x in label_ids)
        else:
            pos_set = set()

        fc = count_kmers(seq, kmer)
        f_prob = probabilities(seq, fc, kmer)
        chaos_k = chaos_game_representation(f_prob, kmer)
        img = chaos_k

        for l in l2fa.keys():
            if l in pos_set:
                labels.append(1)
            else:
                labels.append(0)

            images.append(img)
            hosts.append(l2fa[l])

    return np.array(images), np.array(hosts), np.array(labels)


# standard approach of loading data for valdiation and testing, not relationship with multiple-host cases
def my_collate_fn2(batch, kmer):
    images, labels, phage_name_list = [], [], []

    for name, seq, label_ids in batch:
        phage_name = name
        labels.append(label_ids)

        # FCGR
        fc = count_kmers(seq, kmer)
        f_prob = probabilities(seq, fc, kmer)
        chaos_k = chaos_game_representation(f_prob, kmer)
        img = chaos_k

        images.append(img)
        phage_name_list.append(phage_name)

    return np.array(images), labels, phage_name_list



# same implementation as "previous one"
def my_collate_fn_infoNCE(batch, kmer: int, l2fa: Dict[int, np.ndarray]):
    """
    InfoNCE / TreePU: return phage images + shared host bank + multi-positive mask.
    Returns:
      imgs_ph: [B, H, W]
      imgs_bt: [M, H, W]
      pos_mask: [B, M] (0/1)
    """
    host_ids = list(sorted(l2fa.keys()))
    host_index = {hid:i for i,hid in enumerate(host_ids)}
    host_bank = np.stack([l2fa[h] for h in host_ids], axis=0)  # [M,H,W]

    ph_imgs, pos_mask = [], []
    for name, seq, gold_ids in batch:
        fc = count_kmers(seq, kmer)
        f_prob = probabilities(seq, fc, kmer)
        ph_imgs.append(chaos_game_representation(f_prob, kmer))
        row = np.zeros((len(host_ids),), dtype=np.int64)
        if isinstance(gold_ids, (list, tuple)):
            for gid in gold_ids:
                if gid in host_index:
                    row[host_index[gid]] = 1
        elif isinstance(gold_ids, int):
            if gid in host_index:
                row[host_index[gid]] = 1
        pos_mask.append(row)

    return np.array(ph_imgs), host_bank, np.stack(pos_mask, axis=0)

class fasta_dataset(Dataset):
	def __init__(self, file_name, label_file, host_file):

		wgs = Fasta(file_name)
		self.name = []
		self.seq = []

		self.s2l_dic, self.l2s = get_label_map(label_file)

		# sequence process an put it in queue
		for pn in wgs.keys():
			self.name.append(pn)
			self.seq.append(wgs[pn][:].seq)

		self.label = load_host_label(host_file, self.s2l_dic)	

	def __len__(self):
		return	len(self.name)

	def __getitem__(self, idx):
		if(len(self.label) == 0):
			return self.name[idx], self.seq[idx], []
		return self.name[idx], self.seq[idx], self.label[idx]

	def get_s2l_dic(self):
		return self.s2l_dic

	def get_l2s_dic(self):
		return self.l2s




def load_gtdb_taxonomy(
    tsv_path: str,
    key_lower: bool = True,
) -> Dict[str, Dict[str, str]]:

    rank_prefix = {"domain": "d__","phylum": "p__","class": "c__","order": "o__","family": "f__","genus": "g__","species": "s__",}
    species_tax: Dict[str, Dict[str, str]] = {}

    with open(tsv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split(maxsplit=1)

            if len(parts) < 2:
                continue

            tax_str = parts[1].strip()
            if len(parts) > 2:
                extra = " ".join(p.strip() for p in parts[2:] if p.strip())
                if extra:
                    tax_str = tax_str + " " + extra

            info = {k: "NA" for k in rank_prefix.keys()}

            for token in tax_str.split(";"):
                token = token.strip()
                if not token:
                    continue
                for rk, pre in rank_prefix.items():
                    if token.startswith(pre):

                        val = token.split("__", 1)[1]
                        val = val.split("_")[0].strip()

                        info[rk] = val.lower() if key_lower else val
                        break

            sp = info["species"]
            if sp == "NA" or sp == "" or sp is None:
                continue

            species_key = sp  
            if species_key in species_tax:
                pass

            species_tax[species_key] = info

    return species_tax

# for the genus level mapping ,using the simple split approach 
def build_level_ids(
    host_species_names: List[str],  
    gtdb_tax: Dict[str, Dict[str, str]],
    level: str,        # 'genus' | 'family' | 'order' | 'class' | 'phylum' | 'domain'
    device: torch.device = torch.device("cpu"),
):
    """
    Return:
        level_ids: LongTensor [M]  - integer label per host for given level
        label_map: Dict[int, str]  - int -> original text label
    """
    assert level in {"domain", "phylum", "class", "order", "family", "genus"}

    labels: List[str] = []

    for name in host_species_names:
        if level == "genus":
            lbl = name.split(" ")[0]
        else:
            info = gtdb_tax.get(name, {})
            #lbl = info.get(level, "NA")
            #print(name, info)
            lbl = info.get(level, name) # if not found, using the original species label to avoid ambiuity mapping .
            
        labels.append(lbl)

    uniq = sorted(set(labels))
    # print(uniq)
    # print(len(uniq))
    str2id = {s: i for i, s in enumerate(uniq)}
    id2str = {i: s for s, i in str2id.items()}

    level_ids = torch.tensor(
        [str2id[s] for s in labels],
        dtype=torch.long,
        device=device,
    )

    return level_ids, id2str


def build_tree_mask_from_level(
    gold_idx: torch.Tensor,   # [B], long
    level_ids: torch.Tensor,  # [M], long, same order as host_species_names
) -> torch.Tensor:
    """
    tree_mask[b, j] = True  <=> host j has the same level label
    as the gold host of phage b.

    Example:
        level_ids = genus_ids / family_ids / ...
    """
    device = level_ids.device
    B = gold_idx.shape[0]
    M = level_ids.shape[0]
    assert gold_idx.dtype == torch.long

    gold_level = level_ids[gold_idx]                 # [B]
    # broadcast: [B,1] vs [1,M]
    tree_mask = (gold_level.unsqueeze(1) == level_ids.unsqueeze(0))  # [B,M]
    return tree_mask.to(device)  


def make_tree_mask_level(
    level_keyword: str,           # 'none' | 'all' | 'genus' | 'family' | ...
    pos_mask: torch.Tensor,       # [B,M] (允许 multi-positive)
    level_ids_dict: Dict[str, torch.Tensor],  # {"genus": genus_ids, "family": family_ids, ...}
    device: torch.device,
) -> torch.Tensor | None:
   
    B, M = pos_mask.shape

    if level_keyword is None:
        return None

    level_keyword = level_keyword.lower()

    if level_keyword == "none":
        return None

    if level_keyword == "all":
        return torch.ones(B, M, dtype=torch.bool, device=device)

    if level_keyword not in level_ids_dict:
        return None

    # core part
    level_ids = level_ids_dict[level_keyword]  # [M]
    #print(level_ids)
    level_ids = level_ids.to(device)

    pm = pos_mask.to(device).bool()            # [B,M]

    # same_level[j,k] = (level_ids[j] == level_ids[k])  , shape [M,M]
    same_level = (level_ids.unsqueeze(1) == level_ids.unsqueeze(0))  # [M,M]

    same_level_exp = same_level.unsqueeze(0).expand(B, M, M)   # [B,M,M]
    pm_exp = pm.unsqueeze(1)                                  # [B,1,M]

    # tree_mask[b,j] = ∃k: pm[b,k] && same_level[j,k]
    tree_mask = (same_level_exp & pm_exp).any(dim=2)          # [B,M] bool

    return tree_mask

    
#-----------------------------------------------------------------------------------
def replace_first_underscore_with_space(name: str) -> str:
    if isinstance(name, str) and "_" in name:
        parts = name.split("_", 1)
        return parts[0] + " " + parts[1] if len(parts) > 1 else name
    return name


def _norm_key(x: str) -> str:
    x = str(x)
    x = x.replace("_", " ")
    x = " ".join(x.strip().split())
    return x.lower()

def build_aligned_tree_dist_tensor(tree_df: pd.DataFrame,
                                   l2fa_filter: dict,
                                   s2l: dict,
                                   device: str,
                                   fill_inf: float = np.inf) -> torch.Tensor:
    host_order_ids = list(map(str, l2fa_filter.keys()))
    M = len(host_order_ids)
    l2s = {str(v): str(k) for k, v in (s2l or {}).items()} if isinstance(s2l, dict) else {}

    idx_raw = list(map(str, tree_df.index))
    norm2raw = {_norm_key(k): k for k in idx_raw}
    idx_raw_set = set(idx_raw)
    host_order_names = [l2s.get(hid, hid) for hid in host_order_ids]

    direct_hits = sum(h in idx_raw_set for h in host_order_ids)
    name_hits   = sum(n in idx_raw_set for n in host_order_names)
    direct_hits_norm = sum(_norm_key(h) in norm2raw for h in host_order_ids)
    name_hits_norm   = sum(_norm_key(n) in norm2raw for n in host_order_names)
    score = {"id":direct_hits,"name":name_hits,"id_norm":direct_hits_norm,"name_norm":name_hits_norm}
    chosen = max(score, key=score.get)

    def map_to_csv_key(hid: str) -> str|None:
        if chosen == "id":
            return hid if hid in idx_raw_set else None
        elif chosen == "name":
            nm = l2s.get(hid, hid);  return nm if nm in idx_raw_set else None
        elif chosen == "id_norm":
            nk = _norm_key(hid);     return norm2raw.get(nk, None)
        else:
            nm = l2s.get(hid, hid);  nk = _norm_key(nm);  return norm2raw.get(nk, None)

    mapped_keys = [map_to_csv_key(h) for h in host_order_ids]
    hit_count = sum(k is not None for k in mapped_keys)

    td_np = np.full((M, M), fill_inf, dtype=np.float32)
    if hit_count == 0:
        print("[tree][warn] no host aligned; using all-inf (diag=0).")
    else:
        for i, ki in enumerate(mapped_keys):
            if ki is None: continue
            row = tree_df.loc[ki]
            for j, kj in enumerate(mapped_keys):
                if kj is None: continue
                try: td_np[i, j] = float(row[kj])
                except: pass
    np.fill_diagonal(td_np, 0.0)
    print(f"[tree] alignment hits={hit_count}/{M} ({100.0*hit_count/max(M,1):.1f}%)")
    return torch.from_numpy(td_np).to(device)



def fix_tree_distance_matrix(tree_dists):

    if isinstance(tree_dists, pd.DataFrame):
        td = torch.tensor(tree_dists.values, dtype=torch.float32)
    elif isinstance(tree_dists, np.ndarray):
        td = torch.tensor(tree_dists, dtype=torch.float32)
    elif isinstance(tree_dists, torch.Tensor):
        td = tree_dists.clone().float()
    else:
        raise ValueError("Unsupported input type for tree_dists")

    M = td.shape[0]
    device = td.device
    eye = torch.eye(M, dtype=torch.bool, device=device)

    td = td.clone()
    td[td == float("-inf")] = float("inf")

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

    def block_diameter(idxs):
        if len(idxs) < 2:
            return 0.0
        sub = td[idxs][:, idxs]
        mask = torch.isfinite(sub) & (~torch.eye(len(idxs), dtype=torch.bool, device=device))
        if mask.any():
            return float(sub[mask].max().item())
        return 0.0

    diameters = [block_diameter(c) for c in comps]
    global_max = max(diameters) if diameters else 0.0

    node2block = torch.empty(M, dtype=torch.long, device=device)
    for b_id, comp in enumerate(comps):
        node2block[torch.tensor(comp, dtype=torch.long, device=device)] = b_id

    td_fill = td.clone()
    rows, cols = torch.where(~torch.isfinite(td_fill) & (~eye))

    ratio = 0.10  

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

    return td_fill
