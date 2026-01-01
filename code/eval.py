# prediction 
import argparse, sys
import torch
from data_loading import *
from model import *

import numpy as np
from sklearn import metrics
import torch.nn.functional as F  


def test(phage_enc, host_enc, cached_ph, l2fa, cached_label, device, threshold=1, verbose=True, metric: str = "chord"):

    phage_enc.eval(); host_enc.eval()

    host_vec = np.array([l2fa[l] for l in l2fa.keys()])
    host_vec = torch.tensor(host_vec, dtype=torch.float32).to(device)
    host_vec = torch.unsqueeze(host_vec, dim=1)  # [M, 1, H, W]

    embed_bts = host_enc(host_vec)   # [M, D]
    embed_bts = F.normalize(embed_bts, dim=-1)

    label_list = list(l2fa.keys())  # host label id 
    label_pos = {lab: idx for idx, lab in enumerate(label_list)}  # 

    pred_dist_list = []  # 
    gold_list = []       # 

    total_gold = 0.0
    total_hit = 0.0

    total_batch = len(cached_ph)
    tmp_save = []

    with torch.no_grad():
        for i in range(total_batch):
            phs = cached_ph[i].to(device)     # [B, 1, H, W]
            batch_labels = cached_label[i]    # 
            
            embed_phs = phage_enc(phs)        # [B, D]
            embed_phs = F.normalize(embed_phs, dim=-1)

            for j, e_ph in enumerate(embed_phs):
                gold = batch_labels[j]

                # list[int]
                if isinstance(gold, (int, np.integer)):
                    gold_ids = [int(gold)]
                elif isinstance(gold, (list, tuple, np.ndarray)):
                    gold_ids = [int(x) for x in gold]
                else:
                    continue

                gold_ids = sorted(set(gold_ids))
                if len(gold_ids) == 0:
                    continue

                dist = pairwise_distance_eval(e_ph, embed_bts, metric=metric)  # [M]

                dist_cpu = dist.to("cpu").detach().numpy()
                pred_dist_list.append(dist_cpu)
                gold_list.append(gold_ids)

                # pay attention to the argsort, both torch and np is not stable
                sorted_idx = np.argsort(dist_cpu, kind="stable")
                sorted_host_ids = [label_list[int(idx)] for idx in sorted_idx]

                G = len(gold_ids)
                total_gold += G

                topk_pred_ids = set(sorted_host_ids[:G])
                gold_set = set(gold_ids)

                hit = len(topk_pred_ids & gold_set)  # 
                total_hit += hit
                tmp_save.append((topk_pred_ids, gold_set, hit))

    acc_multi = total_hit / total_gold if total_gold > 0 else 0.0

    if verbose:
        print(f"@ Multi-host aware accuracy (sum(H)/sum(G)) = {acc_multi:.4f}", file=sys.stderr)

    return acc_multi, pred_dist_list, gold_list


# prediction without provide gold standard
def predict(phage_enc, host_enc, cached_ph, l2fa, device, metric= "chord"):

    phage_enc.eval(); host_enc.eval()

    # first generate embeddings for the host.
    host_vec = np.array([l2fa[l] for l in l2fa.keys()])
    host_vec = torch.tensor(host_vec, dtype=torch.float32).to(device)
    host_vec = torch.unsqueeze(host_vec, dim=1)

    embed_bts = host_enc(host_vec)
    embed_bts = F.normalize(embed_bts, dim=-1)

    label_list = list(l2fa.keys())

    pred_list, pred_dist = [], []
    total_batch = len(cached_ph)

    with torch.no_grad():
        for i in range(total_batch):
            phs = cached_ph[i]
            phs = phs.to(device)
            embed_phs = phage_enc(phs)
            embed_phs = F.normalize(embed_phs, dim=-1)

            # local calculation of the distance scores
            for e_ph in embed_phs:
                dist = pairwise_distance_eval(e_ph, embed_bts, metric=metric)  # [M]
                pred_dist.append(dist.to("cpu").detach().numpy())
                
    return pred_dist


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='<Prediciton of using contrastive learning for the phage-host identification>')

    parser.add_argument('--model',       default="CNN", type=str, required=True, help='contrastive learning encoding model')
    parser.add_argument("--enc_mode", type=str, default="seperate", choices=["share", "seperate"],help="share: phage/host use the same encoder; separate: two encoders")
    parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
    parser.add_argument('--device',       default="cuda:0", type=str, required=False, help='GPU Device(s) used for training')

    parser.add_argument('--kmer',       default=6,       type=int, required=True, help='kmer length')
    parser.add_argument('--batch_size' ,default=64,      type=int,  required=False, help="batch_size of the training.")
    parser.add_argument('--workers',     default=8,       type=int, required=False, help='number of worker for data loading')

    # data related input
    parser.add_argument('--host_fa',   default="",  type=str, required=True, help='Host fasta files')
    parser.add_argument('--host_list', default="",  type=str, required=True, help='Host species list')
    parser.add_argument('--test_phage_fa', default="",   type=str, required=True, help='Test phage fasta file')
    parser.add_argument('--test_host_gold', default="",  type=str, required=False, help='Infecting host gold list')

    parser.add_argument('--use_train_bn', action='store_true', required=False, help='use the batch norm statistics in the train')

    parser.add_argument(
        '--metric',
        type=str,
        default="chord",
        choices=["chord", "euclidean", "cosine", "hyperbolic"],
        help="Distance metric used at evaluation time; should match training.",
    )
    
    args = parser.parse_args()

    kmer = args.kmer
    num_workers = args.workers
    batch_size = args.batch_size

    # preparing the test data for the evaluation.
    ## 1. loading model
    if args.model == "CNN":
        if args.enc_mode == "share":
            shared_enc = cnn_module(7, 0)
            #shared_enc = cnn_module_bac(9)	
            shared_enc.load_state_dict(torch.load(args.model_dir))
            shared_enc = shared_enc.to(args.device)
            phage_enc = shared_enc
            host_enc = shared_enc
        elif args.enc_mode == "seperate":
            phage_enc = cnn_module(7, 0)
            host_enc = cnn_module_bac(9, 0)
            phage_enc.load_state_dict(torch.load(args.model_dir+"-phage_enc.pt"))
            host_enc.load_state_dict(torch.load(args.model_dir+"-host_enc.pt"))
            phage_enc = phage_enc.to(args.device)
            host_enc = host_enc.to(args.device)
            
    ## 2. loading data
    spiece_file = args.host_list
    host_fa_file = args.host_fa    
    phage_test_file = args.test_phage_fa
    host_test_file = args.test_host_gold

    fa_test_dataset = fasta_dataset(phage_test_file, spiece_file, host_test_file)
    test_generator = DataLoader(
        fa_test_dataset,
        batch_size,
        collate_fn=partial(my_collate_fn2, kmer=kmer),
        num_workers=num_workers,
    )

    cached_test_ph, cached_test_label, test_phName = [], [], []

    for phs, labels, phName in test_generator:
        imgs_ph = torch.tensor(phs, dtype=torch.float32)
        cached_test_ph.append(torch.unsqueeze(imgs_ph, dim=1))
        # labels: list[list[int]]
        cached_test_label.append(labels)
        test_phName.extend(phName)

    s2l_dic = fa_test_dataset.get_s2l_dic()
    l2fa = get_host_fa(s2l_dic, host_fa_file, kmer)
    l2sn = fa_test_dataset.get_l2s_dic()
    label_list = list(l2fa.keys())
    
    if args.test_host_gold != "":
        acc_test, host_pred_list, gold_list = test(
            phage_enc, host_enc, cached_test_ph, l2fa, cached_test_label, args.device, metric=args.metric)
    else:
        host_pred_list = predict(phage_enc, host_enc, cached_test_ph, l2fa, args.device, metric=args.metric)
    

    for i in range(len(host_pred_list)):

        print(test_phName[i], end="\t")
        idxs = np.argsort(host_pred_list[i], kind="stable")
        for idx in idxs:
            print(l2sn[label_list[idx]] + "_" + str(host_pred_list[i][idx]), end="\t")

        print("")