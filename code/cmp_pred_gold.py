#!/usr/bin/env python3
import argparse
import re

# ======================
# Analysis gold CSV
# ======================
def parse_gold_labels(gold_path):
    gold_species_list = []
    with open(gold_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                gold_species_list.append(set())
                continue
            parts = [p.strip().lower() for p in line.split(",") if p.strip()]
            gold_species_list.append(set(parts))
            # print(len(parts), set(parts))
    return gold_species_list


# ======================
# Analysis prediction
# ======================
def parse_prediction_line(line):
    line = line.strip()
    if not line:
        return None, []

    parts = line.split("\t")
    if len(parts) == 1:
        print("[Error!]: Please check the file format!")
        exit()

    if len(parts) == 0:
        return None, []

    phage_id = parts[0]
    pred_species = []

    for field in parts[1:]:
        field = field.strip()
        if not field:
            continue

        species_part, sep, score_part = field.rpartition("_")
        if sep == "":
            # if no '_'，skip
            continue

        species_part = species_part.strip()
        score_part = score_part.strip()

        # checking score
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?", score_part):
            continue

        if species_part:
            pred_species.append(species_part.lower())

    return phage_id, pred_species


def parse_predictions(pred_path):
    phage_ids = []
    pred_species_list = []

    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pid, preds = parse_prediction_line(line)
            if pid is None:
                continue
            phage_ids.append(pid)
            pred_species_list.append(preds)
            # print(pid+"\t"+"\t".join(preds))

    return phage_ids, pred_species_list


# ======================
# species → genus
# ======================
def species_to_genera(species_set_or_list):
    if isinstance(species_set_or_list, set):
        return {s.split()[0] for s in species_set_or_list if s.strip()}
    else:
        return [s.split()[0] for s in species_set_or_list if s.strip()]


# ================================
# Multi-host Acccuarcy Cacluation
# ================================
def evaluate_multihost(gold_species_list, pred_species_list):

    assert len(gold_species_list) == len(pred_species_list)

    total_gold_species = 0
    total_hit_species = 0

    total_gold_genus = 0
    total_hit_genus = 0

    n = len(gold_species_list)

    for i in range(n):
        gold_species = gold_species_list[i]
        preds_species = pred_species_list[i]

        if not gold_species:
            continue

        # -------- species-level --------
        G = len(gold_species)
        total_gold_species += G

        topK_s = set(preds_species[:G])
        hit_s = len(topK_s & gold_species)
        total_hit_species += hit_s

        # -------- genus-level --------
        gold_genus = species_to_genera(gold_species)
        pred_genus = species_to_genera(preds_species)

        Gg = len(gold_genus)
        if Gg == 0:
            continue

        total_gold_genus += Gg
        topK_g = set(pred_genus[:Gg])
        hit_g = len(topK_g & gold_genus)
        total_hit_genus += hit_g

    acc_species = total_hit_species / total_gold_species if total_gold_species > 0 else 0.0
    acc_genus = total_hit_genus / total_gold_genus if total_gold_genus > 0 else 0.0

    stats = {
        "total_gold_species": total_gold_species,
        "total_hit_species": total_hit_species,
        "total_gold_genus": total_gold_genus,
        "total_hit_genus": total_hit_genus,
    }

    return acc_species, acc_genus, stats


# ================================
# k-best Acccuarcy Cacluation
# ================================
def evaluate_topk(gold_species_list, pred_species_list, k_list=(1, 3, 5, 10)):

    assert len(gold_species_list) == len(pred_species_list)

    k_list = sorted(set(k_list))

    # species level counters
    species_correct = {k: 0 for k in k_list}
    species_total = 0

    # genus level counters
    genus_correct = {k: 0 for k in k_list}
    genus_total = 0

    n = len(gold_species_list)
    for i in range(n):
        gold_species = gold_species_list[i]
        preds_species = pred_species_list[i]

        # ---------- species-level ----------
        if gold_species:
            species_total += 1
            gold_s = gold_species

            for k in k_list:
                if not preds_species:
                    continue
                # top-k predictions
                topk_pred_s = set(preds_species[:k])
                if topk_pred_s & gold_s:
                    species_correct[k] += 1

        # ---------- genus-level ----------
        gold_genus = species_to_genera(gold_species)
        if gold_genus:
            genus_total += 1
            pred_genus = species_to_genera(preds_species)

            for k in k_list:
                if not pred_genus:
                    continue
                topk_pred_g = set(pred_genus[:k])
                if topk_pred_g & gold_genus:
                    genus_correct[k] += 1

    species_topk_acc = {
        k: (species_correct[k] / species_total if species_total > 0 else 0.0)
        for k in k_list
    }
    genus_topk_acc = {
        k: (genus_correct[k] / genus_total if genus_total > 0 else 0.0)
        for k in k_list
    }

    stats_topk = {
        "species_total": species_total,
        "species_correct": species_correct,
        "genus_total": genus_total,
        "genus_correct": genus_correct,
    }

    return species_topk_acc, genus_topk_acc, stats_topk


# ======================
# Main
# ======================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-host aware evaluation for PHI predictions (species & genus level)."
    )
    parser.add_argument("--pred", required=True, help="Prediction file path")
    parser.add_argument("--gold", required=True, help="Gold CSV path (no header, ','-separated multi-host)")
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="If >0, print first N parsed predictions for sanity check.",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading gold labels from: {args.gold}")
    gold_species_list = parse_gold_labels(args.gold)

    print(f"[INFO] Loading predictions from: {args.pred}")
    phage_ids, pred_species_list = parse_predictions(args.pred)

    # length checking
    if len(gold_species_list) != len(pred_species_list):
        n = min(len(gold_species_list), len(pred_species_list))
        print(f"[Error] gold size ({len(gold_species_list)}) != pred size ({len(pred_species_list)}). Using first {n}.")
        exit(-1)

    # Debug
    if args.debug > 0:
        N = min(args.debug, len(pred_species_list))
        print("\n[DEBUG] First {} parsed examples:".format(N))
        for i in range(N):
            pid = phage_ids[i] if i < len(phage_ids) else f"idx_{i}"
            preds = pred_species_list[i]
            genera = species_to_genera(preds)
            print(f"  #{i}  phage_id={pid}")
            print(f"      pred species: {preds}")
            print(f"      pred genera:  {genera}")
        print("")


    acc_species, acc_genus, stats = evaluate_multihost(gold_species_list, pred_species_list)

    print("\n========== MULTI-HOST AWARE EVALUATION (label-wise, top-G) ==========")
    print(f"Species-level accuracy = {acc_species:.4f}")
    print(f"Genus-level accuracy   = {acc_genus:.4f}")
    print("---------------------------------------------------")
    print(f"Total gold species labels: {stats['total_gold_species']}")
    print(f"Total hits (species):      {stats['total_hit_species']}")
    print(f"Total gold genera labels:  {stats['total_gold_genus']}")
    print(f"Total hits (genera):       {stats['total_hit_genus']}")
    print("===================================================\n")

   
    k_list = (1, 3, 5, 10)
    species_topk_acc, genus_topk_acc, stats_topk = evaluate_topk(
        gold_species_list, pred_species_list, k_list=k_list
    )

    print("========== K-BEST EVALUATION (instance-wise) ==========")
    print(f"Species-level (total phages with gold): {stats_topk['species_total']}")
    for k in k_list:
        correct = stats_topk["species_correct"][k]
        acc = species_topk_acc[k]
        print(f"  Top-{k:2d} species-accuracy: {acc:.4f}  (correct={correct})")

    print("---------------------------------------------------")
    print(f"Genus-level (total phages with gold):   {stats_topk['genus_total']}")
    for k in k_list:
        correct = stats_topk["genus_correct"][k]
        acc = genus_topk_acc[k]
        print(f"  Top-{k:2d} genus-accuracy:   {acc:.4f}  (correct={correct})")
    print("===================================================\n")


if __name__ == "__main__":
    main()
