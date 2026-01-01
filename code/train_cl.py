# train the constrastive learning 
import argparse, time, random, os
from data_loading import *
from model import *
from eval import *

import torch
import multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F

from sklearn import metrics
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DTYPE=torch.float32

def set_seed(s):
    # set for the CUDA > 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def train_tpuNCE(dl_model, enc_mode, data_set, model_path, kmer, margin, batch_size, lr, epoch,\
	device="cuda:0", num_workers=64, verbose=True, \
    # === tpuNCE related parameters ===
    temperature=0.07,tree_dist_path="", tree_sigma=-1.0, taxo_dic_file="", tree_level="none", lambda_ph_tree=0.0, l2_lambda=0.0, \
                 metric="chord", out_dim=512, tree_ce_eps=0.02):

	cores = mp.cpu_count()
	if num_workers > 0 and num_workers < cores:
		cores = num_workers

	## data loading phase
	if verbose:
		print(" |- Start preparing dataset...")

	host_fa_file, spiece_file, phage_train_file, host_train_file, phage_valid_file, host_valid_file = data_set
	
	start_dataload = time.time()

    #----------------------------------------------------------------------------------------------------
	train_data_labels = get_data_host_sets([host_train_file, host_valid_file])
	print("	|-* Provided training sets totally has [", len(train_data_labels),"] hosts.")

	fa_train_dataset = fasta_dataset(phage_train_file, spiece_file, host_train_file)
	l2fa = get_host_fa(fa_train_dataset.get_s2l_dic(), host_fa_file, kmer)
	# add filters for host label here
	l2fa_filter = get_host_fa(fa_train_dataset.get_s2l_dic(), host_fa_file, kmer, train_data_labels)
	print("	|-[!] Checking host label information filtering for the training [non_filter:", len(l2fa.keys()), ", filtered:", len(l2fa_filter.keys()), "].")

    # change to the TPU-dataLoader
	train_generator = DataLoader(fa_train_dataset, batch_size, collate_fn=partial(my_collate_fn_infoNCE, kmer=kmer, l2fa=l2fa_filter), num_workers=num_workers)
    
	fa_valid_dataset = fasta_dataset(phage_valid_file, spiece_file, host_valid_file)
	valid_generator = DataLoader(fa_valid_dataset, batch_size, collate_fn=partial(my_collate_fn2, kmer=kmer), num_workers=num_workers) 

	# cached the data set
	cached_train_ph, cached_train_bt, cached_train_label = [], [], []
	cached_valid_ph, cached_valid_label, cached_valid_phageName = [], [], []

	## loading the image pairs for constrastive training
	for phs, bts, labels in train_generator:
		#X = torch.tensor(X, dtype = torch.float32).transpose(1,2)
		imgs_ph = torch.tensor(phs, dtype = torch.float32, device='cpu')
		imgs_bt = torch.tensor(bts, dtype = torch.float32, device='cpu')
		
		cached_train_ph.append(torch.unsqueeze(imgs_ph, dim=1))
		cached_train_bt.append(torch.unsqueeze(imgs_bt, dim=1))
		cached_train_label.append(torch.tensor(labels, device='cpu'))

	for phs, labels, phName in valid_generator:
		imgs_ph = torch.tensor(phs, dtype=torch.float32)
		cached_valid_ph.append(torch.unsqueeze(imgs_ph, dim=1))
    	# list[list[int]]
		cached_valid_label.append(labels)
		cached_valid_phageName.append(phName)

	if not os.path.exists(tree_dist_path):
		raise FileNotFoundError(f"--tree_dist not found: {tree_dist_path}")
	tree_dists_df = pd.read_csv(tree_dist_path, index_col=0)
	tree_dists_df.index   = [replace_first_underscore_with_space(x) for x in tree_dists_df.index]
	tree_dists_df.columns = [replace_first_underscore_with_space(x) for x in tree_dists_df.columns]
	print(f"|- [tree] loaded distance matrix of shape {tree_dists_df.shape} from {tree_dist_path}")
	s2l_dict = fa_train_dataset.get_s2l_dic()   
	tree_dists_tensor = build_aligned_tree_dist_tensor(tree_dists_df, l2fa_filter, s2l_dict, device, fill_inf=np.inf)
    
	print(" |- [tree] Loading GTDB taxonomy from:", taxo_dic_file)
	gtdb_tax = load_gtdb_taxonomy(taxo_dic_file)
	s2l_dict = fa_train_dataset.get_s2l_dic()
	label_ids_sub = list(l2fa_filter.keys())
	label2species = {lab: sp for sp, lab in s2l_dict.items()}
	host_labels = []
	for lab in label_ids_sub:
		sp = label2species.get(lab)
		if sp is None:
			raise KeyError(f"[tree] label_id={lab} species name not found in s2l_dict")
		host_labels.append(sp)

	genus_ids, _  = build_level_ids(host_labels, gtdb_tax, "genus",  device=device)
	family_ids, _ = build_level_ids(host_labels, gtdb_tax, "family", device=device)
	level_ids_dict = {"genus": genus_ids,"family": family_ids,}

	print(" |- loading [ok].")
	used_dataload = time.time() - start_dataload
	print("  |-@ used time:", round(used_dataload,2), "s")
    #-------------------------------------------------------------------------------------------

	start_train = time.time()

	# model part (using CNN module)
	if dl_model == "CNN":
		if args.enc_mode == "share":
			print("[INFO] Using SHARED encoder (cnn_module) for phage & host")
       
			shared_enc = cnn_module().to(device)
			phage_backbone = shared_enc
			host_backbone  = shared_enc

			optimizer = optim.Adam(shared_enc.parameters(),lr=lr, betas=(0.9, 0.999), weight_decay=0)

		else:  # separate
			print("[INFO] Using SEPARATE encoders: phage=cnn_module, host=cnn_module_bac")
			phage_backbone = cnn_module(7).to(device)
			host_backbone  = cnn_module_bac(9).to(device)
			optimizer = optim.Adam([{'params': phage_backbone.parameters(), 'lr': lr}, \
			{'params': host_backbone.parameters(),  'lr': lr * 0.5},],betas=(0.9, 0.999))
    
	elif dl_model == "ResNet":
		shared_enc = ResNet_module(ResidualBlock, [3, 4, 6, 3]).to(device)
		phage_backbone = shared_enc
		host_backbone  = shared_enc
		optimizer = torch.optim.SGD(shared_enc.parameters(), lr=lr, weight_decay = 0.001, momentum = 0.9)  
            
	elif dl_model == "finetune":
		print("finetuning existing model at ", args.finetune_model_dir)

		if args.enc_mode == "share":
			print("[INFO] Finetune SHARED encoder from", args.finetune_model_dir)
			shared_enc = cnn_module(7, 0)
			shared_enc.load_state_dict(torch.load(args.finetune_model_dir)) 
			shared_enc = shared_enc.to(args.device)

			phage_backbone = shared_enc
			host_backbone  = shared_enc

			optimizer = optim.Adam(shared_enc.parameters(),lr=lr, betas=(0.9, 0.999), weight_decay=1e-2,)

		else:
			print("[Error]:Not supported currently. To be added...")

	criterion = TreePUInfoNCE(temperature=temperature, normalize=True, metric=metric, tree_sigma=tree_sigma, alpha=2.0, lambda_ph_tree=lambda_ph_tree, l2_lambda=l2_lambda, margin=margin, tree_ce_eps=tree_ce_eps).to(device)

	if verbose:
		num_phage_params = sum(p.nelement() for p in phage_backbone.parameters())
		num_host_params  = sum(p.nelement() for p in host_backbone.parameters())
    
		print(" |- Parameter statistics:")
		print(f"    |- Phage encoder parameters: {num_phage_params}")
		print(f"    |- Host  encoder parameters: {num_host_params}")
		print("  |- Training started ...")

	# start training
	epoch_acc_valid, epoch_acc_test, epoch_cm = [], [], []
	current_best_valid_acc = -100

	mantel_perms = 0       
	tree_dists_tensor_fix = fix_tree_distance_matrix(tree_dists_tensor)
    
	for ep in range(epoch):
		phage_backbone.train(); host_backbone.train()
		epoch_loss = 0
    
		for i in range(len(cached_train_ph)):
			phs, bts, labels = cached_train_ph[i], cached_train_bt[i], cached_train_label[i]

			phs = phs.to(device)
			bts = bts.to(device)
			pos_mask = labels.to(device)  # [B,M]

			embed_ph = phage_backbone(phs)
			embed_bt = host_backbone(bts)

			B = phs.shape[0]
			M = bts.shape[0]
			ph_flat = phs.view(B, -1)
			bt_flat = bts.view(M, -1)
			ph_flat = F.normalize(ph_flat, dim=-1)
			bt_flat = F.normalize(bt_flat, dim=-1)
			d_euclid = torch.cdist(ph_flat, bt_flat, p=2)
			kmer_dist = d_euclid / 2.0

			tree_mask = make_tree_mask_level(level_keyword=tree_level, pos_mask=pos_mask, \
                     level_ids_dict=level_ids_dict, device=device)

			loss = criterion(z_p=embed_ph,z_h=embed_bt, pos_mask=pos_mask, tree_dists=tree_dists_tensor,\
                             tree_mask=tree_mask, kmer_scores=kmer_dist) 

			epoch_loss += loss.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print("Epoch-%d, Loss=%f" %(ep,epoch_loss))

		acc_valid, _, _ = test(phage_backbone, host_backbone, cached_valid_ph, l2fa_filter, cached_valid_label, device, 1, True)
		epoch_acc_valid.append(acc_valid)

		if acc_valid > current_best_valid_acc: # to be consistent with the following one. 
			current_best_valid_acc = acc_valid
			if args.enc_mode == "share":
			    torch.save(phage_backbone.state_dict(), model_path)
			if args.enc_mode == "seperate":
			    torch.save(phage_backbone.state_dict(), model_path + "-phage_enc.pt")
			    torch.save(host_backbone.state_dict(),  model_path + "-host_enc.pt")
	
	idx = epoch_acc_valid.index(max(epoch_acc_valid))
	print(f"[Valid epoch idx/epoch]:{idx}/{epoch}, [valid acc]:{epoch_acc_valid[idx]:.4f}")
	used_train = time.time() - start_train
	print(" @ used training time:", round(used_train,2), "s. Total time:", round(used_train+used_dataload,2))



def train_mgcl(dl_model, enc_mode, data_set, model_path, kmer, margin, batch_size, lr, epoch,\
	device="cuda:0", num_workers=64, verbose=True,):

	cores = mp.cpu_count()
	if num_workers > 0 and num_workers < cores:
		cores = num_workers

	## data loading phase
	if verbose:
		print(" |- Start preparing dataset...")

	host_fa_file, spiece_file, phage_train_file, host_train_file, phage_valid_file, host_valid_file = data_set
	
	start_dataload = time.time()

    #----------------------------------------------------------------------------------------------------
	train_data_labels = get_data_host_sets([host_train_file, host_valid_file])
	print("	|-* Provided training sets totally has [", len(train_data_labels),"] hosts.")

	fa_train_dataset = fasta_dataset(phage_train_file, spiece_file, host_train_file)
	l2fa = get_host_fa(fa_train_dataset.get_s2l_dic(), host_fa_file, kmer)
	# add filters for host label here
	l2fa_filter = get_host_fa(fa_train_dataset.get_s2l_dic(), host_fa_file, kmer, train_data_labels)
	print("	|-[!] Checking host label information filtering for the training [non_filter:", len(l2fa.keys()), ", filtered:", len(l2fa_filter.keys()), "].")

    
	train_generator = DataLoader(fa_train_dataset, batch_size,collate_fn=partial(my_collate_fn, kmer=kmer, \
                                                                                 l2fa=l2fa_filter),num_workers=num_workers)
    
	fa_valid_dataset = fasta_dataset(phage_valid_file, spiece_file, host_valid_file)
	valid_generator = DataLoader(fa_valid_dataset, batch_size, collate_fn=partial(my_collate_fn2, kmer=kmer), num_workers=num_workers) 

	# cached the data set
	cached_train_ph, cached_train_bt, cached_train_label = [], [], []
	cached_valid_ph, cached_valid_label, cached_valid_phageName = [], [], []

	## loading the image pairs for contrastive training
	for phs, bts, labels in train_generator:
		imgs_ph = torch.tensor(phs, dtype = DTYPE)
		imgs_bt = torch.tensor(bts, dtype = DTYPE)
		
		cached_train_ph.append(torch.unsqueeze(imgs_ph, dim=1))
		cached_train_bt.append(torch.unsqueeze(imgs_bt, dim=1))
		cached_train_label.append(torch.tensor(labels))

	for phs, labels, phName in valid_generator:
		imgs_ph = torch.tensor(phs, dtype=DTYPE)
		cached_valid_ph.append(torch.unsqueeze(imgs_ph, dim=1))
		cached_valid_label.append(labels)
		cached_valid_phageName.append(phName)

	print(" |- loading [ok].")
	used_dataload = time.time() - start_dataload
	print("  |-@ used time:", round(used_dataload,2), "s")
    #-------------------------------------------------------------------------------------------

	start_train = time.time()

	# model part (using CNN module)
	if dl_model == "CNN":
		if args.enc_mode == "share":
			print("[INFO] Using SHARED encoder (cnn_module) for phage & host")
       
			shared_enc = cnn_module(7).to(device)
			#shared_enc = cnn_module_bac(9).to(device)
			phage_backbone = shared_enc
			host_backbone  = shared_enc

			optimizer = optim.Adam(shared_enc.parameters(),lr=lr, betas=(0.9, 0.999))

		else:  # separate
			print("[INFO] Using SEPARATE encoders: phage=cnn_module, host=cnn_module_bac")
			phage_backbone = cnn_module(7).to(device)
			host_backbone  = cnn_module_bac(9).to(device)

			optimizer = optim.Adam([{'params': phage_backbone.parameters(), 'lr': lr}, \
			{'params': host_backbone.parameters(),  'lr': lr * 0.5},],betas=(0.9, 0.999))
            
	elif dl_model == "finetune":
		print("finetuning existing model at ", args.finetune_model_dir)

		if args.enc_mode == "share":
			print("[INFO] Finetune SHARED encoder from", args.finetune_model_dir)
			shared_enc = cnn_module(7, 0)
			shared_enc.load_state_dict(torch.load(args.finetune_model_dir)) 
			shared_enc = shared_enc.to(args.device)

			phage_backbone = shared_enc
			host_backbone  = shared_enc

			optimizer = optim.Adam(shared_enc.parameters(),lr=lr, betas=(0.9, 0.999),)

		else:
			print("[Error]:Not supported currently. To be added...")
    
    # to revise later
	criterion = ContrastiveLoss(margin)     

	if verbose:
		num_phage_params = sum(p.nelement() for p in phage_backbone.parameters())
		num_host_params  = sum(p.nelement() for p in host_backbone.parameters())
    
		print(" |- Parameter statistics:")
		print(f"    |- Phage encoder parameters: {num_phage_params}")
		print(f"    |- Host  encoder parameters: {num_host_params}")
		print("  |- Training started ...")

	# start training
	epoch_acc_valid, epoch_acc_test, epoch_cm = [], [], []
	current_best_valid_acc = -100
    
	for ep in range(epoch):
		phage_backbone.train(); host_backbone.train()
		epoch_loss = 0
		for i in range(len(cached_train_ph)):
			phs, bts, labels = cached_train_ph[i], cached_train_bt[i], cached_train_label[i]

			phs = phs.to(device)
			bts = bts.to(device)
			labels = labels.to(device)

			embed_ph = phage_backbone(phs)
			embed_bt = host_backbone(bts)

			loss = criterion(embed_ph, embed_bt, labels)
			epoch_loss += loss.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print("Epoch-%d, Loss=%f" %(ep,epoch_loss))

		acc_valid, _, _ = test(phage_backbone, host_backbone, cached_valid_ph, l2fa_filter, cached_valid_label, device, 1, True)
		epoch_acc_valid.append(acc_valid)

		if acc_valid > current_best_valid_acc: # to be consistent with the following one. 
			current_best_valid_acc = acc_valid
			if args.enc_mode == "share":
			    torch.save(phage_backbone.state_dict(), model_path)
			if args.enc_mode == "seperate":
			    torch.save(phage_backbone.state_dict(), model_path + "-phage_enc.pt")
			    torch.save(host_backbone.state_dict(),  model_path + "-host_enc.pt")
	
	idx = epoch_acc_valid.index(max(epoch_acc_valid))
	print(f"[Valid epoch idx/epoch]:{idx}/{epoch}, [valid acc]:{epoch_acc_valid[idx]:.4f}")
	used_train = time.time() - start_train
	print(" @ used training time:", round(used_train,2), "s. Total time:", round(used_train+used_dataload,2))


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='<Contrastive learning for the phage-host identification>')

	parser.add_argument('--model',       default="CNN", type=str, required=True, help='contrastive learning encoding model')
	parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
	parser.add_argument("--enc_mode", type=str, default="seperate", choices=["share", "seperate"],help="share: phage/host use the same encoder; separate: two encoders")
    

	parser.add_argument('--finetune_model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
	parser.add_argument('--device',       default="cuda:0", type=str, required=False, help='GPU Device(s) used for training')
	
	parser.add_argument('--kmer',       default=5,       type=int, required=True, help='kmer length')
	parser.add_argument('--seed',       default=123,     type=int, required=False, help='seed [default:123]')
	parser.add_argument('--margin',     default=0.05,    type=float, required=False, help='Margins used in the contrastive training')
	parser.add_argument('--lr',     	default=1e-3,   type=float, required=False, help='Learning rate')
	parser.add_argument('--epoch',      default=20,       type=int, required=False, help='Training epcohs')
	parser.add_argument('--batch_size' ,default=64,      type=int,  required=False, help="batch_size of the training.")
	parser.add_argument('--workers',    default=64,       type=int, required=False, help='number of worker for data loading')

    # loss function related
	parser.add_argument('--lossType', type=str, default="tpuNCE", choices=["margin_cl", "tpuNCE"])
	parser.add_argument('--temperature', default=0.07, type=float)
	parser.add_argument('--tree_dist',   type=str, default="")
	parser.add_argument('--tree_sigma',  type=float, default=-1.0)
	parser.add_argument('--taxo_dic',   type=str, default="/mnt/ws1/phage_host2/data/database/release226/taxonomy/gtdb_taxonomy.tsv")
	parser.add_argument('--tree_level', type=str, default="all", choices=["none", "all", "genus"])
	parser.add_argument('--lambda_ph_tree', default=0.0, type=float)
	parser.add_argument('--l2_lambda', default=0.0, type=float)
	parser.add_argument('--out_dim', default=512, type=int)
	parser.add_argument('--metric',type=str,default="chord", choices=["chord", "euclidean", "cosine", "hyperbolic"],help="Distance metric used in TreePUInfoNCE.",)

	# data related input
	parser.add_argument('--host_fa',   default="",  type=str, required=True, help='Host fasta files')
	parser.add_argument('--host_list', default="",  type=str, required=True, help='Host species list')

	parser.add_argument('--train_phage_fa', default="",   type=str, required=True, help='Trainset Phage fasta file')
	parser.add_argument('--train_host_gold', default="",  type=str, required=True, help='Trainset Phage infectable host label')
	parser.add_argument('--valid_phage_fa', default="",   type=str, required=True, help='Validset Phage fasta file')
	parser.add_argument('--valid_host_gold', default="",  type=str, required=True, help='Validset Phage infectable host label')

	parser.add_argument('--tree_ce_eps', default=0.02, type=float)

	
	args = parser.parse_args()
	set_seed(args.seed)
	print("[@] Setting the seed of: ", args.seed)
	
	data_set=[args.host_fa, args.host_list, args.train_phage_fa, args.train_host_gold, args.valid_phage_fa, args.valid_host_gold]

    # tpuNCE training
	if args.tree_dist != "":
		print("[@] Performing tpuNCE (with or without tree weight for unalbedled sample).")
		train_tpuNCE(dl_model=args.model,enc_mode=args.enc_mode,data_set=data_set,model_path=args.model_dir,kmer=args.kmer,\
		margin=args.margin,batch_size=args.batch_size,lr=args.lr,epoch=args.epoch,device=args.device,num_workers=args.workers, \
                     verbose=True,temperature=args.temperature, tree_dist_path=args.tree_dist, \
                     tree_sigma=args.tree_sigma,taxo_dic_file=args.taxo_dic,tree_level=args.tree_level,\
                     lambda_ph_tree=args.lambda_ph_tree, l2_lambda=args.l2_lambda, metric=args.metric,\
                     out_dim=args.out_dim, tree_ce_eps=args.tree_ce_eps)
	
    # used in the final merge stage
	else:
		print("[@] Performing Margin-based contrastive learning.")
        # model train，contrastive learning
		train_mgcl(args.model, args.enc_mode, data_set, args.model_dir, args.kmer,args.margin, args.batch_size, \
	 	args.lr, args.epoch, args.device, args.workers)
    


