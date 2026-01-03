# CE4PHI： Phylogenetic tree-aware positive-unlabeled deep metric learning for phage-host interaction identification

CE4PHI is an extension of CL4PHI (https://github.com/yaozhong/CL4PHI) that generalizes margin-based contrastive learning to a cross-entropy–based deep metric learning framework with explicit phylogenetic tree awareness for phage–host interaction (PHI) identification.
Unlike traditional approaches that train classification models to strictly separate positive and negative phage–host pairs, CE4PHI learns joint representations under supervision from both experimentally validated PHIs and phylogenetic constraints imposed on non-positive host samples.

![](figures/pipeline.png)

## Environments and Package dependency

- python 3.10.12
- Pytorch 2.6.0+cu124
- pyfaidx

