import torch
# device = torch.device("cpu")
# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# model parameter setting
batch_size = 64
d_model = 64
n_layers_agg=2
n_layers = 2
n_heads = 4
ffn_hidden = 128
drop_prob = 0.1

masked_ratio = 0.2
beta = 2

mu_start = 0.1
mu_end = 1

epoch_stage1=20
epoch_stage2=5

lr_stage1=0.0002
lr_stage2=0.0001