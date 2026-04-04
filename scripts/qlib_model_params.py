#!/usr/bin/env python3
"""Calculate parameter counts for all Qlib official models."""

import torch
import torch.nn as nn
import sys

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total

def mb(params):
    # float32: 4 bytes per param
    return params * 4 / (1024*1024)

print('='*80)
print('Qlib Official Model Parameter Count Analysis')
print('='*80)

# ====== 1. GRU Model ======
from qlib.contrib.model.pytorch_gru import GRUModel

gru_default = GRUModel(d_feat=6, hidden_size=64, num_layers=2)
gru_alpha158 = GRUModel(d_feat=20, hidden_size=64, num_layers=2)

print('\n--- GRU Model ---')
print(f'  Default (d_feat=6, hidden=64, layers=2):')
print(f'    Params: {count_params(gru_default):,}  ({mb(count_params(gru_default)):.4f} MB)')
print(f'  Benchmark Alpha158 (d_feat=20, hidden=64, layers=2):')
print(f'    Params: {count_params(gru_alpha158):,}  ({mb(count_params(gru_alpha158)):.4f} MB)')

# ====== 2. LSTM Model ======
from qlib.contrib.model.pytorch_lstm import LSTMModel

lstm_default = LSTMModel(d_feat=6, hidden_size=64, num_layers=2)
lstm_alpha158 = LSTMModel(d_feat=20, hidden_size=64, num_layers=2)

print('\n--- LSTM Model ---')
print(f'  Default (d_feat=6, hidden=64, layers=2):')
print(f'    Params: {count_params(lstm_default):,}  ({mb(count_params(lstm_default)):.4f} MB)')
print(f'  Benchmark Alpha158 (d_feat=20, hidden=64, layers=2):')
print(f'    Params: {count_params(lstm_alpha158):,}  ({mb(count_params(lstm_alpha158)):.4f} MB)')

# ====== 3. Transformer Model ======
from qlib.contrib.model.pytorch_transformer import Transformer as TransformerNN

trans_inner_default = TransformerNN(d_feat=6, d_model=8, nhead=4, num_layers=2)
trans_outer_default = TransformerNN(d_feat=20, d_model=64, nhead=2, num_layers=2)

print('\n--- Transformer Model ---')
print(f'  Inner class default (d_feat=6, d_model=8, nhead=4, layers=2):')
print(f'    Params: {count_params(trans_inner_default):,}  ({mb(count_params(trans_inner_default)):.4f} MB)')
print(f'  Outer/Benchmark default (d_feat=20, d_model=64, nhead=2, layers=2):')
print(f'    Params: {count_params(trans_outer_default):,}  ({mb(count_params(trans_outer_default)):.4f} MB)')

# ====== 4. ALSTM Model ======
from qlib.contrib.model.pytorch_alstm import ALSTMModel

alstm_default = ALSTMModel(d_feat=6, hidden_size=64, num_layers=2)
alstm_alpha158 = ALSTMModel(d_feat=20, hidden_size=64, num_layers=2)

print('\n--- ALSTM (Attention LSTM) Model ---')
print(f'  Default (d_feat=6, hidden=64, layers=2):')
print(f'    Params: {count_params(alstm_default):,}  ({mb(count_params(alstm_default)):.4f} MB)')
print(f'  Benchmark Alpha158 (d_feat=20, hidden=64, layers=2):')
print(f'    Params: {count_params(alstm_alpha158):,}  ({mb(count_params(alstm_alpha158)):.4f} MB)')

# ====== 5. SFM Model ======
from qlib.contrib.model.pytorch_sfm import SFM_Model

sfm_default = SFM_Model(d_feat=6, hidden_size=64, freq_dim=10, output_dim=1)

print('\n--- SFM (State Frequency Memory) Model ---')
print(f'  Default (d_feat=6, hidden=64, freq_dim=10, output_dim=1):')
print(f'    Params: {count_params(sfm_default):,}  ({mb(count_params(sfm_default)):.4f} MB)')

# ====== 6. TabNet Model ======
from qlib.contrib.model.pytorch_tabnet import TabNet, TabNet_Decoder, FinetuneModel

tabnet_encoder = TabNet(inp_dim=158, out_dim=64, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5)
tabnet_decoder = TabNet_Decoder(64, 158, n_shared=2, n_ind=2, vbs=2048, n_steps=5)
finetune = FinetuneModel(64, 1, tabnet_encoder)

print('\n--- TabNet Model ---')
print(f'  Encoder (inp_dim=158, out_dim=64, n_d/n_a=64, shared=2, ind=2, steps=5):')
print(f'    Params: {count_params(tabnet_encoder):,}  ({mb(count_params(tabnet_encoder)):.4f} MB)')
print(f'  Decoder (pretrain only):')
print(f'    Params: {count_params(tabnet_decoder):,}  ({mb(count_params(tabnet_decoder)):.4f} MB)')
enc_dec = count_params(tabnet_encoder) + count_params(tabnet_decoder)
print(f'  Total Pretrain (Encoder + Decoder): {enc_dec:,}  ({mb(enc_dec):.4f} MB)')
print(f'  FinetuneModel (Encoder + final Linear):')
print(f'    Params: {count_params(finetune):,}  ({mb(count_params(finetune)):.4f} MB)')

# ====== 7. GATs Model ======
from qlib.contrib.model.pytorch_gats import GATModel

gat_default = GATModel(d_feat=6, hidden_size=64, num_layers=2)
gat_alpha158 = GATModel(d_feat=20, hidden_size=64, num_layers=2)

print('\n--- GATs (Graph Attention) Model ---')
print(f'  Default (d_feat=6, hidden=64, layers=2, base=GRU):')
print(f'    Params: {count_params(gat_default):,}  ({mb(count_params(gat_default)):.4f} MB)')
print(f'  Alpha158 (d_feat=20, hidden=64, layers=2, base=GRU):')
print(f'    Params: {count_params(gat_alpha158):,}  ({mb(count_params(gat_alpha158)):.4f} MB)')

# ====== 8. TRA Model ======
from qlib.contrib.model.pytorch_tra import RNN as TRA_RNN, TRA, Transformer as TRA_Transformer

# Benchmark: model_config: hidden_size=256, num_layers=2, rnn_arch=LSTM, use_attn=True
tra_rnn = TRA_RNN(input_size=158, hidden_size=256, num_layers=2, rnn_arch='LSTM', use_attn=True, dropout=0.2)
tra_module = TRA(input_size=tra_rnn.output_size, num_states=3, hidden_size=32, num_layers=1, rnn_arch='LSTM', tau=1.0, src_info='LR_TPE')

print('\n--- TRA (Temporal Routing Adapter) Model ---')
print(f'  RNN backbone (input=158, hidden=256, layers=2, LSTM+Attn):')
print(f'    Params: {count_params(tra_rnn):,}  ({mb(count_params(tra_rnn)):.4f} MB)')
print(f'    output_size: {tra_rnn.output_size}')
print(f'  TRA module (num_states=3, hidden=32, LSTM router):')
print(f'    Params: {count_params(tra_module):,}  ({mb(count_params(tra_module)):.4f} MB)')
tra_total = count_params(tra_rnn) + count_params(tra_module)
print(f'  Total TRA: {tra_total:,}  ({mb(tra_total):.4f} MB)')

# ====== Summary Table ======
print('\n')
print('='*80)
print('SUMMARY TABLE: Qlib Official Models - Benchmark Parameter Counts')
print('='*80)

models = [
    ('GRU',         'd_feat=20, hidden=64, layers=2',            count_params(gru_alpha158)),
    ('LSTM',        'd_feat=20, hidden=64, layers=2',            count_params(lstm_alpha158)),
    ('ALSTM',       'd_feat=20, hidden=64, layers=2 (GRU-based)', count_params(alstm_alpha158)),
    ('Transformer', 'd_feat=20, d_model=64, nhead=2, layers=2',  count_params(trans_outer_default)),
    ('SFM',         'd_feat=6, hidden=64, freq=10',              count_params(sfm_default)),
    ('TabNet',      'inp=158, n_d=n_a=64, steps=5 (finetune)',   count_params(finetune)),
    ('GATs',        'd_feat=20, hidden=64, layers=2, GRU',       count_params(gat_alpha158)),
    ('TRA',         'RNN(158,256,2)+TRA(3,32)',                   tra_total),
]

print(f'{"Model":<15} {"Config":<50} {"Params":>10} {"MB":>8} {"KB":>8}')
print('-'*95)
for name, cfg, params in sorted(models, key=lambda x: x[2]):
    print(f'{name:<15} {cfg:<50} {params:>10,} {mb(params):>8.4f} {mb(params)*1024:>8.1f}')

print()
max_params = 1024*1024//4
print(f'1 MB limit = {max_params:,} float32 parameters')
print()

# Show ratio to 1MB limit
print('--- Ratio to 1 MB Limit ---')
for name, cfg, params in sorted(models, key=lambda x: x[2]):
    ratio = params / max_params * 100
    print(f'  {name:<15} {params:>10,} params = {ratio:>6.2f}% of 1MB limit')

print()

# ====== Bonus: d_feat=158 flat mode (non-TS) ======
print('='*80)
print('BONUS: What if d_feat=158 (flat Alpha158 features)?')
print('='*80)

gru_158 = GRUModel(d_feat=158, hidden_size=64, num_layers=2)
lstm_158 = LSTMModel(d_feat=158, hidden_size=64, num_layers=2)
alstm_158 = ALSTMModel(d_feat=158, hidden_size=64, num_layers=2)
trans_158 = TransformerNN(d_feat=158, d_model=64, nhead=2, num_layers=2)
gat_158 = GATModel(d_feat=158, hidden_size=64, num_layers=2)
sfm_158 = SFM_Model(d_feat=158, hidden_size=64, freq_dim=10, output_dim=1)

for name, model in [('GRU', gru_158), ('LSTM', lstm_158), ('ALSTM', alstm_158),
                     ('Transformer', trans_158), ('GATs', gat_158), ('SFM', sfm_158)]:
    p = count_params(model)
    print(f'  {name:<15} d_feat=158, hidden=64: {p:>10,} params = {mb(p):.4f} MB = {mb(p)*1024:.1f} KB')

# ====== Scaling sensitivity ======
print()
print('='*80)
print('SENSITIVITY: hidden_size scaling with d_feat=158')
print('='*80)
print(f'{"hidden_size":<12} {"GRU params":>12} {"GRU MB":>10} {"LSTM params":>12} {"LSTM MB":>10}')
print('-'*60)
for hs in [64, 128, 256, 512, 1024]:
    gru_x = GRUModel(d_feat=158, hidden_size=hs, num_layers=2)
    lstm_x = LSTMModel(d_feat=158, hidden_size=hs, num_layers=2)
    gp = count_params(gru_x)
    lp = count_params(lstm_x)
    print(f'{hs:<12} {gp:>12,} {mb(gp):>10.4f} {lp:>12,} {mb(lp):>10.4f}')

# ====== Layer-by-layer breakdown for each model ======
print()
print('='*80)
print('LAYER-BY-LAYER BREAKDOWN (Benchmark configs)')
print('='*80)

def print_breakdown(name, model):
    print(f'\n  [{name}]')
    total = 0
    for pname, param in model.named_parameters():
        n = param.numel()
        total += n
        print(f'    {pname:<45} {str(list(param.shape)):<20} {n:>10,}')
    print(f'    {"TOTAL":<45} {"":20} {total:>10,}  ({mb(total):.4f} MB)')

print_breakdown('GRU (d_feat=20, hidden=64, layers=2)', gru_alpha158)
print_breakdown('LSTM (d_feat=20, hidden=64, layers=2)', lstm_alpha158)
print_breakdown('ALSTM (d_feat=20, hidden=64, layers=2)', alstm_alpha158)
print_breakdown('Transformer (d_feat=20, d_model=64, nhead=2, layers=2)', trans_outer_default)
print_breakdown('SFM (d_feat=6, hidden=64, freq=10)', sfm_default)
print_breakdown('GATs (d_feat=20, hidden=64, layers=2)', gat_alpha158)
print_breakdown('TRA RNN backbone', tra_rnn)
print_breakdown('TRA module', tra_module)
