import sys
import torch
import s3prl.optimizers

distill_model_path = '/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert/states-epoch-8.ckpt'
hubert_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert_base.ls960.pt'
hubert_finetune_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert-finetune/hubert-base-finetune-100h.pt'
new_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/distiller.pt'
sys.modules["optimizers"] = s3prl.optimizers
distill_model = torch.load(distill_model_path)

print('distill model')
for key in distill_model['Distiller'].keys():
    print(key, distill_model['Distiller'][key].shape)
hubert_model = torch.load(hubert_model_path) 

print('hubert pretrain model')
for key in hubert_model['model']:
    print(key, hubert_model['model'][key].shape)
hubert_model['model'] = distill_model['Distiller']
hubert_model['args'].encoder_layers = 2

torch.save(hubert_model, new_model_path)
print(f'save model to {new_model_path}')

hubert_finetune_model = torch.load(hubert_finetune_path) 
print('hubert finetune model')
for key in hubert_finetune_model['model']:
    print(key, hubert_finetune_model['model'][key].shape)