import sys
import torch
import s3prl.optimizers

distill_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/distill_data2vec_new.pt'
hubert_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/audio_base_ls.pt'
hubert_finetune_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert-finetune/hubert-base-finetune-100h.pt'
new_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/distiller_data2vec.pt'
sys.modules["optimizers"] = s3prl.optimizers
distill_model = torch.load(distill_model_path)

print('distill model')
for key in distill_model['Distiller'].keys():
    print(key, distill_model['Distiller'][key].shape)
hubert_model = torch.load(hubert_model_path) 

print('hubert pretrain model')
for key in hubert_model['model']:
    if(hasattr(hubert_model['model'][key], 'shape')):
        print(key, hubert_model['model'][key].shape)
hubert_model['model'] = distill_model['Distiller']
hubert_model['cfg']['model']['encoder_layers'] = 2

torch.save(hubert_model, new_model_path)
print(f'save model to {new_model_path}')

