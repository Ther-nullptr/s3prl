import sys
import torch

old_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/audio_base_ls.pt'
maintain_layers = []
maintain_str = ''
for layer in maintain_layers:
    maintain_str += str(layer)
    maintain_str += '_'

new_model_path = f'/mnt/lustre/sjtu/home/xc915/superb/upstream_model/data2vec_0_1.pt'

old_model = torch.load(old_model_path) 
key_list = list(old_model["model"].keys())

for key in key_list:
    if('encoder.layers' in key):
        if('encoder.layers.0.' in key or 'encoder.layers.1.' in key):
            pass
        else:
            del old_model["model"][key]

for key in old_model["model"].keys():
    print(key)

print(old_model['cfg']['model']['encoder_layers'])
old_model['cfg']['model']['encoder_layers'] = 2

torch.save(old_model, new_model_path)
print(f'save model to {new_model_path}')