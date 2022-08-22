import sys
import torch

hubert_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert_base.ls960.pt'
maintain_layers = []
maintain_str = ''
for layer in maintain_layers:
    maintain_str += str(layer)
    maintain_str += '_'

new_model_path = f'/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert_0_1_true.pt'

hubert_model = torch.load(hubert_model_path) 
key_list = list(hubert_model["model"].keys())

for key in key_list:
    if('encoder.layers' in key):
        if('encoder.layers.0.' in key or 'encoder.layers.1.' in key):
            pass
        else:
            del hubert_model["model"][key]

for key in hubert_model["model"].keys():
    print(key)

hubert_model['args'].encoder_layers = 2

torch.save(hubert_model, new_model_path)
print(f'save model to {new_model_path}')