import sys
import torch

hubert_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert_base.ls960.pt'
maintain_layers = [10,11]
maintain_str = ''
for layer in maintain_layers:
    maintain_str += str(layer)
    maintain_str += '_'

new_model_path = f'/mnt/lustre/sjtu/home/xc915/superb/upstream_model/distiller_{maintain_str}.pt'

hubert_model = torch.load(hubert_model_path) 
delete_keys = []
change_keys = []
for key in hubert_model["model"].keys():
    if('encoder.layers' not in key):
        pass
    else:
        if(f'encoder.layers.{maintain_layers[0]}' in key): 
            change_keys.append(key)
        elif(f'encoder.layers.{maintain_layers[1]}' in key):
            change_keys.append(key)
        else:
            delete_keys.append(key)

for key in delete_keys:
    del hubert_model["model"][key]

for key in change_keys:
    if(f'encoder.layers.{maintain_layers[0]}' in key): 
        new_key = key
        new_key = new_key.replace(f'encoder.layers.{maintain_layers[0]}', 'encoder.layers.0')
        hubert_model["model"][new_key] = hubert_model["model"][key]
        del hubert_model["model"][key]

    elif(f'encoder.layers.{maintain_layers[1]}' in key):
        new_key = key
        new_key = new_key.replace(f'encoder.layers.{maintain_layers[1]}', 'encoder.layers.1')
        hubert_model["model"][new_key] = hubert_model["model"][key]
        del hubert_model["model"][key]

hubert_model['args'].encoder_layers = 2

for key in hubert_model["model"].keys():
    print(key)

torch.save(hubert_model, new_model_path)
print(f'save model to {new_model_path}')