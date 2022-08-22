import torch

hubert_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert_0_1.pt'
new_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert_0_1.pt'

hubert_model = torch.load(hubert_model_path) 
hubert_model['args'].encoder_layers = 2

torch.save(hubert_model, new_model_path)
print(f'save model to {new_model_path}')