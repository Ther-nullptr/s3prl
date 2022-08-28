# finetune model distiller

## usage

1. convert a finetuned model to a pretrained model (to use its weight), using script `finetune_to_pretrain.py`.
2. convert the fairseq model in 1) to a s3prl model. 
3. extract linear projection from a finetuned model, using script `extract_linear_projection.py`.
4. modify path of finetune-pretrain model and linear model in config file.