"""
    hubconf for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def distiller_finetune_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def distiller_finetune_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from url
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return distiller_finetune_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def distildata2vec_finetune(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_finetune_data2vec_w_all_loss/states-epoch-1.ckpt"
    print("Use distildata2vec_finetune model")
    return distiller_finetune_local(ckpt=ckpt)