"""
    hubconf for hubert_finetune
    Author: Ther (https://github.com/Ther-nullptr)
"""

import os
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def hubert_finetune_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def hubert_finetune_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from url
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return hubert_finetune_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def hubert_finetune(refresh=False, *args, **kwargs):
    """
    DistilHuBERT
    """
    return hubert_finetune_base(refresh=refresh, *args, **kwargs)


def hubert_finetune_base(refresh=False, *args, **kwargs):
    """
    DistilHuBERT Base
    Default model in https://arxiv.org/abs/2110.01900
    """
    kwargs[
        "ckpt"
    ] = "https://www.dropbox.com/s/hcfczqo5ao8tul3/disilhubert_ls960_4-8-12.ckpt?dl=0"
    return hubert_finetune_url(refresh=refresh, *args, **kwargs)

def hubert_finetune_base_local(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-fairseq/outputs/hubert/libri960h_base/finetune_100h/checkpoints/checkpoint_best.pt"
    print("Use hubert_finetune_base_local model")
    return hubert_finetune_local(ckpt=ckpt)