"""
    hubconf for data2vec_finetune
    Author: Ther (https://github.com/Ther-nullptr)
"""

import os
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def data2vec_finetune_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def data2vec_finetune_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from url
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return data2vec_finetune_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def data2vec_finetune(refresh=False, *args, **kwargs):
    """
    DistilHuBERT
    """
    return data2vec_finetune_base(refresh=refresh, *args, **kwargs)


def data2vec_finetune_base(refresh=False, *args, **kwargs):
    """
    DistilHuBERT Base
    Default model in https://arxiv.org/abs/2110.01900
    """
    kwargs[
        "ckpt"
    ] = "https://www.dropbox.com/s/hcfczqo5ao8tul3/disilhubert_ls960_4-8-12.ckpt?dl=0"
    return data2vec_finetune_url(refresh=refresh, *args, **kwargs)

def data2vec_finetune_base_local(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/upstream_model/data2vec-finetune/audio_base_ls_100h.pt"
    print("Use data2vec_finetune_base_local model")
    return data2vec_finetune_local(ckpt=ckpt)