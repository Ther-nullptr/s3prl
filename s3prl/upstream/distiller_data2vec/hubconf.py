"""
    hubconf for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import os
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def distiller_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def distiller_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from url
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return distiller_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def distilhubert(refresh=False, *args, **kwargs):
    """
    DistilHuBERT
    """
    return distilhubert_base(refresh=refresh, *args, **kwargs)


def distilhubert_base(refresh=False, *args, **kwargs):
    """
    DistilHuBERT Base
    Default model in https://arxiv.org/abs/2110.01900
    """
    kwargs[
        "ckpt"
    ] = "https://www.dropbox.com/s/hcfczqo5ao8tul3/disilhubert_ls960_4-8-12.ckpt?dl=0"
    return distiller_url(refresh=refresh, *args, **kwargs)

def distilhubert_half(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert/states-110000.ckpt"
    print("Use distilhubert_half model")
    return distiller_local(ckpt=ckpt)

def distilhubert_full(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/wyj-s3prl/s3prl/result/pretrain/distill_hubert/states-epoch-8.ckpt"
    print("Use distilhubert_full model")
    return distiller_local(ckpt=ckpt)