from .expert import UpstreamExpert as _UpstreamExpert
import os


def data2vec_ffn(*args, **kwargs):
    return data2vec_ffn_base(*args, **kwargs)

def data2vec_ffn_local(ckpt, *args, **kwargs):
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)

def data2vec_ffn_base(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/upstream_model/audio_base_ls.pt"
    print("Use data2vec base model")
    return data2vec_ffn_local(ckpt=ckpt)

def data2vec_ffn_large(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/upstream_model/vox_pretrained.pt"
    print("Use data2vec large model")
    return data2vec_ffn_local(ckpt=ckpt)