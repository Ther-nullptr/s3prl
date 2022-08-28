"""
    Pre-train expert for distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

from easydict import EasyDict as edict
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pretrain.distillfusion.dataset import OnlineWaveDataset
from upstream.distiller_finetune.model import DistillerConfig, DistillerModel
import wandb

# change according to your finetuning model
# from upstream.hubert.expert import UpstreamExpert

def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


class UpstreamPretrainExpert(nn.Module):
    """
    The Distiller pretrain expert
    """
    def __init__(self,
                 datarc,
                 upstream_config,
                 device="cuda",
                 multi_gpu=False,
                 **kwargs):
        super().__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu

        if type(upstream_config) == str:
            self.upstream_config = yaml.load(open(upstream_config, "r"),
                                             Loader=yaml.FullLoader)
            print(
                "[UpstreamPretrainExpert] - Using upstream config from:",
                upstream_config,
            )
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print(
                "[UpstreamPretrainExpert] - Using upstream config from the previous experiment."
            )
        else:
            raise ValueError

        self._get_train_dataloader()

        print("[UpstreamPretrainExpert] - Initializing model...")
        model_config = DistillerConfig(self.upstream_config["distiller"])
        self.model = DistillerForPretrain(
            model_config, edict(self.upstream_config["teacher"])
        )  #! inside a model has a distiller and a teacher

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print("[UpstreamPretrainExpert] - Multi-GPU training Enabled: " +
                  str(torch.cuda.device_count()))
        print("[UpstreamPretrainExpert] - Number of parameters: " + str(
            sum(p.numel()
                for p in self.model.parameters() if p.requires_grad)))

    def _get_train_dataloader(self):
        dataset = OnlineWaveDataset(
            self.upstream_config["task"],
            self.datarc["train_batch_size"],
            target_level=self.upstream_config["audio"]["target_level"],
            **self.datarc,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=True,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def load_model(self, all_states):
        if self.multi_gpu:
            self.model.module.distiller.load_state_dict(
                all_states["Distiller"])
        else:
            self.model.distiller.load_state_dict(all_states["Distiller"])

    # Interface
    def add_state_to_save(self, all_states):
        all_states["Distiller"] = (
            self.model.float().distiller.state_dict() if not self.multi_gpu
            else self.model.float().module.distiller.state_dict())
        all_states["Config"] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    # Interface
    def forward(self,
                data,
                records={},
                global_step=0,
                log_step=1000,
                **kwargs):  #! true forward step
        """
        Args:
            data:
                [wave_input, pad_mask]

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss
        """

        wave_input, wave_orig, wave_len, pad_mask = data
        wave_input = wave_input.to(self.device)
        wave_len = wave_len.to(self.device)
        pad_mask = pad_mask.type(wave_input.dtype).to(self.device)

        loss, other_res = self.model(  #! DistillerForPretrain
            wave_input,
            wave_orig,
            wave_len,
            pad_mask,
            return_other=global_step % log_step == 0,
        )

        return loss, records

    # interface
    def on_before_zero_grad(self):
        pass

    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            if isinstance(values, torch.Tensor) and len(values.shape) > 1:
                logger.add_image(f"{prefix}{key}",
                                 values,
                                 global_step=global_step)
            elif isinstance(values, float):
                logger.add_scalar(f"{prefix}{key}",
                                  values,
                                  global_step=global_step)


class LinearProj(nn.Module):
    """
    Use linear projection to get the probablity distribution
    """

    #! now the size of linear projection is fixed: (768, 32)

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 32)

    def forward(self, x):
        x = self.linear(x)
        return x


class DistillerForPretrain(nn.Module):
    """
    Distiller for pretraining
    """
    def __init__(self, config: DistillerConfig, teacher_config: edict):
        super().__init__()
        self.config = config
        self.distiller = DistillerModel(
            config
        )  #! build the distiller model #! what is the shape of config
        print("[DistillerModel] - The structure of distiller model")
        print(self.distiller)

        self.teacher_config = teacher_config
        # teacher = torch.hub.load("s3prl/s3prl", teacher_config.model) #! get the teacher model

        #! get the model locally(first model)
        if teacher_config.model.find("wav2vec2") >= 0:
            print("[DistillerForPretrain] - use wav2vec2 as teacher model")
            from upstream.wav2vec2.expert import UpstreamExpert
            teacher = UpstreamExpert(teacher_config.model_path)
            linear_projection = LinearProj()
            linear_projection.load_state_dict(
                torch.load(teacher_config.linear_projection_path))
        elif teacher_config.model.find("data2vec") >= 0:
            print("[DistillerForPretrain] - use data2vec as teacher model")
            from upstream.data2vec.expert import UpstreamExpert
            teacher = UpstreamExpert(teacher_config.model_path)
            linear_projection = LinearProj()
            linear_projection.load_state_dict(
                torch.load(teacher_config.linear_projection_path))
        else:
            print("[DistillerForPretrain] - use hubert as teacher model")
            from upstream.hubert.expert import UpstreamExpert
            teacher = UpstreamExpert(teacher_config.model_path)
            linear_projection = LinearProj()
            linear_projection.load_state_dict(
                torch.load(teacher_config.linear_projection_path))
        print(teacher.model)
        print(f"[DistillerForPretrain] - load linear projection from {teacher_config.linear_projection_path}")
        print(linear_projection)

        #! get the model2 locally(first model)
        if teacher_config.model2.find("wav2vec2") >= 0:
            print("[DistillerForPretrain] - use wav2vec2 as teacher model 2")
            from upstream.wav2vec2.expert import UpstreamExpert
            teacher_2 = UpstreamExpert(teacher_config.model_path_2)
            linear_projection_2 = LinearProj()
            linear_projection_2.load_state_dict(
                torch.load(teacher_config.linear_projection_path_2))
        elif teacher_config.model.find("data2vec") >= 0:
            print("[DistillerForPretrain] - use data2vec as teacher model 2")
            from upstream.data2vec.expert import UpstreamExpert
            teacher_2 = UpstreamExpert(teacher_config.model_path_2)
            linear_projection_2 = LinearProj()
            linear_projection_2.load_state_dict(
                torch.load(teacher_config.linear_projection_path_2))
        else:
            print("[DistillerForPretrain] - use hubert as teacher model 2")
            from upstream.hubert.expert import UpstreamExpert
            teacher_2 = UpstreamExpert(teacher_config.model_path_2)
            linear_projection_2 = LinearProj()
            linear_projection_2.load_state_dict(
                torch.load(teacher_config.linear_projection_path_2))
        print(teacher_2.model)
        print(f"[DistillerForPretrain] - load linear projection 2 from {teacher_config.linear_projection_path_2}")
        print(linear_projection_2)

        if (teacher_config.model.find("hubert") >= 0
                or teacher_config.model.find("wav2vec2") >= 0
                or teacher_config.model.find("data2vec") >= 0):
            teacher.model.encoder.layerdrop = 0
            print(
                "[DistillerForPretrain] - Disabled teacher's encoder layerdrop"
            )

        # if(teacher_config.model.find("data2vec") >= 0):
        #     print(teacher.model.state_dict)
        #     teacher.model.cfg.task.normalize = False

        assert self.distiller.n_tasks <= teacher_config.n_layers, (
            self.distiller.n_tasks,
            teacher_config.n_layers,
        )
        self.teacher = teacher
        self.teacher_2 = teacher_2
        self.linear_projection = linear_projection
        self.linear_projection_2 = linear_projection_2
        freeze_model(self.teacher)
        freeze_model(self.linear_projection)
        freeze_model(self.teacher_2)
        freeze_model(self.linear_projection_2)

        print("[DistillerForPretrain] - Using {} as teacher with {} layers".
              format(teacher_config.model, teacher_config.n_layers))

        if config.loss_type == "l1":  #! use l1 loss
            self.loss_func = nn.L1Loss(reduction="none")
        elif config.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(config.loss_type)

        self.cosine_loss = config.cosine_loss  #! 1.0
        if self.cosine_loss > 0:
            print("[DistillerForPretrain] - Enabled cosine similarity loss.")

        self.hidden_loss = config.hidden_loss  #! 1.0
        if self.hidden_loss > 0:
            print("[DistillerForPretrain] - Enabled hidden loss.")

        self.attn_loss = config.attn_loss  #! 1.0
        if self.attn_loss > 0:
            print("[DistillerForPretrain] - Enabled attn loss.")

        self.kldiv_loss = config.kldiv_loss  #! 1.0
        if self.kldiv_loss > 0:
            print("[DistillerForPretrain] - Enabled kldiv loss.")
        
        self.kldiv_loss_2 = config.kldiv_loss  #! 1.0
        if self.kldiv_loss_2 > 0:
            print("[DistillerForPretrain] - Enabled kldiv loss 2.")

        self.temperature = config.temperature

        #! copy value from teacher
        if config.init_teacher_conv_layers:
            print("[DistillerForPretrain] - "
                  "Initializing feature extractor from teacher")
            self.distiller.feature_extractor.load_state_dict(
                self.teacher.model.feature_extractor.state_dict(),
                strict=False)
            if self.distiller.post_extract_proj is not None:
                self.distiller.post_extract_proj.load_state_dict(
                    self.teacher.model.post_extract_proj.state_dict())

        if config.init_teacher_encoder_layers:
            print("[DistillerForPretrain] - "
                  "Initializing encoder from teacher")
            self.distiller.encoder.pos_conv.load_state_dict(
                self.teacher.model.encoder.pos_conv.state_dict(), strict=False)
            for l in range(config.encoder_layers):  #! 2
                self.distiller.encoder.layers[l].load_state_dict(
                    self.teacher.model.encoder.layers[l].state_dict())

    def forward(
        self,
        wave_input: torch.Tensor,
        wave_orig: list,
        wave_len: torch.Tensor,
        pad_mask: torch.Tensor,
        return_other: bool = False,
    ):
        """
        Forward function.
        Input:
            wave_input: FloatTensor (B x T_wave)
            wave_orig: List of FloatTensor
            wave_len: LongTensor (B)
            pad_mask: FloatTensor (B x T)
            return_other: Bool (returns other information for logging)
        """
        return_other = False
        # Forward model
        feat, feat_final, pred, pad_mask, student_logits = self.distiller(
            wave_input, pad_mask)  #! distiller model(the small one)
        #! feat [12, 752, 512] BNxTxD   feat_final [12, 752, 768]  pred [12, 2, 752, 768] BxNxTxD
        #! feat: after conv  feat_final: after proj  pred: after transformers

        with torch.no_grad():
            wave_orig = [wave.to(wave_input.device) for wave in wave_orig]
            with torch.cuda.amp.autocast(False):
                teacher_hiddens = self.teacher(wave_orig)
                teacher_hiddens_2 = self.teacher_2(wave_orig)
                x = teacher_hiddens["hidden_states"][-1].transpose(0, 1)  # B x T x C -> T x B x C
                x_2 = teacher_hiddens_2["hidden_states"][-1].transpose(0, 1)  # B x T x C -> T x B x C
            if self.config.task_emb_type == "none":
                teacher_hiddens = teacher_hiddens["hidden_states"][
                    self.config.n_tasks]
                teacher_hiddens = teacher_hiddens.unsqueeze(1)
            else:
                if self.config.task_emb_type in [
                        "expand-last", "hnet", "self-hidden"
                ]:
                    teacher_hiddens = [
                        teacher_hiddens["hidden_states"][i]
                        for i in self.distiller.pred_layer_id
                    ]
                else:
                    teacher_hiddens = teacher_hiddens["hidden_states"][1:]
                teacher_hiddens = torch.stack(teacher_hiddens,
                                              dim=1)  # B x N x T x D

            teacher_logits = self.linear_projection(x)  # T x B x kinds
            teacher_logits_2 = self.linear_projection_2(x) # T x B x kinds

        (
            total_loss,
            rec_loss,
            rec_layer_loss,
            feat_pen,
            sim_loss,
            sim_layer_loss,
            kldiv_loss,
            kldiv_loss_2
        ) = self.compute_loss(
            feat, pred, teacher_hiddens, teacher_logits, teacher_logits_2, student_logits,
            return_other
        )  #! feat [12, 752, 512]  pred [12, 2, 752, 768]  teacher_hiddens [12, 2, 752, 768]

        wandb.log({
            "total_loss": total_loss,
            "rec_loss": rec_loss,
            "sim_loss": sim_loss,
            "kldiv_loss": kldiv_loss,
            "kldiv_loss_2": kldiv_loss_2
        })

        for i, item in enumerate(rec_layer_loss):
            wandb.log({f"rec_layer_loss_{i}": item})

        for i, item in enumerate(sim_layer_loss):
            wandb.log({f"sim_layer_loss_{i}": item})

        return_other = False

        if return_other:
            with torch.no_grad():
                other_res = {
                    "rec_loss": rec_loss,
                    "feat_pen": feat_pen,
                    "sim_loss": sim_loss,
                    "norm_feat_final": feat_final.pow(2).mean(),
                }
                teacher_norm = torch.abs(teacher_hiddens).mean((0, 2, 3))
                if self.config.task_emb_type == "none":
                    other_res[f"rec_l{self.config.n_tasks}"] = rec_layer_loss[
                        0]
                    other_res[
                        f"tar_norm_l{self.config.n_tasks}"] = teacher_norm[0]
                    if sim_layer_loss is not None:
                        other_res[
                            f"sim_l{self.config.n_tasks}"] = sim_layer_loss[0]
                else:
                    for i in range(self.config.n_tasks):
                        layer_id = i + 1
                        if self.config.task_emb_type in [
                                "expand-last",
                                "hnet",
                                "self-hidden",
                        ]:
                            layer_id = self.distiller.pred_layer_id[i]
                        other_res[f"rec_l{layer_id}"] = rec_layer_loss[i]
                        other_res[f"tar_norm_l{layer_id}"] = teacher_norm[i]
                        if sim_layer_loss is not None:
                            other_res[f"sim_l{layer_id}"] = sim_layer_loss[i]
                    if self.config.task_emb_type not in [
                            "expand-last",
                            "hnet",
                            "self-hidden",
                    ]:
                        other_res[
                            "norm_task_emb"] = self.distiller.task_embedding.weight.pow(
                                2).mean()
        else:
            other_res = None

        return total_loss, other_res

    def compute_loss(self,
                     feat,
                     pred,
                     target,
                     teacher_logits,
                     teacher_logits_2,
                     student_logits,
                     return_other=False):
        """
        Computes loss.
        Inputs:
            feat: B x T x D
            pred: B x N x T x D
            target: B x N x T x D
        """
        #! why feat is not same as pred in last dimension(because it is the output of conv)
        # Reconstruction loss
        assert pred.shape == target.shape, (pred.shape, target.shape)
        rec_loss = self.loss_func(pred, target)  # B x N x T x D #! L1 loss

        with torch.no_grad():
            rec_layer_loss = rec_loss.mean((0, 2, 3))

        rec_loss = rec_loss.mean()

        # Cosine similarity loss
        if self.cosine_loss > 0:
            sim_loss = -F.logsigmoid(
                F.cosine_similarity(pred, target, dim=-1)
            )  #! what is the dimension of every val? # sim_loss [12, 2, 752]
            # B x N x T
            with torch.no_grad():
                sim_layer_loss = sim_loss.mean((0, 2))
            sim_loss = sim_loss.mean()
        else:
            sim_loss = 0
            sim_layer_loss = None

        # Feature loss
        feat_pen = feat.float().pow(
            2).mean()  #? what is feature loss? (maybe it is not important)

        teacher_prob_log = F.log_softmax(teacher_logits, dim=-1)
        teacher_prob_log_2 = F.log_softmax(teacher_logits_2, dim=-1)
        student_prob_log = F.log_softmax(student_logits, dim=-1)

        prob_size = teacher_prob_log.shape
        teacher_prob_log = teacher_prob_log.view(
            (prob_size[0] * prob_size[1], prob_size[2]))
        teacher_prob_log_2 = teacher_prob_log_2.view(
            (prob_size[0] * prob_size[1], prob_size[2]))
        student_prob_log = student_prob_log.view(
            (prob_size[0] * prob_size[1], prob_size[2]))

        # kldiv loss
        kldiv_loss = F.kl_div(input=student_prob_log / self.temperature,
                              target=teacher_prob_log / self.temperature,
                              log_target=True,
                              reduction='batchmean') * self.temperature**2

        kldiv_loss_2 = F.kl_div(input=student_prob_log / self.temperature,
                                target=teacher_prob_log_2 / self.temperature,
                                log_target=True,
                                reduction='batchmean') * self.temperature**2

        total_loss = (rec_loss + feat_pen * self.config.feat_pen_loss +
                      sim_loss * self.cosine_loss +
                      kldiv_loss * self.kldiv_loss +
                      kldiv_loss_2 * self.kldiv_loss_2)

        return total_loss, rec_loss, rec_layer_loss, feat_pen, sim_loss, sim_layer_loss, kldiv_loss, kldiv_loss_2

    def compute_loss_hidden(self,
                            feat,
                            pred,
                            target,
                            teacher_hiddens,
                            teacher_attns,
                            student_hiddens,
                            student_attns,
                            return_other=False):
        """
        Computes loss.
        Inputs:
            feat: B x T x D
            pred: B x N x T x D
            target: B x N x T x D
            student_hidden: B x N x T x D
            student_attn: B x N x T x D
        """
        # Reconstruction loss
        assert pred.shape == target.shape, (pred.shape, target.shape)
        rec_loss = self.loss_func(pred, target)  # B x N x T x D #! L1 loss

        with torch.no_grad():
            rec_layer_loss = rec_loss.mean((0, 2, 3))

        rec_loss = rec_loss.mean()

        # Cosine similarity loss
        if self.cosine_loss > 0:
            sim_loss = -F.logsigmoid(
                F.cosine_similarity(pred, target, dim=-1)
            )  #! what is the dimension of every val? # sim_loss [12, 2, 752]
            # B x N x T
            with torch.no_grad():
                sim_layer_loss = sim_loss.mean((0, 2))
            sim_loss = sim_loss.mean()
        else:
            sim_loss = 0
            sim_layer_loss = None

        # hidden loss
        assert teacher_hiddens.shape == student_hiddens.shape, (
            teacher_hiddens.shape, student_hiddens.shape)
        hidden_loss = self.loss_func(
            teacher_hiddens, student_hiddens)  # B x N x T x D #! L1 loss

        with torch.no_grad():
            hidden_layer_loss = hidden_loss.mean((0, 2, 3))
        hidden_loss = hidden_loss.mean()

        # attn loss
        attn_loss = None
        attn_layer_loss = None
        # assert teacher_attns.shape == student_attns.shape, (teacher_attns.shape, student_attns.shape)
        # attn_loss = self.loss_func(teacher_attns, student_attns)  # B x N x T x D #! L1 loss

        # if return_other:
        #     with torch.no_grad():
        #         attn_layer_loss = attn_loss.mean((0, 2, 3))
        # else:
        #     attn_layer_loss = None
        # attn_loss = attn_loss.mean()

        # Feature loss
        feat_pen = feat.float().pow(
            2).mean()  #? what is feature loss? (maybe it is not important)

        total_loss = 0

        total_loss = (rec_loss + feat_pen * self.config.feat_pen_loss +
                      sim_loss * self.cosine_loss +
                      hidden_loss * self.hidden_loss)

        return total_loss, rec_loss, rec_layer_loss, feat_pen, sim_loss, sim_layer_loss, hidden_loss, hidden_layer_loss, attn_loss, attn_layer_loss
