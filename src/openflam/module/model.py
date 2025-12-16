"""
CLAP Model and FLAM Model
--------------------------------------------------------
Paper: https://arxiv.org/abs/2505.05335
Code Maintainers: Ke Chen, Yusong Wu, Oriol Nieto, Prem Seetharaman
Support: Adobe Research
"""

from typing import Optional, Dict
import abc
import warnings

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchlibrosa.augmentation import SpecAugmentation

from dataclasses import dataclass
from transformers import RobertaModel, RobertaTokenizer
from .htsat import create_htsat_model, replace_locality_aware_attention
from .contrastive_loss import CLIPLoss, FramewiseContrastiveLoss, PriorLoss

AUDIO_EMB_KEY = "audio_emb"
AUDIO_EMB_MASK_KEY = "audio_emb_mask"


def get_device(verbose=False) -> torch.device:
    # Adapted from fad.py in aeval
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )
    if device == torch.device("mps"):
        if verbose:
            print("[CLAP Test] CLAP does not support MPS device yet, because:")
            print(
                "[CLAP Test] The operator 'aten::upsample_bicubic2d.out' is not currently implemented for the MPS device."
            )
            print("[CLAP Test] Using CPU device instead.")
        device = torch.device("cpu")
    if verbose:
        print("[CLAP Test] Using device: {}".format(device))
    return device


# Audio Config Class


class BaseCLAP(abc.ABC):
    """Abstract class for CLAP API."""

    @abc.abstractmethod
    def encode_text(
        self, text: list[str], device: torch.device
    ) -> torch.Tensor:
        """Encode text to get text branch output."""
        pass

    @abc.abstractmethod
    def encode_audio(
        self, audio: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Encode audio to get audio branch output."""
        pass

    @abc.abstractmethod
    def get_audio_features(
        self, audio: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get normalized audio global audio features."""
        pass

    @abc.abstractmethod
    def get_text_features(self, text: list[str]) -> torch.Tensor:
        """Get normalized text global text features."""
        pass

    @abc.abstractmethod
    def forward(
        self,
        audio: dict[str, torch.Tensor],
        text: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Forward pass to get audio or text embedding."""
        pass

    @abc.abstractmethod
    def get_logit_scale(self) -> torch.Tensor | float:
        """Get logit scale."""
        pass


@dataclass
class CLAPAudioCfg:
    model_type: str = "HTSAT"
    model_name: str = "tiny"
    sample_rate: int = 48000
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 480
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 480000
    audio_embed_dim: int = 768
    enable_fusion: bool = False
    fusion_type: str = "None"
    # Following are only effective for pretrained audio tokenizer
    connector_width: int = 512
    connector_heads: int = 8
    connector_layers: int = 8
    emb_source: str = "model"  # model or batch
    # For HTSAT
    locality_aware_attention: bool = False


# Text Config Class
@dataclass
class CLAPTextCfg:
    context_length: int = 77
    model_type: str = "roberta"
    # Following are only effective for pretrained text encoder
    connector_width: int = 512
    connector_heads: int = 8
    connector_layers: int = 8


class CLAP(nn.Module, BaseCLAP):
    def __init__(
        self,
        audio_cfg: CLAPAudioCfg,
        text_cfg: CLAPTextCfg,
        joint_embed_shape: int = 512,
        fabric: Optional[L.Fabric] = None,
        cache_dir: str = None,
    ):
        super().__init__()
        if isinstance(audio_cfg, dict):
            audio_cfg = CLAPAudioCfg(**audio_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLAPTextCfg(**text_cfg)

        self.audio_cfg = audio_cfg
        self.text_cfg = text_cfg
        self.joint_embed_shape = joint_embed_shape
        self.fabric = fabric

        self.context_length = text_cfg.context_length

        mlp_act_layer = nn.ReLU()

        # audio branch
        self.create_audio_encoder(audio_cfg, mlp_act_layer)
        self.audio_branch_type = audio_cfg.model_type

        # text branch, (yusongw port): removed the support of original CLIP transformer
        self.create_text_encoder(text_cfg.model_type, mlp_act_layer, cache_dir)
        self.text_branch_type = text_cfg.model_type

        self.register_buffer(
            "attn_mask", self.build_attention_mask(), persistent=False
        )

        self.init_logit_scale()
        self.setup_loss_fn()

    def setup_loss_fn(self):
        if self.fabric is None:
            raise ValueError("Fabric is not provided.")
        self.loss_fn = CLIPLoss(self.fabric)

    def create_audio_encoder(
        self,
        audio_cfg: CLAPAudioCfg,
        mlp_act_layer: nn.Module,
    ) -> nn.Module:
        if audio_cfg.model_type == "HTSAT":
            self.audio_branch = create_htsat_model(
                audio_cfg, audio_cfg.enable_fusion, audio_cfg.fusion_type
            )

            if audio_cfg.locality_aware_attention:
                self.audio_branch = replace_locality_aware_attention(
                    self.audio_branch
                )

            self.audio_projection = nn.Sequential(
                nn.Linear(audio_cfg.audio_embed_dim, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        else:
            logging.error(
                f"Audio Model config for {audio_cfg.model_type} not found"
            )
            raise RuntimeError(
                f"Audio Model config for {audio_cfg.model_type} not found."
            )

    def create_text_encoder(
        self, model_type: str, mlp_act_layer: nn.Module, cache_dir: str
    ) -> nn.Module:
        if model_type == "roberta":
            self.text_branch = RobertaModel.from_pretrained(
                "roberta-base", cache_dir=cache_dir
            )
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
            self.tokenizer = RobertaTokenizer.from_pretrained(
                "roberta-base", cache_dir=cache_dir
            )
        else:
            logging.error(f"Text Model config for {model_type} not found")
            raise RuntimeError(f"Text Model config for {model_type} not found.")

    def init_logit_scale(self):
        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @torch.no_grad()
    def tokenize(self, text):
        result = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.text_cfg.context_length,
            return_tensors="pt",
        )
        return result

    def encode_audio(self, audio, device):
        """Audio to audio branch output."""
        if self.audio_branch_type == "HTSAT":
            return self.audio_branch(audio, mixup_lambda=None, device=device)[
                "embedding"
            ]
        else:
            logging.error(f"Model type {self.audio_branch_type} not found")
            raise RuntimeError(
                f"Model type {self.audio_branch_type} not found."
            )

    def encode_local_audio(self, audio, device):
        if self.audio_branch_type == "HTSAT":
            return self.audio_branch(audio, mixup_lambda=None, device=device)[
                "fine_grained_embedding"
            ]
            # downsample to 32 frames because each 32 frames are identical
        else:
            logging.error(f"Model type {self.audio_branch_type} not found")
            raise RuntimeError(
                f"Model type {self.audio_branch_type} not found."
            )

    def encode_text(self, text, device):
        """Text to text branch output."""
        text = self.tokenize(text)
        if self.text_branch_type == "roberta":
            text_branch_output = self.text_branch(
                input_ids=text["input_ids"].to(
                    device=device, non_blocking=True
                ),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
            )["pooler_output"]
        else:
            logging.error(f"Model type {self.text_branch_type} not found")
            raise RuntimeError(f"Model type {self.text_branch_type} not found.")

        return text_branch_output

    def get_logit_scale(self):
        return self.logit_scale_a.exp()

    def contrastive_loss(self, audio, text, device=None):
        metrics = {}
        (
            audio_features,
            text_features,
            logit_scale_a,
        ) = self._forward(audio, text, device=device)
        total_loss = self.loss_fn(
            audio_features=audio_features,
            text_features=text_features,
            logit_scale_a=logit_scale_a,
        )

        metrics["loss"] = total_loss
        metrics["logit_scale"] = logit_scale_a
        metrics["contrastive_loss"] = total_loss
        return metrics

    def loss(self, audio, text, device=None, batch=None):
        return self.contrastive_loss(audio, text, device=device)

    @staticmethod
    def get_audio_input(
        audio: torch.Tensor,
        audio_emb: Optional[torch.Tensor] = None,
        audio_emb_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference helper function to get audio input for the model."""
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        audio_input = {"waveform": audio}
        if audio_emb is not None:
            audio_input[AUDIO_EMB_KEY] = audio_emb
        if audio_emb_mask is not None:
            audio_input[AUDIO_EMB_MASK_KEY] = audio_emb_mask
        return audio_input

    def get_audio_features(
        self,
        audio: torch.Tensor,
        audio_emb: Optional[torch.Tensor] = None,
        audio_emb_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        audio_input = self.get_audio_input(audio, audio_emb, audio_emb_mask)
        audio_features = self.audio_projection(
            self.encode_audio(audio_input, device=device)
        )
        audio_features = F.normalize(audio_features, dim=-1)
        return audio_features

    def get_local_audio_features(
        self,
        audio: torch.Tensor,
        audio_emb: Optional[torch.Tensor] = None,
        audio_emb_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        audio_input = self.get_audio_input(audio, audio_emb, audio_emb_mask)
        audio_features = self.audio_projection(
            self.encode_local_audio(audio_input, device=device)
        )
        audio_features = F.normalize(audio_features, dim=-1)
        return audio_features

    def get_text_features(self, text: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        text_branch_output = self.encode_text(text, device=device)
        text_features = self.text_projection(text_branch_output)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def validate(self, batch):
        return {}

    def _forward(self, audio, text, device=None):
        """Forward audio and text into the CLAP

        Parameters
        ----------
        audio: torch.Tensor (batch_size, audio_length)
            the time-domain audio input / the batch of mel_spec and longer list.
        text: list[str]
            the text token input
        device: torch.device (optional)
            the device to run the model
        """
        if device is None:
            if audio is not None:
                device = audio["waveform"].device
            elif text is not None:
                print("No device specified, automatically selecting device.")
                device = get_device()
        if audio is None and text is None:
            # a hack to get the logit scale
            return self.logit_scale_a.exp()
        elif audio is None:
            return self.text_projection(self.encode_text(text, device=device))
        elif text is None:
            return self.audio_projection(
                self.encode_audio(audio, device=device)
            )
        audio_features = self.audio_projection(
            self.encode_audio(audio, device=device)
        )
        audio_features = F.normalize(audio_features, dim=-1)

        text_branch_output = self.encode_text(text, device=device)
        text_features = self.text_projection(text_branch_output)
        text_features = F.normalize(text_features, dim=-1)

        return (
            audio_features,
            text_features,
            self.logit_scale_a.exp(),
        )

    def forward(self, audio, text, device=None, return_loss=False, batch=None):
        """Forward audio and text into the CLAP.

        Parameters
        ----------
        audio: torch.Tensor (batch_size, audio_length)
            the time-domain audio input / the batch of mel_spec and longer list.
        text: list[str]
            the text token input
        device: torch.device (optional)
            the device to run the model
        return_loss: bool
            whether to return the loss
        """
        if return_loss:
            return self.loss(audio, text, device=device, batch=batch)
        return self._forward(audio, text, device=device)


class FLAM(CLAP):
    """Joint training global contrastive and local contrastive (framewise contrastive) CLAP."""

    def __init__(
        self,
        audio_cfg: CLAPAudioCfg,
        text_cfg: CLAPTextCfg,
        joint_embed_shape: int = 512,
        fabric: Optional[L.Fabric] = None,
        sed_loss_weight: float = 200.0,
        no_global_loss: bool = False,
        sed_clip_logit_scale: float = -1.0,
        per_text_logit_bias_init: float = -8,
        per_text_logit_scale_init: float = 10.0,
        cache_dir: str = None,
    ):
        super().__init__(
            audio_cfg=audio_cfg,
            text_cfg=text_cfg,
            joint_embed_shape=joint_embed_shape,
            fabric=fabric,
            cache_dir=cache_dir,
        )

        self.sed_loss_weight = sed_loss_weight
        self.no_global_loss = no_global_loss
        self.sed_clip_logit_scale = sed_clip_logit_scale
        self.per_text_logit_bias_init = per_text_logit_bias_init
        self.per_text_logit_scale_init = per_text_logit_scale_init

        self.init_aed_logit_scale_bias()
        self.init_per_text_logit_scale_bias()
        self.init_sed_loss_fn()

        # remove spec augmenter for SED
        self.audio_branch.spec_augmenter = SpecAugmentation(
            time_drop_width=0,
            time_stripes_num=0,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

    def init_sed_loss_fn(self):
        self.local_contrastive_loss_fn = FramewiseContrastiveLoss(self.fabric)
        self.prior_loss = PriorLoss(self.fabric)

    def init_aed_logit_scale_bias(self):
        self.aed_logit_bias = 0
        self.aed_logit_scale = nn.Parameter(torch.zeros([]))  # exp(0) = 1
        self.aed_logit_scale.requires_grad = False

    def init_per_text_logit_scale_bias(self):
        text_branch_output_dim = self.text_projection[0].in_features
        self.fc_per_text_logit_bias = nn.Sequential(
            nn.Linear(text_branch_output_dim, text_branch_output_dim),
            nn.GELU(),
            nn.Linear(text_branch_output_dim, 1),
        )
        # add self.per_text_logit_bias_init to the final bias layer of the fc_per_text_logit_bias
        self.fc_per_text_logit_bias[-1].bias.data.add_(
            self.per_text_logit_bias_init
        )
        self.fc_per_text_logit_scale = nn.Sequential(
            nn.Linear(text_branch_output_dim, text_branch_output_dim),
            nn.GELU(),
            nn.Linear(text_branch_output_dim, 1),
        )
        # add self.per_text_logit_scale_init to the final bias layer of the fc_per_text_logit_scale
        self.fc_per_text_logit_scale[-1].bias.data.add_(
            np.log(self.per_text_logit_scale_init)
        )

    def encode_local_and_global_audio(self, audio, device):
        output = self.audio_branch(audio, mixup_lambda=None, device=device)[
            "fine_grained_embedding"
        ]
        # downsample to 32 frames because each 32 frames are identical
        local_audio_features = output
        global_audio_features = local_audio_features.mean(dim=1)
        return local_audio_features, global_audio_features

    def encode_audio(self, audio, device):
        return self.audio_branch(audio, mixup_lambda=None, device=device)[
            "embedding"
        ]

    def encode_local_audio(self, audio, device):
        return self.audio_branch(audio, mixup_lambda=None, device=device)[
            "fine_grained_embedding"
        ]

    def _get_features(self, audio, text, device=None):
        local_audio_features, global_audio_features = (
            self.encode_local_and_global_audio(audio, device)
        )
        local_audio_features = self.audio_projection(local_audio_features)
        global_audio_features = self.audio_projection(global_audio_features)
        text_branch_output = self.encode_text(text, device=device)
        text_features = self.text_projection(text_branch_output)

        local_audio_features = F.normalize(local_audio_features, dim=-1)
        global_audio_features = F.normalize(global_audio_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        all_features = {
            "local_audio_features": local_audio_features,
            "global_audio_features": global_audio_features,
            "text_features": text_features,
            "text_branch_output": text_branch_output,
        }

        return all_features

    def _get_inputs(self, batch):
        audio = {"waveform": batch["wav"].squeeze(1)}
        if AUDIO_EMB_KEY in batch and AUDIO_EMB_MASK_KEY in batch:
            audio[AUDIO_EMB_KEY] = batch[AUDIO_EMB_KEY]
            audio[AUDIO_EMB_MASK_KEY] = batch[AUDIO_EMB_MASK_KEY]
        texts = batch["caption"]
        return audio, texts

    def global_contrastive_loss(self, batch, device):
        metrics = {}

        audio, text = self._get_inputs(batch)

        all_features = self._get_features(audio, text, device=device)

        global_audio_features = all_features["global_audio_features"]
        text_features = all_features["text_features"]

        logit_scale_a = self.logit_scale_a.exp()

        # global contrastive loss
        global_contrastive_loss = self.loss_fn(
            audio_features=global_audio_features,
            text_features=text_features,
            logit_scale_a=logit_scale_a,
        )
        metrics["logit_scale"] = logit_scale_a
        metrics["global_contrastive_loss"] = global_contrastive_loss
        return global_contrastive_loss, metrics

    def local_contrastive_loss(self, batch, device):
        metrics = {}

        audio, text = self._get_inputs(batch)

        all_features = self._get_features(audio, text, device=device)

        local_audio_features = all_features["local_audio_features"]
        text_features = all_features["text_features"]
        text_branch_output = all_features["text_branch_output"]

        # local contrastive loss
        if self.sed_clip_logit_scale > 0:
            self.aed_logit_scale.data = torch.clamp(
                self.aed_logit_scale, max=np.log(self.sed_clip_logit_scale)
            )

        per_text_logit_bias = self.fc_per_text_logit_bias(
            text_branch_output
        ).squeeze()

        per_text_logit_scale = (
            self.fc_per_text_logit_scale(text_branch_output).exp().squeeze()
        )

        # With prior loss, stop the per_text_logit_bias gradient
        # from the local contrastive loss
        per_text_logit_bias_inp = per_text_logit_bias.detach()

        sed_loss_metric = self.local_contrastive_loss_fn(
            local_audio_features,
            text_features,
            self.aed_logit_scale.exp(),
            batch["framewise_label"],
            batch["audio_mask"],
            batch["text_mask"],
            self.aed_logit_bias,
            per_text_logit_bias=per_text_logit_bias_inp,
            per_text_logit_scale=per_text_logit_scale,
        )
        local_contrastive_loss = sed_loss_metric["loss"] * self.sed_loss_weight
        metrics["local_contrastive_loss"] = local_contrastive_loss
        metrics["aed_logit_bias_overall"] = sed_loss_metric["logit_bias"]
        metrics["aed_logit_bias_global"] = self.aed_logit_bias
        metrics["aed_logit_scale"] = self.aed_logit_scale.exp()

        metrics["per_text_logit_bias_mean"] = per_text_logit_bias.mean()
        metrics["per_text_logit_bias_std"] = per_text_logit_bias.std()

        metrics["per_text_logit_scale_mean"] = per_text_logit_scale.mean()
        metrics["per_text_logit_scale_std"] = per_text_logit_scale.std()

        prior_loss = self.prior_loss(
            per_text_logit_bias,
            batch["framewise_label"],
            batch["audio_mask"],
            batch["text_mask"],
        )
        metrics["prior_loss"] = prior_loss
        local_contrastive_loss += prior_loss

        return local_contrastive_loss, metrics

    def loss(self, audio, text, device=None, batch=None):
        metrics = {}

        if self.no_global_loss:
            global_contrastive_loss = 0
        else:
            global_batch = batch["global_batch"]
            global_contrastive_loss, global_metrics = (
                self.global_contrastive_loss(global_batch, device)
            )
            metrics.update(global_metrics)

        if "sed_batch" in batch:
            local_batch = batch["sed_batch"]
        else:
            local_batch = batch
        local_contrastive_loss, local_metrics = self.local_contrastive_loss(
            local_batch, device
        )
        metrics.update(local_metrics)

        metrics["loss"] = global_contrastive_loss + local_contrastive_loss
        return metrics

    def forward(self, audio, text, device=None, return_loss=False, batch=None):
        """Forward audio and text into the CLAP.

        Parameters
        ----------
        audio: torch.Tensor (batch_size, audio_length)
            the time-domain audio input / the batch of mel_spec and longer list.
        text: list[str]
            the text token input
        device: torch.device (optional)
            the device to run the model
        return_loss: bool
            whether to return the loss
        batch: dict (optional)
            the batch data containing other labels
        """
        if return_loss:
            return self.loss(audio, text, device=device, batch=batch)
        return self._forward(audio, text, device=device)
