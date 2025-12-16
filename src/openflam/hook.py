"""
OpenFLAM Hook
--------------------------------------------------------
Paper: https://arxiv.org/abs/2505.05335
Code Maintainers: Ke Chen, Yusong Wu, Oriol Nieto, Prem Seetharaman
Support: Adobe Research
"""

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from huggingface_hub import hf_hub_download
from module.model import CLAPTextCfg, CLAPAudioCfg, FLAM


def get_approximate_framewise_act_map(
    local_audio_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    cross_product: bool = False,
) -> torch.Tensor:
    """Get the framewise attention map between audio and text embeddings using approximate method.
        This function corresponds to the Eq. 8 in the paper.
        This function is an approximation of the unbiased framewise attention map (Eq. 9) in the paper,
            without the need for computing the per-text logit bias ($\beta(t)$).
        The approximation is valid when the per-text logit bias is small (e.g., $\beta(t) < -10$).

    Args:
        local_audio_features (batch, T, dim): Tensor containing local audio features.
        text_features (batch, dim): Tensor containing text features.
        logit_scale (torch.Tensor): Per-text logit scale ($\alpha(t)$), in shape [num_text].
        cross_product (bool): Whether to compute the cross product of audio and text embeddings,
            i.e, if True, the output will be the similarity of each audio detected by all text,
            otherwise, the output will be the similarity of each audio detected by corresponding text of the same index.
    Returns:
        torch.Tensor (batch, T) or (batch, num_text, T): Tensor representing the local similarity between audio and text embeddings.
            If cross_product is True, the output will be in shape (batch, num_text, T),
            otherwise, the output will be in shape (batch, T).
    """

    local_audio_features = F.normalize(local_audio_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    if cross_product:
        sim = torch.einsum("btd,cd->bct", local_audio_features, text_features)
        logits = logit_scale[None, :, None] * sim
    else:
        if local_audio_features.size(0) != text_features.size(0):
            raise ValueError(
                "The number of local audio features should be the same as the number of text features."
            )
        sim = torch.einsum("btd,bd->bt", local_audio_features, text_features)
        logits = logit_scale[:, None] * sim

    out = torch.sigmoid(logits)
    return out


def get_unbiased_framewise_act_map(
    local_audio_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    cross_product: bool = False,
) -> torch.Tensor:
    """Get the output map between audio and text embeddings. According to logit correction.
        This function corresponds to the Eq. 7 in the paper.

    Args:
        local_audio_features (batch, T, dim): Tensor containing local audio features.
        text_features (batch, dim): Tensor containing text features.
        cross_product (bool): Whether to compute the cross product of audio and text embeddings,
            i.e, if True, the output will be the similarity of each audio detected by all text,
            otherwise, the output will be the similarity of each audio detected by corresponding text of the same index.
        logit_scale (torch.Tensor): Per-text logit scale ($\alpha(t)$), in shape [num_text].
        logit_bias (torch.Tensor): Per-text logit bias ($\beta(t)$), in shape [num_text].
    Returns:
        torch.Tensor (batch, T) or (batch, num_text, T): Tensor representing the local similarity between audio and text embeddings.
            If cross_product is True, the output will be in shape (batch, num_text, T),
            otherwise, the output will be in shape (batch, T).
    """

    assert (
        logit_bias.shape[0] == text_features.shape[0]
    ), "logit_bias shape should be equal to text_features shape"

    local_audio_features = F.normalize(local_audio_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    if cross_product:
        sim = torch.einsum("btd,cd->bct", local_audio_features, text_features)
        logit_bias = logit_bias[None, :, None]
        logits = logit_scale[None, :, None] * sim + logit_bias
    else:
        if local_audio_features.size(0) != text_features.size(0):
            raise ValueError(
                "The number of local audio features should be the same as the number of text features."
            )
        sim = torch.einsum("btd,bd->bt", local_audio_features, text_features)
        logit_bias = logit_bias[:, None]
        logits = logit_scale[:, None] * sim + logit_bias

    prob = F.sigmoid(logits)  # biased output p(z=1|x,y)
    out = prob / (prob + F.sigmoid(logit_bias))  # unbiased output s(x,y)
    return out


class OpenFLAM(torch.nn.Module):
    def __init__(
        self,
        model_name="v1-base",
        audio_cfg=None,
        text_cfg=None,
        default_ckpt_path="/tmp/openflam",
        **kwargs,
    ) -> None:
        """The init of OpenFLAM model

        Args:
            model_name (str, optional): The name of existing model. Defaults to "v1-base".
            audio_cfg (_type_, optional): The audio configuration class. Defaults to None.
            text_cfg (_type_, optional): The text configuration class. Defaults to None.
            default_ckpt_path (str, optional): The default folder to save the openflam model ckpt. Defaults to /tmp/openflam.
        """
        super(OpenFLAM, self).__init__()
        self.default_ckpt_path = default_ckpt_path
        if model_name == "v1-base":
            text_cfg = CLAPTextCfg(connector_layers=12)
            audio_cfg = CLAPAudioCfg(
                model_name="base",
                audio_embed_dim=1024,
            )
            self.model = FLAM(
                audio_cfg=audio_cfg,
                text_cfg=text_cfg,
                joint_embed_shape=512,
                fabric=L.Fabric(),
                cache_dir=default_ckpt_path,
            )
            ckpt_path = hf_hub_download(
                repo_id="kechenadobe/OpenFLAM",
                filename="open_flam_oct17.pth",
                cache_dir=self.default_ckpt_path,
            )
            self.load_ckpt(ckpt_path)

        else:
            self.model = FLAM(
                audio_cfg=audio_cfg,
                text_cfg=text_cfg,
                fabric=L.Fabric(),
                cache_dir=default_ckpt_path**kwargs,
            )

    def load_ckpt(self, ckpt_path: str):
        """Load the pretrained checkpoint of OpenFLAM model

        Args:
            ckpt_path (str): The checkpoint path
        """

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt)

    @torch.inference_mode
    def get_global_audio_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get global audio embeddings

        Args:
            x (torch.Tensor): mono audio sample input [batch, 480000], as 10 seconds under 48 kHz.

        Returns:
            audio_global_feature (torch.Tensor): Global audio embeddings [batch, dim].
        """
        self.model.eval()
        audio_global_feature = self.model.get_audio_features(x)
        return audio_global_feature

    @torch.inference_mode
    def get_local_audio_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get local audio embeddings

        Args:
            x (torch.Tensor): mono audio sample input [batch, 480000], as 10 seconds under 48 kHz.

        Returns:
            audio_local_feature (torch.Tensor): Local audio embeddings [batch, frame, dim].
        """
        self.model.eval()
        audio_local_feature = self.model.get_local_audio_features(x)
        return audio_local_feature

    @torch.inference_mode
    def get_text_features(self, x: list[str]) -> torch.Tensor:
        """Get text embeddings

        Args:
            x (list): a list of texts [batch,]

        Returns:
            text_feature (torch.Tensor): text embeddings [batch, dim].
        """
        self.model.eval()
        text_feature = self.model.get_text_features(x)
        return text_feature

    @torch.inference_mode
    def get_logit_scale(self) -> torch.Tensor:
        """Get the global logit scale.
            Global logit scale is fixed as 1 in FLAM; include here for compatibility.

        Returns:
            logit_scale (torch.Tensor): logit scale [num_text].
        """
        self.model.eval()
        logit_scale = self.model.aed_logit_scale.exp().cpu()
        return logit_scale

    def get_per_text_logit_bias(
        self, text_branch_output: torch.Tensor
    ) -> torch.Tensor:
        """Get per-text logit bias ($\beta(t)$).

        Args:
            text_branch_output (torch.Tensor): text branch output [batch, dim].

        Returns:
            per_text_logit_bias (torch.Tensor): per-text logit bias [num_text].
        """
        self.model.eval()
        per_text_logit_bias = (
            self.model.fc_per_text_logit_bias(text_branch_output)
            .squeeze()
            .cpu()
        )
        return per_text_logit_bias

    def get_per_text_logit_scale(
        self, text_branch_output: torch.Tensor
    ) -> torch.Tensor:
        """Get per-text logit scale ($\alpha(t)$).

        Args:
            text_branch_output (torch.Tensor): text branch output [batch, dim].

        Returns:
            per_text_logit_scale (torch.Tensor): per-text logit scale [num_text].
        """
        self.model.eval()
        per_text_logit_scale = (
            self.model.fc_per_text_logit_scale(text_branch_output)
            .exp()
            .squeeze()
            .cpu()
        )
        return per_text_logit_scale

    def get_text_branch_output(self, text: list[str]) -> torch.Tensor:
        """Get text branch encoder output

        Returns:
            text_branch_output (torch.Tensor): text branch output [batch, dim].
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        text_branch_output = self.model.encode_text(text, device=device)
        return text_branch_output

    @torch.inference_mode
    def get_local_similarity(
        self,
        audio: torch.Tensor,
        text: list[str],
        method: str = "unbiased",
        cross_product: bool = False,
    ) -> torch.Tensor:
        """Get local similarity between audio and text

        Args:
            audio (torch.Tensor): audio sample input [batch, 480000], as 10 seconds under 48 kHz.
            text (list[str]): a list of texts [batch,]
            method (str): the method to compute the local similarity, "unbiased" or "approximate".
            cross_product (bool): Whether to compute the cross product of audio and text embeddings,
                i.e, if True, the output will be the similarity of each audio detected by all text,
                otherwise, the output will be the similarity of each audio detected by corresponding text of the same index.

        Returns:
            local_similarity (torch.Tensor): local similarity between audio and text [batch, dim].
        """
        self.model.eval()

        # Get local audio features and text features
        local_audio_features = self.model.get_local_audio_features(audio)
        text_features = self.model.get_text_features(text)

        # Get logit scale and bias from the model
        logit_scale = self.model.aed_logit_scale.exp().cpu()
        text_branch_output = self.get_text_branch_output(text)
        per_text_logit_scale = self.get_per_text_logit_scale(text_branch_output)
        logit_scale = per_text_logit_scale * logit_scale

        # Compute local similarity based on the method
        if method == "unbiased":
            per_text_logit_bias = self.get_per_text_logit_bias(
                text_branch_output
            )
            local_similarity = get_unbiased_framewise_act_map(
                local_audio_features.cpu(),
                text_features.cpu(),
                logit_scale=logit_scale,
                logit_bias=per_text_logit_bias,
                cross_product=cross_product,
            )
        elif method == "approximate":
            local_similarity = get_approximate_framewise_act_map(
                local_audio_features.cpu(),
                text_features.cpu(),
                logit_scale=logit_scale,
                cross_product=cross_product,
            )
        else:
            raise ValueError(
                f"Invalid method: {method}. Must be 'unbiased' or 'approximate'."
            )
        return local_similarity

    @torch.inference_mode
    def sanity_check(self):
        data_path = hf_hub_download(
            repo_id="kechenadobe/OpenFLAM",
            filename="sanity_data.npy",
            cache_dir=self.default_ckpt_path,
        )
        gt_path = hf_hub_download(
            repo_id="kechenadobe/OpenFLAM",
            filename="sanity_gt.npy",
            cache_dir=self.default_ckpt_path,
        )

        data = np.load(data_path, allow_pickle=True)
        gt = np.load(gt_path, allow_pickle=True)

        metrics = {
            "global_audio_feature": [],
            "local_audio_feature": [],
            "text_feature": [],
        }

        for d, g in zip(data, gt):
            audio = d["audio"]
            text = [d["text"]]
            g_gloabl_audio_feature = g["global_audio"]
            g_local_audio_feature = g["local_audio"]
            g_text_feature = g["text"]
            global_audio_feature = (
                self.get_global_audio_features(torch.from_numpy(audio))
                .cpu()
                .numpy()
            )
            local_audio_feature = (
                self.get_local_audio_features(torch.from_numpy(audio))
                .cpu()
                .numpy()
            )
            text_feature = self.get_text_features(text).cpu().numpy()

            ga_similarity = np.diag(
                global_audio_feature @ g_gloabl_audio_feature.T
            ).mean()
            t_similarity = np.diag(text_feature @ g_text_feature.T).mean()

            la_similarity = np.array(0.0)
            for k in range(local_audio_feature.shape[1]):
                c_emb = local_audio_feature[:, k, :]
                p_emb = g_local_audio_feature[:, k, :]
                la_similarity += np.diag(p_emb @ c_emb.T).mean()

            la_similarity = la_similarity / local_audio_feature.shape[1]

            metrics["global_audio_feature"].append(ga_similarity)
            metrics["local_audio_feature"].append(la_similarity)
            metrics["text_feature"].append(t_similarity)

        print("Sanity Check Results:")
        print("----------------------------")
        for key in metrics:
            print(key + "_similarity:", np.array(metrics[key]).mean())
        print("----------------------------")
