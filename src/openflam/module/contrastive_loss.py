"""
Implementation of FLAM frame-wise loss.
We adopt the SigLIP implementation from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
--------------------------------------------------------
Paper: https://arxiv.org/abs/2505.05335
Code Maintainers: Ke Chen, Yusong Wu, Oriol Nieto, Prem Seetharaman
Support: Adobe Research
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import lightning as L
from torch import distributed as dist


class CLIPLoss(nn.Module):
    """InfoNCE loss used in CLIP."""

    def __init__(self, fabric: L.Fabric):
        self.fabric = fabric
        super().__init__()

    def gather_features(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather features from all GPUs and pods.
        """
        if self.fabric.world_size > 1:
            # [world_size, batch_size, feature_dim]
            audio_features = self.fabric.all_gather(
                audio_features, sync_grads=True
            )
            text_features = self.fabric.all_gather(
                text_features, sync_grads=True
            )
            world_size, batch_size, feature_dim = audio_features.shape
            audio_features = audio_features.reshape(
                world_size * batch_size, feature_dim
            )
            text_features = text_features.reshape(
                world_size * batch_size, feature_dim
            )
        return audio_features, text_features

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale_a: float,
    ) -> torch.Tensor:
        device = audio_features.device

        audio_features, text_features = self.gather_features(
            audio_features=audio_features, text_features=text_features
        )

        logits_per_audio = logit_scale_a * audio_features @ text_features.T
        logits_per_text = logits_per_audio.T

        # calculated ground-truth
        num_logits = logits_per_audio.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits_per_audio, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    """Exchange a tensor with a neighboring rank.

    Args:
        from_rank (int): Rank that owns the tensor to be sent.
        to_rank (int): Destination rank that will receive the tensor.
        tensor (torch.Tensor): Tensor payload to exchange with the neighbor.
        group (Optional[dist.ProcessGroup]): Communication group that defines participating ranks.

    Returns:
        torch.Tensor: Tensor received from the neighboring rank, matching the input shape.
    """
    tensor = tensor.contiguous()
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    """Exchange tensors with the left and right neighboring ranks.

    Args:
        left_rank (int): Rank of the left neighbor that receives `tensor_to_left`.
        right_rank (int): Rank of the right neighbor that receives `tensor_to_right`.
        tensor_to_left (torch.Tensor): Tensor sent to the left neighbor.
        tensor_to_right (torch.Tensor): Tensor sent to the right neighbor.
        group (Optional[dist.ProcessGroup]): Communication group that defines participating ranks.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors received from the right and left neighbors respectively.
    """
    tensor_to_left = tensor_to_left.contiguous()
    tensor_to_right = tensor_to_right.contiguous()
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_right, recv_op_left]
    )
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    """Autograd function for neighbor exchange with gradient support."""

    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        """Exchange a tensor with a neighboring rank.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context used to stash metadata for backward.
            from_rank (int): Rank identifying the current process in the process group.
            to_rank (int): Destination rank that receives the tensor during the forward pass.
            group (Optional[dist.ProcessGroup]): Communication group the exchange belongs to.
            tensor (torch.Tensor): Tensor data to send to the neighboring rank.

        Returns:
            torch.Tensor: Tensor received from the target rank, shaped like the original tensor.
        """
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backpropagate gradients through the neighbor exchange.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context containing cached rank metadata.
            grad_output (torch.Tensor): Gradient propagated from the subsequent operation.

        Returns:
            Tuple[None, None, None, torch.Tensor]: Gradient with respect to the tensor argument; the remaining entries are non-differentiable.
        """
        return (None, None, None) + (
            NeighbourExchange.apply(
                ctx.to_rank, ctx.from_rank, ctx.group, grad_output
            ),
        )


class NeighbourExchangeBidir(torch.autograd.Function):
    """Autograd function for bidirectional neighbor exchange with gradient support."""

    @staticmethod
    def forward(
        ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right
    ):
        """Perform bidirectional tensor exchange with neighboring ranks.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context used to store exchange metadata.
            left_rank (int): Rank of the left neighbor involved in the exchange.
            right_rank (int): Rank of the right neighbor involved in the exchange.
            group (Optional[dist.ProcessGroup]): Communication group coordinating the exchange.
            tensor_to_left (torch.Tensor): Tensor to send to the left neighbor.
            tensor_to_right (torch.Tensor): Tensor to send to the right neighbor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors received from the right and left neighbors respectively.
        """
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(
            left_rank, right_rank, tensor_to_left, tensor_to_right, group=group
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backpropagate gradients through the bidirectional exchange.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context containing neighbor metadata for the exchange.
            *grad_outputs (torch.Tensor): Gradients corresponding to each tensor returned during the forward pass.

        Returns:
            Tuple[None, None, None, torch.Tensor, torch.Tensor]: Gradients with respect to the tensors sent left and right; other entries remain non-differentiable.
        """
        return (None, None, None) + NeighbourExchangeBidir.apply(
            ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs
        )


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    """Perform neighbor exchange with autograd tracking.

    Args:
        from_rank (int): Rank that originates the tensor payload.
        to_rank (int): Rank that receives the tensor payload.
        tensor (torch.Tensor): Tensor to exchange while preserving gradients.
        group (Optional[dist.ProcessGroup]): Communication group that defines participating ranks.

    Returns:
        torch.Tensor: Tensor received from the peer rank with gradients preserved.
    """
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


def neighbour_exchange_bidir_with_grad(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    """Exchange tensors bidirectionally with autograd tracking.

    Args:
        left_rank (int): Rank of the left neighbor participating in the exchange.
        right_rank (int): Rank of the right neighbor participating in the exchange.
        tensor_to_left (torch.Tensor): Tensor to send left while preserving gradients.
        tensor_to_right (torch.Tensor): Tensor to send right while preserving gradients.
        group (Optional[dist.ProcessGroup]): Communication group that defines participating ranks.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Gradient-preserving tensors received from right and left neighbors respectively.
    """
    return NeighbourExchangeBidir.apply(
        left_rank, right_rank, group, tensor_to_left, tensor_to_right
    )


class FramewiseContrastiveLoss(nn.Module):
    """
    Sigmoid-based local contrastive loss for Sound Event Detection (SED).

    1. Computes frame-wise sigmoid loss between audio and text features
    2. Supports per-text logit bias and scale adjustments
    3. Handles distributed training with neighbor exchange
    4. Applies masking for incomplete audio or text batch

    This class corresponds to the Eq. 4 in the paper.
    """

    def __init__(
        self, fabric: L.Fabric, cache_labels: bool = False, bidir: bool = True
    ):
        super().__init__()
        self.fabric = fabric
        self.rank = fabric.global_rank
        self.world_size = fabric.world_size
        self.cache_labels = cache_labels
        self.bidir = bidir

    def get_labels_from_batch(
        self,
        framewise_label: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert frame-wise labels to {-1, 1} contrastive targets.

        Args:
            framewise_label (torch.Tensor): Frame wise labels shaped `[batch_size_audio, batch_size_text, T]`;
                `T` is the number of frame steps.
            device (torch.device): Device where the output tensor should reside.
            dtype (torch.dtype): Desired dtype for the output.

        Returns:
            torch.Tensor: Pairwise contrastive targets shaped `[batch_size_audio, batch_size_text, T]`,
        """
        # Binary frame-wise labels are precomputed for each audio-text pair.
        # We prepare them in the dataloader.
        return 2 * framewise_label - 1

    def get_ground_truth(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_audios: int,
        num_texts: int,
        num_frames: int,
        framewise_label: Optional[torch.Tensor] = None,
        negative_only: bool = False,
    ) -> torch.Tensor:
        """Generate frame-wise contrastive labels for the audio-text pairs.

        Args:
            device (torch.device): Target device for the generated targets.
            dtype (torch.dtype): Desired dtype for the returned tensor.
            num_audios (int): `batch_size_audio`, the number of audio samples in the local batch.
            num_texts (int): `batch_size_text`, the number of text prompts paired with the audios.
            num_frames (int): `T`, the number of frame steps per audio clip.
            framewise_label (Optional[torch.Tensor]): Frame wise labels shaped `[batch_size_audio, batch_size_text, T]`;
                `T` is the number of frame steps.
            negative_only (bool): If True, populate the tensor with {-1} contrastive labels only.

        Returns:
            torch.Tensor: Frame-wise contrastive targets shaped `[batch_size_audio, batch_size_text, T]`.
        """
        if negative_only:
            labels = -torch.ones(
                (num_audios, num_texts, num_frames), device=device, dtype=dtype
            )
        else:
            labels = self.get_labels_from_batch(framewise_label, device, dtype)
        return labels

    def get_logits(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute frame-wise logits for audio-text similarities.

        Args:
            audio_features (torch.Tensor): Audio embeddings shaped `[batch_size_audio, T, feature_dim]`;
                `T` is the number of frame steps.
            text_features (torch.Tensor): Text embeddings shaped `[batch_size_text, feature_dim]`.
            logit_scale (torch.Tensor): Temperature scale ($\alpha(t)$) applied to logits.
            logit_bias (Optional[torch.Tensor]): Logit bias ($\beta(t)$) broadcastable to
                `[batch_size_audio, batch_size_text, T]`.

        Returns:
            torch.Tensor: Frame-wise logits shaped `[batch_size_audio, batch_size_text, T]`.
        """
        # audio_features: [batch_size_a, T, feature_dim]
        # text_features: [batch_size_t, feature_dim]
        # logits: [batch_size_a, batch_size_t, T]
        logits = torch.einsum("btd,cd->bct", audio_features, text_features)
        logits = logit_scale * logits
        if logit_bias is not None:
            if logit_bias.dim() == 1:  # [batch_size]
                # Just for compatibility with previous experiments.
                logit_bias = logit_bias[:, None, None]
            elif logit_bias.dim() == 2:  # [batch_size_a, batch_size_t]
                logit_bias = logit_bias[:, :, None]
            logits += logit_bias
        return logits

    def _loss(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        framewise_label: Optional[torch.Tensor] = None,
        negative_only: bool = False,
    ) -> torch.Tensor:
        """Compute contrastive loss for each audio-text pair.

        Args:
            audio_features (torch.Tensor): Audio embeddings shaped `[batch_size_audio, T, feature_dim]`.
            text_features (torch.Tensor): Text embeddings shaped `[batch_size_text, feature_dim]`.
            logit_scale (torch.Tensor): Temperature scale ($\alpha(t)$) applied to logits shaped `[batch_size_text]`.
            logit_bias (Optional[torch.Tensor]): Logit bias ($\beta(t)$) applied to logits shaped `[batch_size_text]`.
            framewise_label (Optional[torch.Tensor]): Frame-wise contrastive targets shaped
                `[batch_size_audio, batch_size_text, T]` or `None` when negatives are synthesized.
            negative_only (bool): If True, assumes the batch contains only negative contrastive pairs.

        Returns:
            torch.Tensor: Per-pair frame-wise contrastive loss shaped `[batch_size_audio, batch_size_text, T]`.
        """
        logits = self.get_logits(
            audio_features, text_features, logit_scale, logit_bias
        )
        batch_size_audio = audio_features.shape[0]
        batch_size_text = text_features.shape[0]
        num_frames = audio_features.shape[1]
        labels = self.get_ground_truth(
            audio_features.device,
            audio_features.dtype,
            batch_size_audio,
            batch_size_text,
            num_frames,
            framewise_label=framewise_label,
            negative_only=negative_only,
        )
        per_item_loss = -F.logsigmoid(labels * logits)
        # per_item_loss: [batch_size_audio, batch_size_text, T]
        return per_item_loss

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: float,
        framewise_label: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
        logit_bias: Optional[torch.Tensor],
        per_text_logit_bias: Optional[torch.Tensor] = None,
        per_text_logit_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the AED full-event contrastive loss for a batch.

        Args:
            audio_features (torch.Tensor): Audio embeddings shaped `[batch_size_audio, T, feature_dim]`;
                `T` is the number of frame steps.
            text_features (torch.Tensor): Text embeddings shaped `[batch_size_text, feature_dim]`.
            logit_scale (float): Global temperature ($\alpha$) applied to the logits.
                During local training, we fix this value to 1.0.
            framewise_label (torch.Tensor): Frame-wise labels shaped `[batch_size_audio, batch_size_text, T]`.
            audio_mask (torch.Tensor): Mask shaped `[batch_size_audio]` indicating valid audio entries.
                In dataloader, if N*batch_size_audio texts are reached with not enough audio samples,
                the audio_mask will be set to False for the remaining audio samples.
                In the paper we use N=5.
            text_mask (torch.Tensor): Mask shaped `[batch_size_text]` indicating valid text entries.
                In dataloader, if batch_size_audio audio samples are reached with not enough text samples,
                the text_mask will be set to False for the remaining text samples.
            logit_bias (Optional[torch.Tensor]): Global logit bias ($\beta$) broadcastable to `[batch_size_audio, batch_size_text, T]`.
                During local training, we set this value to 0.
            per_text_logit_bias (Optional[torch.Tensor]): Per-text logit bias adjustments ($\beta(t)$) shaped `[batch_size_text]`.
            per_text_logit_scale (Optional[torch.Tensor]): Per-text scale adjustments ($\alpha(t)$) shaped `[batch_size_text]`.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the scalar loss and diagnostic statistics.
        """
        audio_features_aed = audio_features
        text_features_aed = text_features
        framewise_label_aed = framewise_label

        # Create data mask for valid audio-text pairs
        data_mask = (
            audio_mask[:, None] * text_mask[None]
        )  # [num_audio, num_text]

        # Initialize logit bias if not provided
        if logit_bias is None:
            n_gpu = self.fabric.world_size
            # Apply logit bias correction based on positive ratio
            framewise_label_masked = framewise_label_aed[data_mask]
            positive_num = framewise_label_masked.sum()
            # Calculate positive ratio across all GPUs
            positive_ratio = positive_num / (
                torch.prod(torch.tensor(framewise_label_masked.shape))
                * n_gpu
                * n_gpu
            )
            logit_bias = torch.log(positive_ratio)

        # Apply per-text adjustments
        if per_text_logit_bias is not None:
            logit_bias = logit_bias + per_text_logit_bias[None, :, None]
            has_per_text_logit_bias = True
        else:
            has_per_text_logit_bias = False

        if per_text_logit_scale is not None:
            logit_scale = logit_scale * per_text_logit_scale[None, :, None]
            has_per_text_logit_scale = True
        else:
            has_per_text_logit_scale = False

        # Compute per-item loss
        per_item_loss = self._loss(
            audio_features_aed,
            text_features_aed,
            logit_scale,
            logit_bias=logit_bias,
            framewise_label=framewise_label_aed,
        )

        # Apply data mask
        per_item_loss = data_mask[:, :, None] * per_item_loss
        # per_item_loss: [num_audio, num_text, T]

        # Compute final loss: average over valid pairs and frames
        loss = per_item_loss.sum() / data_mask.sum() / audio_features.shape[1]

        # Handle distributed training with neighbor exchange
        if self.world_size > 1:
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size

            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                text_mask_to_right = text_mask_to_left = text_mask

                if has_per_text_logit_bias:
                    logit_bias_to_right = logit_bias_to_left = logit_bias
                if has_per_text_logit_scale:
                    logit_scale_to_right = logit_scale_to_left = logit_scale

                num_bidir, remainder = divmod(self.world_size - 1, 2)

                # Bidirectional exchanges
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    text_mask_recv = neighbour_exchange_bidir(
                        left_rank,
                        right_rank,
                        text_mask_to_left,
                        text_mask_to_right,
                    )

                    if has_per_text_logit_bias:
                        per_text_logit_bias_recv = (
                            neighbour_exchange_bidir_with_grad(
                                left_rank,
                                right_rank,
                                logit_bias_to_left,
                                logit_bias_to_right,
                            )
                        )
                    if has_per_text_logit_scale:
                        per_text_logit_scale_recv = (
                            neighbour_exchange_bidir_with_grad(
                                left_rank,
                                right_rank,
                                logit_scale_to_left,
                                logit_scale_to_right,
                            )
                        )

                    for j in range(len(text_features_recv)):
                        f = text_features_recv[j]
                        m = text_mask_recv[j]

                        if has_per_text_logit_bias:
                            curr_logit_bias = per_text_logit_bias_recv[j]
                        else:
                            curr_logit_bias = logit_bias

                        if has_per_text_logit_scale:
                            curr_logit_scale = per_text_logit_scale_recv[j]
                        else:
                            curr_logit_scale = logit_scale

                        curr_per_item_loss = self._loss(
                            audio_features_aed,
                            f,
                            curr_logit_scale,
                            curr_logit_bias,
                            framewise_label=None,
                            negative_only=True,
                        )

                        # Avoid duplicate pairs by checking for identical text features
                        equal_text = torch.isclose(
                            text_features_aed, f, atol=1e-6
                        ).all(dim=-1)
                        curr_text_mask = m & ~equal_text  # [num_text]

                        curr_data_mask = (
                            audio_mask[:, None] * curr_text_mask[None]
                        )
                        curr_loss = (
                            curr_data_mask[:, :, None] * curr_per_item_loss
                        )
                        loss += (
                            curr_loss.sum()
                            / curr_data_mask.sum()
                            / audio_features.shape[-1]
                        )

                    text_features_to_left, text_features_to_right = (
                        text_features_recv
                    )

                # Handle remainder if world_size is odd
                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )
                    text_mask_recv = neighbour_exchange(
                        left_rank, right_rank, text_mask_to_right
                    )

                    if has_per_text_logit_bias:
                        curr_logit_bias = neighbour_exchange_with_grad(
                            left_rank, right_rank, logit_bias_to_right
                        )
                    else:
                        curr_logit_bias = logit_bias

                    if has_per_text_logit_scale:
                        curr_logit_scale = neighbour_exchange_with_grad(
                            left_rank, right_rank, logit_scale_to_right
                        )
                    else:
                        curr_logit_scale = logit_scale

                    curr_per_item_loss = self._loss(
                        audio_features,
                        text_features_recv,
                        curr_logit_scale,
                        curr_logit_bias,
                        framewise_label=None,
                        negative_only=True,
                    )

                    equal_text = torch.isclose(
                        text_features_aed, text_features_recv, atol=1e-6
                    ).all(dim=-1)
                    curr_text_mask = text_mask_recv & ~equal_text

                    curr_data_mask = audio_mask[:, None] * curr_text_mask[None]
                    curr_loss = curr_data_mask[:, :, None] * curr_per_item_loss
                    loss += (
                        curr_loss.sum()
                        / curr_data_mask.sum()
                        / audio_features.shape[-1]
                    )
            else:
                # Unidirectional exchange (fallback)
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )
                    text_mask_from_left = neighbour_exchange(
                        left_rank, right_rank, text_mask_to_right
                    )

                    if has_per_text_logit_bias:
                        curr_logit_bias = neighbour_exchange_with_grad(
                            left_rank, right_rank, logit_bias_to_right
                        )
                    else:
                        curr_logit_bias = logit_bias

                    if has_per_text_logit_scale:
                        curr_logit_scale = neighbour_exchange_with_grad(
                            left_rank, right_rank, logit_scale_to_right
                        )
                    else:
                        curr_logit_scale = logit_scale

                    curr_loss = self._loss(
                        audio_features,
                        text_features_from_left,
                        curr_logit_scale,
                        curr_logit_bias,
                        framewise_label=None,
                        negative_only=True,
                    )

                    equal_text = torch.isclose(
                        text_features_aed, text_features_from_left, atol=1e-6
                    ).all(dim=-1)
                    curr_text_mask = text_mask_from_left & ~equal_text

                    curr_data_mask = audio_mask[:, None] * curr_text_mask[None]
                    curr_loss = curr_data_mask[:, :, None] * curr_loss
                    loss += (
                        curr_loss.sum()
                        / curr_data_mask.sum()
                        / audio_features.shape[-1]
                    )
                    text_features_to_right = text_features_from_left

        metrics = {
            "loss": loss,
            "logit_bias": logit_bias.mean(),
        }
        return metrics


class PriorLoss(nn.Module):
    """Loss for training prior distribution used with per-text logit bias.

    This class corresponds to the Eq. 10 in the paper.
    """

    def __init__(self, fabric: L.Fabric):
        self.fabric = fabric
        self.n_gpu = fabric.world_size
        super().__init__()

    def gather_features(self, audio_mask: torch.Tensor) -> torch.Tensor:
        """Synchronize audio masks across devices.

        Args:
            audio_mask (torch.Tensor): Local audio mask tensor shaped `[batch_size_audio]`.

        Returns:
            torch.Tensor: Flattened audio mask of shape `[world_size * batch_size_audio]` containing entries from all devices.
        """
        if self.fabric.world_size > 1:
            audio_mask = self.fabric.all_gather(audio_mask, sync_grads=True)
            audio_mask = audio_mask.reshape(-1)
        return audio_mask

    def forward(
        self,
        prior_logits: torch.Tensor,
        framewise_label: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the prior loss used for per-text bias calibration.

        Args:
            prior_logits (torch.Tensor): Predicted logits for positive ratios shaped `[batch_size_text]`.
                This is the logit bias ($\beta(t)$) for each text prompt.
            framewise_label (torch.Tensor): Binary frame-wise labels shaped `[batch_size_audio, batch_size_text, T]`;
                `T` is the number of frame steps.
            audio_mask (torch.Tensor): Mask identifying valid audio samples shaped `[batch_size_audio]`.
            text_mask (torch.Tensor): Mask identifying valid text samples shaped `[batch_size_text]`.

        Returns:
            torch.Tensor: Scalar tensor containing the prior loss value.
        """
        # framewise_label: [num_audio, num_text, T]
        positive_count = framewise_label.sum(-1)  # [num_audio, num_text]
        positive_count = positive_count.sum(0)  # [num_text]

        audio_mask_all = self.gather_features(audio_mask)
        # denominator: n_gpu * num_audio * T with mask
        denominator = audio_mask_all.sum() * framewise_label.shape[-1]
        positive_ratio = positive_count / denominator

        loss = F.binary_cross_entropy_with_logits(
            prior_logits, positive_ratio, reduction="none"
        )
        loss = loss * text_mask / text_mask.sum()
        return loss.sum()


# Example usage and helper functions
def create_framewise_contrastive_loss(
    fabric: L.Fabric,
) -> FramewiseContrastiveLoss:
    """Instantiate the framewise contrastive loss module with bidirectional exchange enabled.

    Args:
        fabric (L.Fabric): Fabric instance that manages the distributed environment.

    Returns:
        FramewiseContrastiveLoss: Configured loss module ready for training.
    """
    return FramewiseContrastiveLoss(fabric=fabric, bidir=True)


def create_prior_loss(fabric: L.Fabric) -> PriorLoss:
    """Instantiate the prior loss module for per-text bias learning.

    Args:
        fabric (L.Fabric): Fabric instance that manages the distributed environment.

    Returns:
        PriorLoss: Configured prior loss module ready for training.
    """
    return PriorLoss(fabric=fabric)
