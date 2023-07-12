import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional, Tuple, Any, Dict, Iterable, Union
import math


class TakeLSTMOutput(nn.Module):
    # Take the last hidden state from the output of the LSTM
    def forward(self, x):
        tensor, _ = x
        return tensor


def hexa_loss(logits, labels, attention_mask, num_even_tags, num_odd_tags):
    # shape: (batch_size, seq_len, num_tags) -> (batch_size, num_tags, seq_len)
    logits = torch.movedim(logits, -1, 1)
    odd_logits, even_logits = torch.split(logits, [num_odd_tags, num_even_tags], dim=1)

    odd_labels = (labels // (num_even_tags + 1)) - 1
    even_labels = (labels % (num_even_tags + 1)) - 1
    # The last word will have only even label

    # Only keep active parts of the loss
    active_even_labels = torch.where(attention_mask, even_labels, -1)
    active_odd_labels = torch.where(attention_mask, odd_labels, -1)
    loss = (F.cross_entropy(even_logits, active_even_labels, ignore_index=-1)
            + F.cross_entropy(odd_logits, active_odd_labels, ignore_index=-1))

    ground_truth_likelihood = 0.

    return loss, ground_truth_likelihood


def calc_loss_helper(logits, labels, attention_mask):
    # Only keep active parts of the loss

    # logits: shape: (batch_size, *, num_tags)
    # active_logits: shape: (batch_size, num_tags, *)
    active_logits = torch.movedim(logits, -1, 1)
    if active_logits.dim() == 4:  # for permute logits
        attention_mask = attention_mask.unsqueeze(2)

    # shape: (batch_size, seq_len, ...)
    active_labels = torch.where(
        (attention_mask == 1), labels, -1
    )
    loss = F.cross_entropy(active_logits, active_labels, ignore_index=-1)
    ground_truth_likelihood = 0.

    return loss, ground_truth_likelihood


def interval_sort(
        values,  # shape: (batch_size, seq_len)
        gt_values,  # shape: (batch_size, seq_len)
        mask,  # shape: (batch_size, seq_len)
        intervals  # shape: (batch_size, num_intervals, 4)
):
    # Args:
    #   value: (batch_size, seq_len)
    #   intervals: (batch_size, num_intervals, 4)
    #       tuples of (start, end, split, self) intervals. end is inclusive.
    # Returns:
    #   interval_stats: (batch_size, num_intervals, 3)
    #       tuples of (min, median, max) values in the interval

    # shape: (batch_size, seq_len)
    batch_size, seq_len = values.size()
    num_intervals = intervals.size(1)

    _, sorted_indices = torch.where(
        mask, gt_values, float('inf')  # ascending
    ).sort(dim=1)
    sorted_values = values.gather(dim=1, index=sorted_indices)

    range_vec = torch.arange(  # shape: (1, seq_len, 1)
        0, values.size(1), device=intervals.device, dtype=torch.long
    )[None, :, None]
    # shape: (batch_size, num_intervals)
    relative_indices = (intervals.select(dim=-1, index=3) -
                        intervals.select(dim=-1, index=0))
    # shape: (batch_size, seq_len, num_intervals)
    in_interval = ((range_vec >= intervals[:, None, :, 0]) &
                   (range_vec <= intervals[:, None, :, 1]))
    # shape: (batch_size, seq_len, num_intervals)
    projected_indices = torch.where(
        in_interval, sorted_indices[:, :, None], seq_len - 1
    ).sort(dim=1)[0]

    # shape (batch_size, num_intervals)
    projected_indices = projected_indices.take_along_dim(
        relative_indices[:, None, :], dim=1
    ).squeeze(1)

    # shape: (batch_size, num_intervals)
    # projected_values = values.take_along_dim(projected_indices, dim=1)
    projected_values = sorted_values.take_along_dim(projected_indices, dim=1)

    # shape: (batch_size, num_intervals)
    split_rank = intervals.select(dim=-1, index=2).long()

    # shape: (batch_size, num_intervals), avoid out of range
    safe_split_rank = torch.where(
        split_rank == 0, 1, split_rank
    )
    # shape: (batch_size, num_intervals)
    split_thresholds = (sorted_values.gather(dim=1, index=safe_split_rank - 1) +
                        sorted_values.gather(dim=1, index=safe_split_rank)) / 2.0
    # shape: (batch_size, num_intervals)
    # > 0: go right, < 0: go left
    split_logits = projected_values - split_thresholds

    return split_logits


def logsoftperm(
        input: torch.Tensor,  # shape: (*, num_elements)
        perm: torch.Tensor,  # shape: (*, num_elements)
        mask: Optional[torch.BoolTensor] = None  # shape: (*, num_elements)
):
    # Args:
    #   input: (*, num_elements)
    #   perm: (*, num_elements)
    # Returns:
    #   output: (*, num_elements)
    max_value = input.max().detach() + 1.0
    if mask is not None:
        input = input.masked_fill(~mask, max_value)
        perm = perm.masked_fill(~mask, max_value)

    # shape: (*, num_elements)
    sorted_input, _ = input.sort(dim=-1)
    # shape: (*, num_elements, num_elements)
    logits_matrix = -torch.abs(perm[:, None, :] - sorted_input[:, :, None])
    # shape: (*, num_elements)
    log_likelihood_perm = F.log_softmax(logits_matrix, dim=-1).diagonal(dim1=-2, dim2=-1)

    if mask is not None:
        log_likelihood_perm = log_likelihood_perm.masked_fill(~mask, 0.0)

    return log_likelihood_perm


def _batched_index_select(
        target: torch.Tensor,
        indices: torch.LongTensor,
        flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    # Args:
    #   target: (..., seq_len, ...)
    #   indices: (..., num_indices)
    # Returns:
    #   selected_targets (..., num_indices, ...)

    # dim is the index of the last dimension of indices
    dim = indices.dim() - 1
    unidim = False
    if target.dim() == 2:
        # (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
        unidim = True
        target = target.unsqueeze(-1)

    target_size, indices_size = target.size(), indices.size()
    # flatten dimensions before dim, make a pseudo batch dimension
    indices = indices.view(math.prod([*indices_size[:dim]]), indices_size[dim])

    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(
            indices, target_size[dim]
        )

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.reshape(-1, *target_size[dim + 1:])
    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices_size) + ([] if unidim else list(target_size[dim + 1:]))
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.reshape(*selected_shape)
    return selected_targets


def flatten_and_batch_shift_indices(
        indices: torch.Tensor, sequence_length: int
) -> torch.Tensor:
    # Input:
    #   indices: (batch_size, num_indices)
    #   sequence_length: int, d_1*d_2*...*d_n
    # Returns:
    #   offset_indices Shape: (batch_size*d_1*d_2*...*d_n)
    offsets = torch.arange(0, indices.size(0), dtype=torch.long,
                           device=indices.device) * sequence_length

    for _ in range(indices.dim() - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets
    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)

    return offset_indices


def onehot_with_ignore_label(labels, num_class, ignore_label):
    # One-hot encode the modified labels
    one_hot_labels = torch.nn.functional.one_hot(
        labels.masked_fill((labels == ignore_label), num_class),
        num_classes=num_class + 1
    )
    # Remove the last row in the one-hot encoding
    # shape: (*, num_class+1 -> num_class)
    one_hot_labels = one_hot_labels[..., :-1]
    return one_hot_labels
