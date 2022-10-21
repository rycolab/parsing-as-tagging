# Code is inspired from https://github.com/mtreviso/linear-chain-crf
import torch
from torch import nn


class CRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF).
    Args:
        nb_labels (int): number of labels in your tagset, including special symbols.
        device: cpu or gpu,
        batch_first (bool): Whether the first dimension represents the batch dimension.
    """

    def __init__(
            self, nb_labels, device=None, batch_first=True
    ):
        super(CRF, self).__init__()

        self.nb_labels = nb_labels
        self.batch_first = batch_first
        self.device = device

        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
        self.start_transitions = nn.Parameter(torch.empty(self.nb_labels))
        self.end_transitions = nn.Parameter(torch.empty(self.nb_labels))

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask=None):
        """Compute the negative log-likelihood. See `log_likelihood` method."""
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self, emissions, tags, mask=None):
        """Compute the probability of a sequence of tags given a sequence of
        emissions scores.
        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape of (batch_size, seq_len, nb_labels) if batch_first is True,
                (seq_len, batch_size, nb_labels) otherwise.
            tags (torch.LongTensor): Sequence of labels.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
        Returns:
            torch.Tensor: the log-likelihoods for each sequence in the batch.
                Shape of (batch_size,)
        """

        # fix tensors order by setting batch as the first dimension
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float).to(self.device)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return torch.sum(scores - partition)

    def _compute_scores(self, emissions, tags, mask, last_idx=None):
        """Compute the scores for a given batch of emissions with their tags.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            tags (Torch.LongTensor): (batch_size, seq_len)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size).to(self.device)

        alpha_mask = torch.zeros((batch_size,), dtype=int).to(self.device)
        previous_tags = torch.zeros((batch_size,), dtype=int).to(self.device)

        for i in range(0, seq_length):
            is_valid = mask[:, i]

            current_tags = tags[:, i]

            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()

            first_t_scores = self.start_transitions[current_tags]
            t_scores = self.transitions[previous_tags, current_tags]
            t_scores = (1 - alpha_mask) * first_t_scores + alpha_mask * t_scores
            alpha_mask = is_valid + (1 - is_valid) * alpha_mask

            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

            previous_tags = current_tags * is_valid + previous_tags * (1 - is_valid)

        scores += self.end_transitions[previous_tags]

        return scores

    def _compute_log_partition(self, emissions, mask):
        """Compute the partition function in log-space using the forward-algorithm.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length, nb_labels = emissions.shape

        alphas = torch.zeros((batch_size, nb_labels)).to(self.device)
        alpha_mask = torch.zeros((batch_size, 1), dtype=int).to(self.device)

        for i in range(0, seq_length):
            is_valid = mask[:, i].unsqueeze(-1)

            first_alphas = self.start_transitions + emissions[:, i]

            # (bs, nb_labels) -> (bs, 1, nb_labels)
            e_scores = emissions[:, i].unsqueeze(1)
            # (nb_labels, nb_labels) -> (bs, nb_labels, nb_labels)
            t_scores = self.transitions.unsqueeze(0)
            # (bs, nb_labels)  -> (bs, nb_labels, 1)
            a_scores = alphas.unsqueeze(2)
            scores = e_scores + t_scores + a_scores

            new_alphas = torch.logsumexp(scores, dim=1)

            new_alphas = (1 - alpha_mask) * first_alphas + alpha_mask * new_alphas
            alpha_mask = is_valid + (1 - is_valid) * alpha_mask

            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

        end_scores = alphas + self.end_transitions

        return torch.logsumexp(end_scores, dim=1)
