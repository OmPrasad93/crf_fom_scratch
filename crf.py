import torch
from torch import nn


class CRF(nn.Module):
    """
        Linear-chain Conditional Random Field (CRF).
        Args:
            nb_labels (int): number of labels in your tagset, including special symbols.
            bos_tag_id (int): integer representing the beginning of sentence symbol in
                your tagset.
            eos_tag_id (int): integer representing the end of sentence symbol in your tagset.
            batch_first (bool): Whether the first dimension represents the batch dimension.
        """

    def __init__(self, nb_labels, bos_tag_id, eos_tag_id, batch_first=True):
        super().__init__()
        self.nb_labels = nb_labels
        self.BOS_TAG_ID = bos_tag_id
        self.EOS_TAG_ID = eos_tag_id
        self.batch_first = batch_first
        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))

    def init_weights(self):
        # initialize transitions from a random uniform distribution between -0.1 and 0.1
        nn.init.uniform(self.transitions, -0.1, 0.1)
        # enforce contraints (rows=from, columns=to) with a big negative number
        # so exp(-10000) will tend to zero
        # no transitions allowed to the beginning of sentence
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
        # no transitions allowed from the end of sentence
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

    def forward(self, emissions, tags, mask=None):
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

        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2])

        scores = self._compute_scores(emissions=emissions, tags=tags, mask=mask)
        partition = self._compute_log_partition(emissions=emissions, mask=mask)

    def _compute_scores(self, emissions, tags, mask):
        """Computes the scores of a given batch of emissions with their tags
            Args:
                emissions (torch.Tensor): (batch_size,seq_len,nb_labels)
                tags (Torch.LongTensor): (batch_size,seq_len)
                mask (Torch.FloatTensor): (batch_size,seq_len)
            Returns:
                torch.Tensor: Scores for each batch
                Shape: (batch_size,)
        """
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size)

        # save first and last tags to be used later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(
            1) - 1  # This step is to just get an index format for last_tags as mask and tags have same shape
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(
            1)).squeeze()  # Gets all the last tags across column(Dim = 1) with index from last_valid_idx
        t_scores = self.transitions[
            self.BOS_TAG_ID, first_tags]  # Add the transition from BOS to the first tags for each batch,self.transitions is a nn.Parameter.

        # add the [unary] emission scores for the first tags for each batch
        # for all batches, the first word, see the correspondent emissions
        # for the first tags (which is a list of ids):
        # emissions[:, 0, [tag_1, tag_2, ..., tag_nblabels]]
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()

        # the scores for a word is just the sum of both scores
        scores += e_scores + t_scores

        # For all remaining words

        for i in range(1, seq_length):
            is_valid = mask[:, i]

            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]

            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(
                1)).squeeze()  # Same as for the first word, but now we are doing it for the current words
            t_scores = self.transitions[previous_tags, current_tags]

            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

        # Add transition from last tags to EOS tag
        scores += self.transitions[last_tags, self.EOS_TAG_ID]
        return scores

    def _compute_log_partition(self, emissions, mask):
        """
        Compute the partition function in log-space using the forward-algorithm/forward-backward algorithm.
        Args:
            emissions: torch.Tensor: (batch_size,seq_len,nb_labels)
            mask: torch.FloatTensor: (batch_size,seq_len)

        Returns:
            torch.Tensor: the partition scores for each batch
            Shape: (batch_size,)
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_length):
            # (bs, nb_labels) -> (bs, 1, nb_labels)
            e_scores = emissions[:, i].unsqueeze(1)

            # (nb_labels,nb_labels) -> (bs, nb_labels, nb_labels)
            t_scores = self.transitions.unsqueeze(0)

            # (bs, nb_labels) -> (bs_nb_labels,1)
            a_scores = alphas.unsqueeze(2)

            scores = e_scores + t_scores + a_scores
            new_alphas = torch.logsumexp(scores, dim=1)

            # set alphas if the mask is valid, else keep current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas
        # add scores for final transition
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # return a log of sum of exps
        return torch.logsumexp(end_scores, dim=1)
