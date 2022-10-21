## PUBLIC API: code from: https://github.com/nikitakit/tetra-tagging

import numpy as np


class Beam:
    def __init__(self, scores, stack_depths, prev, backptrs, labels):
        self.scores = scores
        self.stack_depths = stack_depths
        self.prev = prev
        self.backptrs = backptrs
        self.labels = labels


class BeamSearch:
    def __init__(
            self,
            tag_moderator,
            initial_stack_depth,
            max_depth=5,
            min_depth=1,
            keep_per_depth=1,
            crf_transitions=None,
            initial_label=None,
    ):
        # Save parameters
        self.tag_moderator = tag_moderator
        self.valid_depths = np.arange(min_depth, max_depth)
        self.keep_per_depth = keep_per_depth
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.crf_transitions = crf_transitions

        # Initialize the beam
        scores = np.zeros(1, dtype=float)
        stack_depths = np.full(1, initial_stack_depth)
        prev = backptrs = labels = None
        if initial_label is not None:
            labels = np.full(1, initial_label)
        self.beam = Beam(scores, stack_depths, prev, backptrs, labels)

    def compute_new_scores(self, label_log_probs, is_last):
        if self.crf_transitions is None:
            return self.beam.scores[:, None] + label_log_probs
        else:
            if self.beam.labels is not None:
                all_new_scores = self.beam.scores[:, None] + label_log_probs + \
                                 self.crf_transitions["transitions"][self.beam.labels]
            else:
                all_new_scores = self.beam.scores[:, None] + label_log_probs + \
                                 self.crf_transitions["start_transitions"]
            if is_last:
                all_new_scores += self.crf_transitions["end_transitions"]
            return all_new_scores

    # This extra mask layer takes care of invalid reduce actions when there is not an empty
    # slot in the tree, which is needed in the top-down shift reduce tagging schema
    def extra_mask_layer(self, all_new_scores, all_new_stack_depths):
        depth_mask = np.zeros(all_new_stack_depths.shape)
        depth_mask[all_new_stack_depths < 0] = -np.inf
        depth_mask[all_new_stack_depths > self.max_depth] = -np.inf
        all_new_scores = all_new_scores + depth_mask

        all_new_stack_depths = (
                all_new_stack_depths
                + self.tag_moderator.stack_depth_change_by_id
        )
        return all_new_scores, all_new_stack_depths

    def advance(self, label_logits, is_last=False):
        label_log_probs = label_logits

        all_new_scores = self.compute_new_scores(label_log_probs, is_last)
        if self.tag_moderator.mask_binarize and self.beam.labels is not None:
            labels = self.beam.labels
            all_new_scores = self.tag_moderator.mask_scores_for_binarization(labels,
                                                                             all_new_scores)

        if self.tag_moderator.stack_depth_change_by_id_l2 is not None:
            all_new_stack_depths = (
                    self.beam.stack_depths[:, None]
                    + self.tag_moderator.stack_depth_change_by_id_l2[None, :]
            )
            all_new_scores, all_new_stack_depths = self.extra_mask_layer(all_new_scores,
                                                                         all_new_stack_depths)
        else:
            all_new_stack_depths = (
                    self.beam.stack_depths[:, None]
                    + self.tag_moderator.stack_depth_change_by_id[None, :]
            )

        masked_scores = all_new_scores[None, :, :] + np.where(
            all_new_stack_depths[None, :, :]
            == self.valid_depths[:, None, None],
            0.0,
            -np.inf,
        )
        masked_scores = masked_scores.reshape(self.valid_depths.shape[0], -1)
        idxs = np.argsort(-masked_scores)[:, : self.keep_per_depth].flatten()
        backptrs, labels = np.unravel_index(idxs, all_new_scores.shape)

        transition_valid = all_new_stack_depths[
                               backptrs, labels
                           ] == self.valid_depths.repeat(self.keep_per_depth)

        backptrs = backptrs[transition_valid]
        labels = labels[transition_valid]

        self.beam = Beam(
            all_new_scores[backptrs, labels],
            all_new_stack_depths[backptrs, labels],
            self.beam,
            backptrs,
            labels,
        )

    def get_path(self, idx=0, required_stack_depth=1):
        if required_stack_depth is not None:
            assert self.beam.stack_depths[idx] == required_stack_depth
        score = self.beam.scores[idx]
        assert score > -np.inf

        beam = self.beam
        label_idxs = []
        while beam.prev is not None:
            label_idxs.insert(0, beam.labels[idx])
            idx = beam.backptrs[idx]
            beam = beam.prev

        return score, label_idxs


class GreedySearch(BeamSearch):
    def advance(self, label_logits, is_last=False):
        label_log_probs = label_logits

        all_new_scores = self.compute_new_scores(label_log_probs, is_last)
        if self.tag_moderator.mask_binarize and self.beam.labels is not None:
            labels = self.beam.labels
            all_new_scores = self.tag_moderator.mask_scores_for_binarization(labels,
                                                                             all_new_scores)

        if self.tag_moderator.stack_depth_change_by_id_l2 is not None:
            all_new_stack_depths = (
                    self.beam.stack_depths[:, None]
                    + self.tag_moderator.stack_depth_change_by_id_l2[None, :]
            )

            all_new_scores, all_new_stack_depths = self.extra_mask_layer(all_new_scores,
                                                                         all_new_stack_depths)
        else:
            all_new_stack_depths = (
                    self.beam.stack_depths[:, None]
                    + self.tag_moderator.stack_depth_change_by_id[None, :]
            )

        masked_scores = all_new_scores + np.where((all_new_stack_depths >= self.min_depth)
                                                  & (all_new_stack_depths <= self.max_depth),
                                                  0.0,
                                                  -np.inf)

        masked_scores = masked_scores.reshape(-1)
        idxs = np.argsort(-masked_scores)[:self.keep_per_depth].flatten()

        backptrs, labels = np.unravel_index(idxs, all_new_scores.shape)

        transition_valid = (all_new_stack_depths[
                                backptrs, labels
                            ] >= self.min_depth) & (all_new_stack_depths[
                                                        backptrs, labels
                                                    ] <= self.max_depth)

        backptrs = backptrs[transition_valid]
        labels = labels[transition_valid]


        self.beam = Beam(
            all_new_scores[backptrs, labels],
            all_new_stack_depths[backptrs, labels],
            self.beam,
            backptrs,
            labels,
        )

