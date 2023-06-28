import torch

class Boundary_skip_Sampling:
    def __init__(self, unlabeled_idxs, dist, lower_bound, R, up_bound):

        self.unlabeled_idxs = unlabeled_idxs
        self.dist = dist
        self.lower_bound = lower_bound
        self.R = R
        self.up_bound = up_bound

    def query(self, m):
        unlabeled_dist = self.dist[self.unlabeled_idxs]
        condidate_index = torch.where((unlabeled_dist > self.lower_bound) & (unlabeled_dist < self.up_bound))
        if len(condidate_index[0]) > m:
            condidate_margin = torch.abs(unlabeled_dist[condidate_index] - self.R)
            anchor = condidate_margin.sort()[1][0]
            t_diversity = torch.abs(condidate_index[0][anchor] - condidate_index[0]) / torch.sum(torch.abs(condidate_index[0][anchor] - condidate_index[0]))
            uncertainty = condidate_margin / t_diversity
            final_select = condidate_index[0][uncertainty.sort()[1][-m:]]
        else:
            final_select = condidate_index[0]
        return self.unlabeled_idxs[final_select.cpu()]