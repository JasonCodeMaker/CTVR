import torch
import os

def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta


class WISE_FT:
    def __init__(self, args, prev_task, curr_task) -> None:
        theta_0 = {k: v.clone() for k, v in prev_task.items()}
        theta_1 = {k: v.clone() for k, v in curr_task.items()}
        assert set(theta_0.keys()) == set(theta_1.keys())

        fishers = None
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.fishers = fishers
        self.fisher_floor = None

    def __call__(self, alpha):
        theta = _merge(
            alpha, self.theta_0, self.theta_1, self.fishers, self.fisher_floor
        )
        return theta


def evaluate_wise_ft(args, prev_task, curr_task):
    wise_ft = WISE_FT(args, prev_task, curr_task)

    theta = wise_ft(args.wise_alpha)
    
    return theta