import torch

def max_pointwise_l2_distance(trajectory, trajectories_set):
    max_distances = torch.max(torch.sqrt(torch.sum(torch.pow(trajectory-trajectories_set, 2), dim=2, keepdim=True)), dim=1).values
    return torch.min(max_distances)

def reduce_num_trajectories(trajectories, seed_tr, epsilon=3):
    reduced = seed_tr
    for tr in trajectories:
        tr = tr.unsqueeze(0)
        score = max_pointwise_l2_distance(tr, reduced)
        if score > epsilon:
            reduced = torch.cat([reduced, tr])

    return reduced

def get_epsilon_coverage(in_trajectories, out_trajectories):
    epsilon = 0

    for idx, tr in enumerate(out_trajectories):
        max_distances = torch.max(torch.sqrt(torch.sum(torch.pow(tr-in_trajectories, 2), dim=2, keepdim=True)), dim=1).values
        score = torch.min(max_distances)
        if score > epsilon:
            epsilon = score

    return epsilon