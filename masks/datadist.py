import torch
import numpy as np
import matplotlib.pyplot as plt

LABELS_FULL_PATH = './inspection/labels_3.0/labels_full_fut_10_fix_3.0.pt'
LABELS_VAL_PATH = './inspection/labels_3.0/labels_validation.pt'
labels_full = torch.load(LABELS_FULL_PATH).numpy()
labels_val = torch.load(LABELS_VAL_PATH)
trajectories = torch.load('./data/trajectories/fixed_eps_3.0.pt')

def get_labels_dist(labels):
    num_of_samples = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts/num_of_samples
    dist = list(zip(unique, counts, probs))
    dist.sort(key=lambda x: x[1], reverse=True)
    return dist

def plot_tr(tr, title=''):
    plt.scatter(tr[:, 0], tr[:, 1])
    plt.axis('equal')
    plt.title(title)
    plt.show()

def find_in_tuples_list(tlist, item, key):
    for x in tlist:
        if x[key] == item:
            return x
    return None

full_dist = get_labels_dist(labels_full)
val_dist = get_labels_dist(labels_val)

full_dist.sort(key=lambda x: x[2], reverse=True)
full_dist_probs = [ x[2] for x in full_dist ]
cdf = np.cumsum(full_dist_probs)
plt.plot(cdf)
plt.show()
p = 0.95

print(f'There are {(cdf > p).sum()} labels that together make only {((1-p)*100):.0f}% of the dataset.')
print('RARE LABELS:\n', [x[0] for x in full_dist[-(cdf > p).sum():]])
