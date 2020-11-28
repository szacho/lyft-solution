import torch
import matplotlib.pyplot as plt

from dynamic import generate_dynamic
from reduction import reduce_num_trajectories

def plot_trajectories(trajectories):
    for tr in trajectories:
        plt.plot(tr[:, 0], tr[:, 1])
    plt.axis('equal')
    plt.show()

DATA_PATH = './data/trajectories'
EPSILON = 10
SAVE_TRAJECTORIES = False
FIXED_ONLY = True

fixed_trajectories = torch.load(f'{DATA_PATH}/sample_100k.pt')
reduced_fixed = reduce_num_trajectories(fixed_trajectories, fixed_trajectories[0].unsqueeze(0), EPSILON)

if not FIXED_ONLY:
    dynamic_trajectories = generate_dynamic(velocities=[2, 4, 6, 8, 10], accelerations_shape=(31, 41))
    dynamic_trajectories = torch.Tensor(dynamic_trajectories)
    reduced_dynamic = reduce_num_trajectories(dynamic_trajectories, dynamic_trajectories[0].unsqueeze(0), EPSILON)

    final_trajectories = reduce_num_trajectories(reduced_fixed, reduced_dynamic, EPSILON)
else:
    final_trajectories = reduced_fixed


print(f'Generated {final_trajectories.shape[0]} trajectories.')
plot_trajectories(final_trajectories)

if SAVE_TRAJECTORIES:
    filename = f'hybrid_eps_{EPSILON:.1f}.pt' if not FIXED_ONLY else f'fixed_eps_{EPSILON:.1f}.pt'
    torch.save(final_trajectories, f'{DATA_PATH}/{filename}')