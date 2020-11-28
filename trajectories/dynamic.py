import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def car_model(m, t, b, accel_x, accel_y):
    x, y, theta, v = m
    v = np.max([v, 2])
    phi = np.arctan(accel_x*b/v**2)
    dmdt = [v*np.cos(theta), v*np.sin(theta), accel_x/v, accel_y]
    return dmdt

def isin_acceleration_constraint(ax, ay):
    x, y = ay, ax
    if all([x >= -3.5, x < 0, y >= -4/7*x-2, y <= 4/7*x+2]):
        return True
    elif all([x >= 0, x <= 3, y >= 3/2*x-2, y <= -3/2*x+2]):
        return True
    else:
        return False

def get_accelerations(accelerations_shape=(41, 51)):
    accels = []

    n_ax, n_ay = accelerations_shape
    ax_lin = np.linspace(-2, 2, n_ax)
    ay_lin = np.linspace(-3.5, 3, n_ay)

    for ax in ax_lin:
        for ay in ay_lin:
            if isin_acceleration_constraint(ax, ay):
                accels.append([ax, ay])

    return np.array(accels)

def generate_dynamic(velocities=[2, 4, 6, 8, 10], accelerations_shape=(41, 51)):
    b = 3
    
    accels = get_accelerations(accelerations_shape)
    dynamic_trajectories = []

    for ax, ay in accels:
        for v0 in velocities:
            m0 = [0, 0, 0, v0]
            t = np.linspace(0, 5, 51)
            sol = odeint(car_model, m0, t, args=(b, ax, ay))
            dynamic_trajectories.append(sol[1:, :2])

    return dynamic_trajectories

