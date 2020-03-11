import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt


### Evolution Tools ###
def evolve(deriv, state0, length, deriv_params):
    t = np.linspace(0, length, length)
    history = odeint(deriv, state0, t, args=deriv_params)
    return history


### Visualization Tools ###
def show(history, legend=None, title=None, first_t=0, last_t=-1):
    plt.plot(history[first_t : last_t])
    if legend is not None:
        plt.legend(legend)
    if title is not None:
        plt.title(title)

    plt.show()
