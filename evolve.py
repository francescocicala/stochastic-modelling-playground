import numpy as np 
import matplotlib.pyplot as plt


### Evolution Tools ###
def evolve(state, deriv, length, **kwargs):
    history = np.zeros((length, len(state)))
    history[0] = state
    for i in range(1, length):
        state = state + deriv(state, **kwargs)
        history[i] = state 

    return history


### Visualization Tools ###
def print_args(**kwargs):
    print("Parameters: ")
    for key, item in kwargs.items():
        print('{} : {}'.format(key, item))


def show(history, legend=None, title=None, first_t=0, last_t=-1, **kwargs):
    plt.plot(history[first_t : last_t])
    if legend is not None:
        plt.legend(legend)
    if title is not None:
        plt.title(title)

    plt.show()
