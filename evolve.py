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
    for key, item in kwargs.items():
        print('{} : {}'.format(key, item))

def show(history, legend=None, title=None, **kwargs):
    print_args(**kwargs)
    plt.plot(history)
    if legend is not None:
        plt.legend(legend)
    if title is not None:
        plt.title(title)
    plt.show()